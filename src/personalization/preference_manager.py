"""
DermaSense - Feedback-Driven Adaptive Personalization Engine
Features:
1. Dynamic User Preference Profile tracking (ingredient affinities, brand affinities, texture preferences, price sensitivity).
2. Explicit feedback (thumbs up/down, star ratings, irritation/breakout reports) and implicit feedback (routine additions, comparisons).
3. Hard and soft ingredient avoidance (penalizing or excluding products containing user-flagged irritants).
4. Adaptive score modulation blending smoothly with the Stage-2 Cross-Encoder recommender.
5. Persistent JSON storage with seamless cold-start fallback.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROFILES_FILE = os.path.join(PROJECT_ROOT, "data", "user_profiles.json")


class AdaptivePersonalizer:
    """
    Manages persistent user profiles and computes adaptive personalization weights.
    """
    _instance = None

    def __init__(self, storage_path: str = PROFILES_FILE):
        self.storage_path = storage_path
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self._load_profiles()

    @classmethod
    def get_instance(cls, storage_path: str = PROFILES_FILE):
        if cls._instance is None:
            cls._instance = cls(storage_path=storage_path)
        return cls._instance

    def _load_profiles(self):
        """Loads user profiles from JSON storage if exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
            except Exception:
                self.profiles = {}
        else:
            self.profiles = {}

    def _save_profiles(self):
        """Persists user profiles to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, indent=2)
        except Exception:
            pass

    def get_or_create_profile(self, user_id: str, skin_type: Optional[str] = None, concerns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Returns existing user profile or initializes a default cold-start profile."""
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                'user_id': user_id,
                'created_at': time.time(),
                'updated_at': time.time(),
                'skin_type': skin_type,
                'concerns': concerns or [],
                'ingredient_affinities': {},   # active -> float (-1.0 to +1.0)
                'disliked_ingredients': [],    # hard avoidance list
                'liked_ingredients': [],       # favorites list
                'preferred_brands': {},        # brand -> weight
                'disliked_brands': [],
                'interaction_count': 0,
                'feedback_history': []
            }
            self._save_profiles()
        else:
            if skin_type:
                self.profiles[user_id]['skin_type'] = skin_type
            if concerns:
                self.profiles[user_id]['concerns'] = concerns
        return self.profiles[user_id]

    def record_feedback(
        self,
        user_id: str,
        product_id: str,
        feedback_type: str,            # 'like', 'dislike', 'tried_loved', 'tried_hated', 'routine_am', 'routine_pm', 'compare'
        rating: Optional[float] = None,
        reason: Optional[str] = None,  # 'caused breakouts', 'too drying', 'loved texture', 'too heavy', 'overpriced'
        brand: Optional[str] = None,
        actives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Updates dynamic user preferences based on explicit and implicit feedback.
        """
        profile = self.get_or_create_profile(user_id)
        profile['interaction_count'] += 1
        profile['updated_at'] = time.time()

        actives = [a.lower().strip() for a in (actives or [])]

        # 1. Update Ingredient Affinities
        affinities = profile['ingredient_affinities']
        disliked_set = set(profile['disliked_ingredients'])
        liked_set = set(profile['liked_ingredients'])

        # Explicit Positive
        if feedback_type in ['like', 'tried_loved'] or (rating and rating >= 4):
            delta = 0.25 if feedback_type == 'tried_loved' else 0.15
            for a in actives:
                affinities[a] = min(1.0, affinities.get(a, 0.0) + delta)
                liked_set.add(a)
                disliked_set.discard(a)
            if brand:
                profile['preferred_brands'][brand] = profile['preferred_brands'].get(brand, 0) + 1

        # Implicit Positive (Added to Routine)
        elif feedback_type in ['routine_am', 'routine_pm']:
            for a in actives:
                affinities[a] = min(1.0, affinities.get(a, 0.0) + 0.10)
                liked_set.add(a)
            if brand:
                profile['preferred_brands'][brand] = profile['preferred_brands'].get(brand, 0) + 1

        # Explicit Negative (Breakouts / Irritation / Dislike)
        elif feedback_type in ['dislike', 'tried_hated'] or (rating and rating <= 2) or (reason in ['caused breakouts', 'too drying']):
            penalty = 0.50 if (reason in ['caused breakouts', 'too drying'] or feedback_type == 'tried_hated') else 0.25
            for a in actives:
                affinities[a] = max(-1.0, affinities.get(a, 0.0) - penalty)
                if affinities[a] <= -0.4:
                    disliked_set.add(a)
                    liked_set.discard(a)
            if brand and reason == 'overpriced':
                profile['disliked_brands'].append(brand)

        profile['disliked_ingredients'] = sorted(list(disliked_set))
        profile['liked_ingredients'] = sorted(list(liked_set))

        # 2. Append to history log
        profile['feedback_history'].append({
            'timestamp': time.time(),
            'product_id': product_id,
            'feedback_type': feedback_type,
            'rating': rating,
            'reason': reason,
            'brand': brand,
            'actives': actives
        })

        self._save_profiles()
        return profile

    def compute_personalization_multiplier(self, user_id: str, row: Any) -> float:
        """
        Computes an adaptive personalized multiplier for a candidate recommendation.
        Returns multiplier in range [0.4, 1.4]:
        - 1.0 for cold-start (unbiased)
        - > 1.0 for products matching learned positive affinities and favorite brands
        - < 1.0 for products with negative affinities or disliked ingredients
        """
        if user_id not in self.profiles:
            return 1.0

        profile = self.profiles[user_id]
        if profile['interaction_count'] == 0:
            return 1.0

        p_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

        def _parse(val):
            if isinstance(val, str):
                try: return json.loads(val)
                except: return []
            return val if isinstance(val, list) else []

        actives = [a.lower().strip() for a in _parse(p_dict.get('matched_actives'))]
        brand = str(p_dict.get('brand', ''))

        # Check Hard Avoidance: If product contains an ingredient user flagged for breakouts/distress
        disliked_set = set(profile.get('disliked_ingredients', []))
        clashing_dislikes = [a for a in actives if a in disliked_set]
        if clashing_dislikes:
            # Significant penalty for containing disliked ingredients
            return max(0.40, 0.80 - (0.15 * len(clashing_dislikes)))

        # Soft Multiplier based on learned affinities
        affinities = profile.get('ingredient_affinities', {})
        score_mod = 0.0

        for a in actives:
            score_mod += affinities.get(a, 0.0) * 0.15

        # Brand preference bonus
        pref_brands = profile.get('preferred_brands', {})
        if brand in pref_brands:
            score_mod += min(0.15, pref_brands[brand] * 0.05)

        multiplier = 1.0 + score_mod
        return max(0.60, min(1.35, multiplier))

    def reset_profile(self, user_id: str):
        """Resets user profile back to cold-start baseline."""
        if user_id in self.profiles:
            del self.profiles[user_id]
            self._save_profiles()
