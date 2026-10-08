/**
 * DermaSense — Frontend Interactive Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  const state = {
    skinType: null,
    concerns: new Set(),
    budget: 60,
    category: 'All',
    preset: 'balanced',
    comparisonList: new Map(), // product_id -> product_object
  };

  // DOM Elements
  const skinTypeBtns = document.querySelectorAll('.skin-type-btn');
  const concernPills = document.querySelectorAll('.concern-pill');
  const budgetSlider = document.getElementById('budgetSlider');
  const budgetDisplay = document.getElementById('budgetDisplay');
  const categorySelect = document.getElementById('categorySelect');
  const presetSelect = document.getElementById('presetSelect');
  const queryInput = document.getElementById('queryInput');
  const btnGetRecommendations = document.getElementById('btnGetRecommendations');
  const resultsSection = document.getElementById('resultsSection');
  const productsGrid = document.getElementById('productsGrid');
  const resultsCount = document.getElementById('resultsCount');

  // Navigation Tabs
  const navTabBtns = document.querySelectorAll('.nav-tab-btn');
  const tabViews = document.querySelectorAll('.tab-view');

  // Comparison Elements
  const compareTray = document.getElementById('compareTray');
  const trayCount = document.getElementById('trayCount');
  const btnLaunchTrayCompare = document.getElementById('btnLaunchTrayCompare');
  const compareSearchA = document.getElementById('compareSearchA');
  const compareSearchB = document.getElementById('compareSearchB');
  const compareIdA = document.getElementById('compareIdA');
  const compareIdB = document.getElementById('compareIdB');
  const compareSuggestionsA = document.getElementById('compareSuggestionsA');
  const compareSuggestionsB = document.getElementById('compareSuggestionsB');
  const btnRunComparison = document.getElementById('btnRunComparison');
  const comparisonResults = document.getElementById('comparisonResults');

  // Ingredient Dictionary
  const ingredientsCatalog = document.getElementById('ingredientsCatalog');

  /* --------------------------------------------------------------------------
     1. Navigation Tabs
     -------------------------------------------------------------------------- */
  navTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navTabBtns.forEach(b => b.classList.remove('active'));
      tabViews.forEach(v => v.classList.remove('active'));

      btn.classList.add('active');
      const targetTabId = btn.getAttribute('data-tab');
      const targetView = document.getElementById(targetTabId);
      if (targetView) targetView.classList.add('active');

      if (targetTabId === 'tab-ingredients' && ingredientsCatalog.children.length === 0) {
        loadIngredientDictionary();
      }
    });
  });

  /* --------------------------------------------------------------------------
     2. Skin Profile Intake Handlers
     -------------------------------------------------------------------------- */
  skinTypeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      skinTypeBtns.forEach(b => b.classList.remove('active'));
      const type = btn.getAttribute('data-type');
      if (state.skinType === type) {
        state.skinType = null;
      } else {
        btn.classList.add('active');
        state.skinType = type;
      }
    });
  });

  concernPills.forEach(pill => {
    pill.addEventListener('click', () => {
      const concern = pill.getAttribute('data-concern');
      if (state.concerns.has(concern)) {
        state.concerns.delete(concern);
        pill.classList.remove('active');
      } else {
        state.concerns.add(concern);
        pill.classList.add('active');
      }
    });
  });

  budgetSlider.addEventListener('input', (e) => {
    state.budget = Number(e.target.value);
    budgetDisplay.textContent = `$${state.budget}`;
  });

  categorySelect.addEventListener('change', (e) => {
    state.category = e.target.value;
  });

  presetSelect.addEventListener('change', (e) => {
    state.preset = e.target.value;
  });

  /* --------------------------------------------------------------------------
     3. Fetch Recommendations
     -------------------------------------------------------------------------- */
  btnGetRecommendations.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    
    // UI Loading state
    resultsSection.style.display = 'block';
    productsGrid.innerHTML = `
      <div class="loading-spinner" style="grid-column: 1 / -1;">
        🌸 AI is formulating your personalized recommendations...
      </div>
    `;
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    try {
      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          skin_type: state.skinType,
          target_concerns: Array.from(state.concerns),
          category: state.category,
          max_price: state.budget,
          preset: state.preset,
          top_k: 6
        })
      });

      const data = await response.json();
      if (data.status === 'success') {
        renderRecommendations(data.recommendations);
      } else {
        productsGrid.innerHTML = `<p style="grid-column: 1 / -1; text-align: center; color: var(--primary-rose);">Error: ${data.message}</p>`;
      }
    } catch (err) {
      productsGrid.innerHTML = `<p style="grid-column: 1 / -1; text-align: center; color: var(--primary-rose);">Unable to fetch recommendations. Please try again.</p>`;
    }
  });

  /* --------------------------------------------------------------------------
     4. Render Recommendation Cards & Explainability Accordion
     -------------------------------------------------------------------------- */
  function renderRecommendations(products) {
    if (!products || products.length === 0) {
      productsGrid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 40px;">
          <p style="font-size: 1.1rem; color: var(--text-secondary);">No matching products found. Try adjusting your budget or selecting fewer filters.</p>
        </div>
      `;
      resultsCount.textContent = '0 matches';
      return;
    }

    resultsCount.textContent = `Showing top ${products.length} matches`;
    productsGrid.innerHTML = '';

    products.forEach((prod, idx) => {
      const card = document.createElement('article');
      card.className = 'product-card';
      
      const priceText = prod.price_usd ? `$${prod.price_usd.toFixed(2)}` : 'Check Retailer';
      const isSelected = state.comparisonList.has(prod.product_id);
      
      // Skin type badge
      let stBadge = '';
      if (state.skinType && prod.skin_type_metrics.rating) {
        stBadge = `<span class="skin-type-badge">🌿 ${prod.skin_type_metrics.rating.toFixed(1)}★ for ${capitalize(state.skinType)} Skin (${prod.skin_type_metrics.review_count} revs)</span>`;
      }

      // Benefits badges
      const benefitTags = (prod.matched_benefits || []).slice(0, 3).map(b => `<span class="badge-benefit">${escapeHtml(b)}</span>`).join('');
      const activeTags = (prod.matched_actives || []).slice(0, 3).map(a => `<span class="badge-active">🔬 ${escapeHtml(capitalize(a))}</span>`).join('');

      card.innerHTML = `
        <div>
          <div class="card-top-bar">
            <span class="product-brand">${escapeHtml(prod.brand || 'Skincare')}</span>
            <span class="product-category-badge">${escapeHtml(prod.category || 'Product')}</span>
          </div>
          <h3 class="product-title">${escapeHtml(prod.product_name)}</h3>
          
          <div class="ratings-row">
            <span class="star-rating">★ ${prod.rating.toFixed(1)}</span>
            <span style="font-size: 0.8rem; color: var(--text-muted);">(${prod.review_count.toLocaleString()} reviews)</span>
            ${stBadge}
          </div>

          <div class="price-tag">${priceText}</div>

          <div class="badges-row">
            ${benefitTags}
            ${activeTags}
          </div>
        </div>

        <div>
          <div class="card-actions">
            <button type="button" class="btn-explain" data-id="${prod.product_id}">
              <span>💡</span> Why This?
            </button>
            <button type="button" class="btn-compare-toggle ${isSelected ? 'selected' : ''}" data-id="${prod.product_id}">
              ${isSelected ? '✓ In Compare' : '+ Compare'}
            </button>
          </div>

          <!-- Explainability Expander Panel -->
          <div class="explain-panel" id="explain-${prod.product_id}">
            <h4 class="explain-section-title">Why it matches your profile</h4>
            <ul class="explain-list">
              <li>Formulated as a <strong>${escapeHtml(prod.category)}</strong> for target skin compatibility.</li>
              ${(prod.explanation.matched_concerns || []).length > 0 ? `<li>Directly targets your concerns: <strong>${escapeHtml(prod.explanation.matched_concerns.join(', '))}</strong>.</li>` : ''}
            </ul>

            ${(prod.explanation.ingredient_rationales || []).length > 0 ? `
              <h4 class="explain-section-title">Key Active Ingredients</h4>
              <ul class="explain-list">
                ${prod.explanation.ingredient_rationales.map(r => `<li>${formatMarkdownSnippet(r)}</li>`).join('')}
              </ul>
            ` : ''}

            ${(prod.explanation.sentiment_highlights || []).length > 0 ? `
              <h4 class="explain-section-title">Customer Sentiment Highlights</h4>
              <ul class="explain-list">
                ${prod.explanation.sentiment_highlights.map(h => `<li>${formatMarkdownSnippet(h)}</li>`).join('')}
              </ul>
            ` : ''}
          </div>
        </div>
      `;

      // Event listener for Explainability expander
      const btnExplain = card.querySelector('.btn-explain');
      const explainPanel = card.querySelector('.explain-panel');
      btnExplain.addEventListener('click', () => {
        explainPanel.classList.toggle('open');
        btnExplain.classList.toggle('active');
      });

      // Event listener for Compare toggle
      const btnCompare = card.querySelector('.btn-compare-toggle');
      btnCompare.addEventListener('click', () => {
        toggleCompareProduct(prod, btnCompare);
      });

      productsGrid.appendChild(card);
    });
  }

  /* --------------------------------------------------------------------------
     5. Comparison Tray Management
     -------------------------------------------------------------------------- */
  function toggleCompareProduct(prod, btn) {
    if (state.comparisonList.has(prod.product_id)) {
      state.comparisonList.delete(prod.product_id);
      btn.classList.remove('selected');
      btn.textContent = '+ Compare';
    } else {
      if (state.comparisonList.size >= 2) {
        alert("You can compare up to 2 products at a time. Remove one first or compare existing.");
        return;
      }
      state.comparisonList.set(prod.product_id, prod);
      btn.classList.add('selected');
      btn.textContent = '✓ In Compare';
    }

    updateCompareTray();
  }

  function updateCompareTray() {
    const count = state.comparisonList.size;
    trayCount.textContent = count;
    if (count > 0) {
      compareTray.classList.add('visible');
    } else {
      compareTray.classList.remove('visible');
    }
  }

  btnLaunchTrayCompare.addEventListener('click', () => {
    const ids = Array.from(state.comparisonList.keys());
    if (ids.length === 2) {
      const prodA = state.comparisonList.get(ids[0]);
      const prodB = state.comparisonList.get(ids[1]);

      compareIdA.value = ids[0];
      compareSearchA.value = `${prodA.product_name} (${prodA.brand})`;
      compareIdB.value = ids[1];
      compareSearchB.value = `${prodB.product_name} (${prodB.brand})`;

      // Switch to Comparison Tab
      document.getElementById('tabBtnCompare').click();
      btnRunComparison.click();
    } else {
      alert("Please select 2 products to compare head-to-head.");
    }
  });

  /* --------------------------------------------------------------------------
     6. Side-by-Side Comparison Engine Handler
     -------------------------------------------------------------------------- */
  btnRunComparison.addEventListener('click', async () => {
    const idA = compareIdA.value;
    const idB = compareIdB.value;

    if (!idA || !idB) {
      alert("Please select both Product A and Product B to compare.");
      return;
    }

    comparisonResults.style.display = 'block';
    comparisonResults.innerHTML = '<div class="loading-spinner">⚖️ Analyzing active chemical overlap and review satisfaction...</div>';

    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          product_id_a: idA,
          product_id_b: idB,
          skin_type: state.skinType,
          target_concerns: Array.from(state.concerns)
        })
      });

      const data = await response.json();
      if (data.status === 'success') {
        renderComparison(data.comparison);
      } else {
        comparisonResults.innerHTML = `<p style="text-align: center; color: var(--primary-rose);">${data.message}</p>`;
      }
    } catch (err) {
      comparisonResults.innerHTML = '<p style="text-align: center; color: var(--primary-rose);">Unable to run comparison.</p>';
    }
  });

  function renderComparison(comp) {
    const pA = comp.product_a;
    const pB = comp.product_b;

    const sharedActives = comp.shared_actives.length > 0
      ? comp.shared_actives.map(a => `<span class="badge-active" style="background: var(--sage-green-light); color: var(--sage-green-dark);">🌿 ${capitalize(a)}</span>`).join(' ')
      : '<span style="color: var(--text-muted); font-size: 0.85rem;">None shared</span>';

    const uniqueA = comp.unique_actives_a.length > 0
      ? comp.unique_actives_a.map(a => `<span class="badge-active">🔬 ${capitalize(a)}</span>`).join(' ')
      : '<span style="color: var(--text-muted); font-size: 0.85rem;">None unique</span>';

    const uniqueB = comp.unique_actives_b.length > 0
      ? comp.unique_actives_b.map(a => `<span class="badge-active">🔬 ${capitalize(a)}</span>`).join(' ')
      : '<span style="color: var(--text-muted); font-size: 0.85rem;">None unique</span>';

    comparisonResults.innerHTML = `
      <div class="comparison-grid">
        
        <!-- Product A -->
        <div class="comparison-card">
          <div class="product-brand">${escapeHtml(pA.brand)}</div>
          <h3 class="product-title" style="font-size: 1.3rem;">${escapeHtml(pA.product_name)}</h3>
          <div class="price-tag">$${pA.price_usd ? pA.price_usd.toFixed(2) : 'N/A'}</div>
          
          <div style="margin: 14px 0;">
            <strong>Overall Rating:</strong> ★ ${comp.rating_a ? comp.rating_a.toFixed(2) : '4.2'} (${comp.reviews_a} reviews)
          </div>

          <div style="margin: 14px 0;">
            <strong style="display: block; margin-bottom: 6px;">Unique Actives:</strong>
            ${uniqueA}
          </div>
        </div>

        <!-- Product B -->
        <div class="comparison-card">
          <div class="product-brand">${escapeHtml(pB.brand)}</div>
          <h3 class="product-title" style="font-size: 1.3rem;">${escapeHtml(pB.product_name)}</h3>
          <div class="price-tag">$${pB.price_usd ? pB.price_usd.toFixed(2) : 'N/A'}</div>

          <div style="margin: 14px 0;">
            <strong>Overall Rating:</strong> ★ ${comp.rating_b ? comp.rating_b.toFixed(2) : '4.2'} (${comp.reviews_b} reviews)
          </div>

          <div style="margin: 14px 0;">
            <strong style="display: block; margin-bottom: 6px;">Unique Actives:</strong>
            ${uniqueB}
          </div>
        </div>

      </div>

      <div class="card" style="margin-top: 24px;">
        <h4 style="font-family: var(--font-serif); font-size: 1.15rem; color: var(--primary-rose); margin-bottom: 8px;">Shared Active Ingredients</h4>
        <div>${sharedActives}</div>
      </div>

      <div class="verdict-box">
        <div class="verdict-title">🌟 Algorithmic Recommendation Verdict</div>
        <p style="font-size: 1rem; color: var(--text-primary); max-width: 700px; margin: 0 auto;">
          ${formatMarkdownSnippet(comp.verdict)}
        </p>
      </div>
    `;
  }

  /* --------------------------------------------------------------------------
     7. Autocomplete Search for Comparison Inputs
     -------------------------------------------------------------------------- */
  function setupAutocomplete(inputEl, idEl, dropdownEl) {
    let timeout;
    inputEl.addEventListener('input', () => {
      clearTimeout(timeout);
      const val = inputEl.value.trim();
      if (val.length < 2) {
        dropdownEl.innerHTML = '';
        return;
      }

      timeout = setTimeout(async () => {
        try {
          const res = await fetch(`/api/search?q=${encodeURIComponent(val)}&limit=6`);
          const data = await res.json();
          if (data.results && data.results.length > 0) {
            dropdownEl.innerHTML = data.results.map(r => `
              <div class="suggestion-item" data-id="${r.product_id}" data-title="${escapeHtml(r.product_name)} (${escapeHtml(r.brand)})" style="padding: 8px 12px; cursor: pointer; border-bottom: 1px solid var(--border-soft); font-size: 0.88rem;">
                <strong>${escapeHtml(r.product_name)}</strong> <span style="color: var(--primary-rose); font-size: 0.8rem;">(${escapeHtml(r.brand)})</span>
              </div>
            `).join('');

            dropdownEl.querySelectorAll('.suggestion-item').forEach(item => {
              item.addEventListener('click', () => {
                idEl.value = item.getAttribute('data-id');
                inputEl.value = item.getAttribute('data-title');
                dropdownEl.innerHTML = '';
              });
            });
          } else {
            dropdownEl.innerHTML = '';
          }
        } catch (e) { }
      }, 250);
    });
  }

  setupAutocomplete(compareSearchA, compareIdA, compareSuggestionsA);
  setupAutocomplete(compareSearchB, compareIdB, compareSuggestionsB);

  /* --------------------------------------------------------------------------
     8. Ingredient Dictionary Fetcher
     -------------------------------------------------------------------------- */
  async function loadIngredientDictionary() {
    ingredientsCatalog.innerHTML = '<div class="loading-spinner">🌿 Loading active ingredient database...</div>';
    try {
      const response = await fetch('/api/ingredients');
      const data = await response.json();
      
      ingredientsCatalog.innerHTML = '';
      for (const [category, items] of Object.entries(data.categories)) {
        const catCard = document.createElement('div');
        catCard.className = 'ingredient-category-card';
        
        const itemsHtml = items.map(item => `
          <div class="ingredient-item">
            <div class="ingredient-item-name">🔬 ${escapeHtml(item.name)}</div>
            <div class="ingredient-item-desc">${escapeHtml(item.description)}</div>
          </div>
        `).join('');

        catCard.innerHTML = `
          <h3 class="ingredient-category-title">💧 ${escapeHtml(category)}</h3>
          <div>${itemsHtml}</div>
        `;
        ingredientsCatalog.appendChild(catCard);
      }
    } catch (e) {
      ingredientsCatalog.innerHTML = '<p style="color: var(--primary-rose);">Unable to load ingredient database.</p>';
    }
  }

  /* --------------------------------------------------------------------------
     9. Utility Helpers
     -------------------------------------------------------------------------- */
  function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  function escapeHtml(text) {
    if (!text) return '';
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatMarkdownSnippet(text) {
    if (!text) return '';
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  }

  // Load initial recommendation example on launch
  btnGetRecommendations.click();
});
