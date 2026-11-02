# Final report outline (BT4222 deliverable guideline, AY2026-27 T1)

Plan for the report: where each number and figure comes from. Paths are under `outputs/full/` after the full run.
Text in *[brackets]* is for the team to write: the guideline says analysis and conclusions must be your own.

**Format:** cover page (topic, group number, members) · Arial / Calibri / Times New Roman 11 · 1.5 line spacing ·
normal margins · single column. Due **Tue 10 Nov, 11:59 PM** (.pdf or .docx on Canvas).

---

## Abstract (own page, ≤ 300 words)

| part | content | source |
|---|---|---|
| Business objective & formulation | Re-rank the items JD displays for a search query so the ones the user clicks, carts or buys come first: query-conditioned learning to rank on the displayed slate | README "The task" |
| Prior baseline & diagnosis | Previous team's Hybrid NCF + LSTM: no query input, cold items, attribute bug, 1:4 sampled lists | `results/prior_ncf.json`, `results/prior_ncf_gaps.json` |
| Contributions | Retention-window features, two-tower model with history segments, cross-fitted LambdaRank, leakage fixes | |
| Quantitative insights | 2-3 headline numbers: ranker vs best baseline vs prior model (test), window-ablation gain, one behavioural insight | `summary_test.csv`, `window_ablation.csv`, `diag_insights.csv` |

---

## Section 1 — Data understanding & feature engineering

Don't repeat the interim report: focus on what changed and what's new.

1. **History timeline (interim feedback).**
   - Reproducible evidence: `notebooks/step1b_history_feedback_eda.ipynb` (four saved plots and full-file counts); source calculations in `scripts/01b_history_feedback_eda.py`.
   - The same 0–7-day window has 6.22 clicks per cart and 6.58 carts per order. These are action-count ratios, not conversion rates. The 107 clicks older than 7 days and 1,204 carts older than 90 days are small exceptions to the observed horizons.
   - 25,432 users have an order but no recorded click; the click fraction correlates 0.953 with the share of each user's history in the last seven days. Explain why older missing clicks are mainly a coverage gap, then show how this changed the features.
2. **What that changed in the features.** Per-type windowed counts; no mixed-type ratios; short/mid/long tower segments.
   - Evidence: `window_ablation.csv` and `window_ablation.png`.
   - *[Interpret the gap between windowed and naive, and the per-segment panel.]*
3. **Impressions and graded labels.**
   - `eda.json → slates`: positives per slate, deepest label, cold items by label.
   - Leakage: `prepare_stats.json → positives_first_in_raw_order` (the raw slate order gives the answer away).
4. **Feature groups table:** group, example features, data field used, window. Take it from README "Feature groups" and the step 2 notebook.

---

## Section 2 — Models and performance (≤ 4 pages)

1. **Architecture diagrams:**
   - pipeline data flow (start from the mermaid chart in the README);
   - the two-tower model: item tower, query + user tower, candidate-aware attention over short history; counts and match flags for all three segments;
   - cross-fitting: which fold model scores whom.
2. **Sample splits:**
   - 80/10/10 split by user, seed 42; only users with a positive are evaluated;
   - 2 folds over training users for out-of-fold CTR and tower scores;
   - training metrics use a 25% subsample of training users.
3. **Performance table:** `summary_all_splits.csv` (train / val / test) and `summary_test.csv`.
   - Metrics: MRR, P@1, NDCG@10 (JDsearch paper), graded NDCG@10, HR@10.
   - Rows: random → popularity → BM25 → mix → prior NCF (published / fixed) → two-tower → two-tower + LambdaRank.
4. **Tower training curves:** `tower/*_history.json`.
5. *[Discuss the train vs test gap and what it says about overfitting.]*

---

## Section 3 — Contribution and justification (1–3 pages per aspect)

Self-evaluation table. Fill in effort and effectiveness yourselves.

| Aspect | Basic | Optional / advanced | Evidence |
|---|---|---|---|
| 1. Data understanding & features | | | Section 1, window ablation |
| 2. Prior baseline & gap diagnosis | | | `prior_ncf.json`, `prior_ncf_gaps.json`, prior NCF row in `diag_segments.csv` |
| 3. Architecture design | | | two-tower with window-aligned segments + cross-fitted LambdaRank; tower ablations `tower_no_history.json`, `tower_no_query.json` |
| 4. Diagnostics & insights | | | `diag_segments.csv`/`.png`, `diag_insights.csv`, `diag_shap*.csv`, `diag_attention.json`, `diag_ablations.csv` |
| 5. Smart use of AI | | | AI interaction log (Section 4) |

**Aspect 2: points for the critique.** Each point has a measured number.
- The query is never an input, but the task is ranking results *for a query*. Compare with the `no_query` tower ablation.
- Cold items get no item embedding: `prior_ncf_gaps.json`, displayed items and positives without one. The prior NCF row in the `positive_items` segment of `diag_segments.csv` shows what this costs.
- The attribute-lookup bug: share of items given another item's attributes, and the published vs fixed rows.
- It trains on 1:4 sampled lists (~5 items) but is tested on ~89-item slates. Their sampled-list "reference" score vs the full-slate score shows how evaluating on sampled lists inflates results.
- Its whole-history click/cart/order ratios mix retention windows. Use `recency_confound` and the "naive" row of the window ablation.
- 4-class cross-entropy, ranked by max class probability (the published variant), vs a ranking loss.

**Aspect 4: where the insights come from.** `diag_insights.csv` gives, for each signal (open cart, repurchase,
recent click, brand or shop loyalty, similar past query, cold item):
- the share of displayed items that have it;
- the positive and order rates with and without it, and the lift;
- the share of all positives and all orders it covers.

*[Turn these into business recommendations, e.g. what a cart-recovery or repurchase module could target.]*

---

## Section 4 — AI interaction log (appendix)

See `report/ai_interaction_log.md`. Keep only substantive entries. The guideline says not to paste generated code.

## Section 5 — References (≤ 1 page)

- Liu, J. et al. (2023). JDsearch: A Personalized Product Search Dataset with Real Queries and Full Interactions. SIGIR '23. <https://github.com/rucliujn/JDsearch>
- Previous team's repository (Hybrid NCF + LSTM). *[add link]*
- He, X. et al. (2017). Neural Collaborative Filtering. WWW '17.
- Covington, P., Adams, J., Sargin, E. (2016). Deep Neural Networks for YouTube Recommendations. RecSys '16.
- Yi, X. et al. (2019). Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. RecSys '19.
- Burges, C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research TR-2010-82.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS '17.
- Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Lundberg, S., Lee, S. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS '17.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning (cross-fitting). Econometrics Journal.
- Krichene, W., Rendle, S. (2020). On Sampled Metrics for Item Recommendation. KDD '20.

---

## Other deliverables

| deliverable | due | status |
|---|---|---|
| Source code on GitHub, notebooks with outputs kept | Mon 9 Nov | run `notebooks/step1…step7` on Colab with `default.yaml`, then commit the executed notebooks |
| Datasets / intermediate files: links | Mon 9 Nov | share `MyDrive/bt4222/` (raw + `outputs/full/data`, `results`, `figures`) as a view-only link |
| Single PDF: verified repo links + file-structure descriptions | Mon 9 Nov | can be built from the README "Repository layout" |
| Slides (.pptx), 10 min, everyone presents | Tue 10 Nov | |
| Report | Tue 10 Nov | |
| Peer evaluation (.docx) | Wed 11 Nov | |
