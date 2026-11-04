# Overfitting and match-flag fixes: diagnosis and latest full run

## What was overfitting

The latest code had already fixed the two-tower overfit (train out-of-fold was *below* validation), but the
**LightGBM re-ranker was overfitting**:

| model | train | val | test | train − test |
|---|---|---|---|---|
| two-tower NDCG@10 | 0.1826 (OOF) | 0.1859 | 0.1822 | 0.000 |
| re-ranker NDCG@10 | 0.4678 | 0.4569 | 0.4596 | **0.0082** |
| re-ranker MRR | 0.4720 | 0.4581 | 0.4601 | **0.0119** |

(The previous run had a train − test NDCG gap of **0.032** and MRR gap of **0.042**.)

## Why it happened

1. **Popularity moved into the least regularised part of the pipeline.** Removing popularity from the item tower
   took the tower SHAP contribution from **0.488 → 0.054**, while `ipop_click_7_pct` gain rose from **0.50 M →
   1.57 M**. The same signal re-entered as raw, high-cardinality item-level features that trees can memorise.
2. **Capacity vs label sparsity.** 108 features and 63 leaves over 12.3 M rows, but only ~1.86 positives per slate
   (~255 k positive labels in total). Rare item/CTR keys have very little support per split.
3. **Noisy implicit labels.** 97.7 % of impressions are label 0; clicks are position-biased. Graded gains let the
   ranker chase individual clicks.
4. **Cross-fitting mismatch.** With 2 folds, train users' tower/CTR features came from models trained on 50 % of
   users, while val/test used all training users.

## Fixes applied

- `prepare.n_folds: 2 → 5` — train out-of-fold features now use 80 % of users, closer to the val/test regime.
- `features.ctr_prior: 20 → 50` — stronger shrinkage of rare item-level CTR keys.
- Regularised LambdaRank: `lr 0.03`, `num_leaves 31`, `min_data_in_leaf 1000`, `max_depth 7`,
  `feature_fraction 0.5`, `bagging_fraction 0.7`, `lambda_l2 5`, `path_smooth 10`.
- `07_diagnostics.py` attention step updated for candidate-aware attention (it called the removed per-segment pools).

## Results

### Two-tower (5 folds + full; 2 epochs)

| split | MRR | NDCG@10 |
|---|---|---|
| train (out-of-fold) | 0.2049 | 0.1826 |
| val | 0.2085 | 0.1859 |
| test | 0.2047 | 0.1822 |

Train ≈ val ≈ test: the tower is **not overfitting**. Every fold's validation gNDCG@10 improved monotonically through
epoch 1 (e.g. fold3: 0.1638 → 0.1758 → 0.1829 → 0.1830).

### Re-ranker (train metrics on 25 % of training users)

| split | MRR | P@1 | NDCG@10 | gNDCG@10 | HR@10 | R@10 | R@50 |
|---|---|---|---|---|---|---|---|
| train | 0.4720 | 0.3295 | 0.4678 | 0.4620 | 0.7671 | 0.6454 | 0.9246 |
| val | 0.4581 | 0.3141 | 0.4569 | 0.4513 | 0.7600 | 0.6394 | 0.9225 |
| **test** | **0.4601** | **0.3165** | **0.4596** | **0.4539** | **0.7613** | **0.6418** | **0.9233** |

Against the previous run (v6): test NDCG@10 **0.4615 → 0.4596**, MRR **0.4622 → 0.4601**. The small drop comes
from the category-match fix removing a spurious signal; the train–test gap also fell from **0.0168 → 0.0082**
(NDCG) and **0.0225 → 0.0119** (MRR), so the model is less overfit. The latest numbers include both the
hard-negative leak fix and the match-flag fix.

### Baselines (test)

| model | MRR | NDCG@10 |
|---|---|---|
| Popularity (all history) | 0.3435 | 0.3525 |
| Prior team NCF+LSTM (fixed) | 0.1708 | 0.1644 |
| Two-tower only | 0.2047 | 0.1822 |
| **Two-tower + LambdaRank** | **0.4601** | **0.4596** |

## Category-match fix (v7)

`QueryUserTower.match_flags` compared raw category ids. Padding history slots are item row 0, whose category is
0, so a candidate with an unknown category (`cate_id_1 == 0`) was marked as matching an empty or partially empty
history. The same happened when a real history event had an unknown category.

Fix: product flags require a real candidate (`cand > 0`); category flags require both a known candidate category
and a valid history event with a known category. A regression test covers padding, empty history and unknown
categories. Validation/test remain otherwise unchanged (same features, splits and ranker settings).

## Why 0.9 is not reachable

The test slates average 89.2 items with 1.86 positives; 97.7 % of labels are 0 and 23.6 % of users have no positive
in the model's top 10. Oracle bounds: perfect ordering of the current top-10 gives MRR 0.7736 / NDCG@10 0.6666;
top-50 gives 0.9705 / 0.9365; the whole slate gives 1.0 / 1.0. A 0.9 headline is therefore only available through
metrics that match the sparsity, e.g. **recall@50 = 0.9233** or **HR@50 = 0.970**.

## Reproduce

```bash
.venv/bin/python scripts/00_prepare.py --set paths.work_dir=outputs_v7
.venv/bin/python scripts/02_features.py --set paths.work_dir=outputs_v7
.venv/bin/python scripts/03_train_tower.py --set paths.work_dir=outputs_v7
.venv/bin/python scripts/04_score_tower.py --set paths.work_dir=outputs_v7
.venv/bin/python scripts/05_train_ranker.py --set paths.work_dir=outputs_v7
.venv/bin/python scripts/06_predict.py --set paths.work_dir=outputs_v7
```
