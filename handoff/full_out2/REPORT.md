# LightGBM ranker v2: lr 0.05, n_estimators 8000 (configs/full2.yaml); features/candidates identical to full-results

```
[rank:train] 54,593,626 pairs, 7,578,192 positives, 2,206,821 S1
  fold 0: best_iter 8000, val logloss 0.0163
  fold 1: best_iter 7996, val logloss 0.0164
  fold 2: best_iter 7986, val logloss 0.0164
  fold 3: best_iter 8000, val logloss 0.0164
  fold 4: best_iter 7999, val logloss 0.0164
  decision params {'method': 'expf', 'alpha': 3.0, 'one_to_one': True} -> macro F0.5 0.9797
[rank:train] stage1 OOF macro F0.5 = 0.9797 on 300,000 S1
  fold 0: best_iter 1557, val logloss 0.0154
  fold 1: best_iter 1356, val logloss 0.0155
  fold 2: best_iter 1351, val logloss 0.0155
  fold 3: best_iter 1564, val logloss 0.0155
  fold 4: best_iter 1256, val logloss 0.0155
  decision params {'method': 'expf', 'alpha': 1.5, 'one_to_one': True} -> macro F0.5 0.9808
[rank:train] stage2 OOF macro F0.5 = 0.9808
[export] 1,732,544 S1 rows, 5.6% empty, 3.39 matches per S1
PASS — no blocking issues found. Safe to submit.
```
