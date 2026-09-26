# xlm-roberta-XL (3.5B, MIT) cross-encoder: 1M hard full-data pairs, lr 1e-5, bs 32; scores = all handoff fold-0 + contested (0.02-0.98) step-3 fold-0 and test pairs

```
[ce 12:43:22] train sample: core 10,490,256 (pos + prob>=0.001), random fill 1,659,299 -> using 1,000,000
[ce 12:43:25] fold-0 list: step-3 264,628 + handoff 684,947 -> union 890,838
[ce 12:43:29] train pairs 1,000,000 (pos 577,968)
[ce 12:43:45] steps 31,250 (31,250/epoch), bs 32, lr 1e-05
[ce 14:50:24] test list: step-3 10,797,489 + handoff 13,047,424 -> union 1,269,434
```
## fold0_metrics.json
```
{
 "n_union": 890838,
 "n_step3_rows": 264628,
 "pos_step3_rows": 83818,
 "ce_full_on_step3_rows": {
  "auc": 0.8730860133128575,
  "logloss": 0.4186660941893106
 },
 "step3_lgbm_on_step3_rows": {
  "auc": 0.887356057098908,
  "logloss": 0.3865707260131875
 },
 "minutes_so_far": 126.3
}```
## Entity-level held-out macro F0.5 (src/blend_eval.py, 99,964 handoff fold-0 S1; ref ce_full 0.9824 / blend 0.9875)
```
fold-0: 99,964 S1 | 684,947 pairs with lgbm_prob >= 0.001 | CE missing on 0
  lgbm             held-out macro F0.5 0.9723 | india 0.9677 | us 0.9754 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  ce               held-out macro F0.5 0.9805 | india 0.9824 | us 0.9792 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.3    held-out macro F0.5 0.9802 | india 0.9792 | us 0.9810 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  logit_avg_w0.3   held-out macro F0.5 0.9849 | india 0.9854 | us 0.9846 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.5    held-out macro F0.5 0.9830 | india 0.9829 | us 0.9831 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  logit_avg_w0.5   held-out macro F0.5 0.9865 | india 0.9868 | us 0.9862 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.7    held-out macro F0.5 0.9845 | india 0.9852 | us 0.9840 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.7   held-out macro F0.5 0.9856 | india 0.9859 | us 0.9853 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.85   held-out macro F0.5 0.9830 | india 0.9844 | us 0.9821 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  logit_avg_w0.85  held-out macro F0.5 0.9837 | india 0.9844 | us 0.9833 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
```
