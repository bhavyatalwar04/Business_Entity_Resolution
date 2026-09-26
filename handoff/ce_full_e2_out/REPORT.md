# CE-large epoch 2 (continued on 6.5M unseen full-data pairs, lr 3e-6)

```
[ce 12:05:26] epoch 2: first run used 4,000,000 rows; 39,668,971 rows unseen
[ce 12:05:29] train sample: core 6,490,256 (pos + prob>=0.001), random fill 1,657,021 -> using 6,600,000
[ce 12:05:31] fold-0 list: step-3 2,635,403 + handoff 684,947 -> union 2,755,596
[ce 12:05:34] train pairs 6,600,000 (pos 3,751,255)
[ce 12:05:48] steps 51,563 (51,563/epoch), bs 128, lr 3e-06
[ce 16:11:47] test list: step-3 10,797,489 + handoff 13,047,424 -> union 13,535,894
```
## fold0_metrics.json
```
{
 "n_union": 2755596,
 "n_step3_rows": 2635403,
 "pos_step3_rows": 1515931,
 "ce_full_on_step3_rows": {
  "auc": 0.9981954978268459,
  "logloss": 0.04765689035743341
 },
 "step3_lgbm_on_step3_rows": {
  "auc": 0.996289284527138,
  "logloss": 0.06402155206681674
 },
 "minutes_so_far": 246.0
}```
## Entity-level held-out macro F0.5 (src/blend_eval.py, 99,964 handoff fold-0 S1; ref ce_full 0.9824 / blend 0.9875)
```
fold-0: 99,964 S1 | 684,947 pairs with lgbm_prob >= 0.001 | CE missing on 0
  lgbm             held-out macro F0.5 0.9723 | india 0.9677 | us 0.9754 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  ce               held-out macro F0.5 0.9828 | india 0.9843 | us 0.9817 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  prob_avg_w0.3    held-out macro F0.5 0.9808 | india 0.9795 | us 0.9816 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  logit_avg_w0.3   held-out macro F0.5 0.9868 | india 0.9872 | us 0.9866 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.5    held-out macro F0.5 0.9839 | india 0.9839 | us 0.9840 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  logit_avg_w0.5   held-out macro F0.5 0.9878 | india 0.9882 | us 0.9876 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.7    held-out macro F0.5 0.9860 | india 0.9865 | us 0.9856 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.7   held-out macro F0.5 0.9871 | india 0.9873 | us 0.9870 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.85   held-out macro F0.5 0.9849 | india 0.9860 | us 0.9842 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
  logit_avg_w0.85  held-out macro F0.5 0.9855 | india 0.9862 | us 0.9851 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
```
