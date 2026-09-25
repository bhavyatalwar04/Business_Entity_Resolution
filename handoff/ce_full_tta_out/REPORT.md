# Test-time augmentation for ce_full (CE-large continued): swapped record order, logit-averaged with the original

> **DO NOT USE.** Swapped-order TTA hurts: ce 0.9538 vs 0.9824, logit blend 0.9797 vs 0.9875 (held-out, 99,964 S1). The CE was trained only with S1 first, so swapped scores are off by 0.13–0.16 on average. Use handoff/ce_full_out instead.

- ce_oof_fold0 / ce_test: averaged scores (same pairs as handoff/ce_full_out); *_swap: swapped-order scores only
- Code: jobs/ce_tta.py

## Entity-level held-out macro F0.5 (src/blend_eval.py; reference ce_full: ce 0.9824, logit_avg_w0.5 0.9875)
```
fold-0: 99,964 S1 | 684,947 pairs with lgbm_prob >= 0.001 | CE missing on 0
  lgbm             held-out macro F0.5 0.9723 | india 0.9677 | us 0.9754 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  ce               held-out macro F0.5 0.9538 | india 0.9686 | us 0.9440 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  prob_avg_w0.3    held-out macro F0.5 0.9760 | india 0.9747 | us 0.9768 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  logit_avg_w0.3   held-out macro F0.5 0.9818 | india 0.9819 | us 0.9817 | params(all) {'method': 'expf', 'alpha': 1.0, 'one_to_one': True}
  prob_avg_w0.5    held-out macro F0.5 0.9744 | india 0.9748 | us 0.9741 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  logit_avg_w0.5   held-out macro F0.5 0.9797 | india 0.9810 | us 0.9788 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  prob_avg_w0.7    held-out macro F0.5 0.9679 | india 0.9727 | us 0.9647 | params(all) {'method': 'expf', 'alpha': 2.0, 'one_to_one': True}
  logit_avg_w0.7   held-out macro F0.5 0.9732 | india 0.9774 | us 0.9704 | params(all) {'method': 'expf', 'alpha': 5.0, 'one_to_one': True}
  prob_avg_w0.85   held-out macro F0.5 0.9593 | india 0.9702 | us 0.9521 | params(all) {'method': 'expf', 'alpha': 1.5, 'one_to_one': True}
  logit_avg_w0.85  held-out macro F0.5 0.9658 | india 0.9746 | us 0.9601 | params(all) {'method': 'expf', 'alpha': 3.0, 'one_to_one': True}
```
[ce 03:42:45] ce_oof_fold0: 2,755,596 pairs, mean |swap - orig| = 0.1270
[ce 04:15:35] ce_test: 13,535,894 pairs, mean |swap - orig| = 0.1647
