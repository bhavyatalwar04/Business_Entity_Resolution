"""Gate for test-time canonicalisation of France text (rachit #439/#442, hpc #441, hpc-server #440/#447).

For each canonicaliser variant scored with qwen_score.py --canon (v11 Qwen3 adapter, no training):
  IN/US gate : inputs/gate_fold0_50k.parquet (25k contested + 25k other fold-0 pairs), AUC all + contested vs the RAW
               v11 Qwen3 scores on the same rows. PASS = neither drops by more than 0.001.
  France gate: inputs/gate_fr_50k.parquet (25k confident + 25k contested france_catalogue pairs), mean logit vs RAW.
               PASS = confident pairs' mean logit does not fall AND house-number decoys (contested & house_num_differs)
               do not rise by more than 0.1.
Picks the passing variant with the largest confident-pair logit gain and writes its name to <out>/CHOSEN (or NONE).
usage: python gate_frcanon.py <out_dir> <variant> [<variant> ...]
"""
import os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

out, variants = sys.argv[1], sys.argv[2:]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
gi = pd.read_parquet("inputs/gate_fold0_50k.parquet")
gf = pd.read_parquet("inputs/gate_fr_50k.parquet")
c = gi.contested.values
base = {"all": roc_auc_score(gi.label, gi.q3), "contested": roc_auc_score(gi.label[c], gi.q3[c])}
decoy = (gf.kind != "confident").values & gf.house_num_differs.values
conf = (gf.kind == "confident").values
print(f"RAW v11 Qwen3 | IN/US AUC all {base['all']:.4f} contested {base['contested']:.4f} | FR mean logit confident "
      f"{lg(gf.q3[conf]).mean():.3f} contested {lg(gf.q3[~conf]).mean():.3f} house-number decoys {lg(gf.q3[decoy]).mean():.3f}")
best, best_gain = "NONE", -1e9
for v in variants:
    si = gi.merge(pd.read_parquet(f"{out}/gate_in_{v}_part00.parquet"), on=["s1_id", "pool_id"])
    sf = gf.merge(pd.read_parquet(f"{out}/gate_fr_{v}_part00.parquet"), on=["s1_id", "pool_id"])
    ci = si.contested.values
    a_all, a_con = roc_auc_score(si.label, si.ce_prob), roc_auc_score(si.label[ci], si.ce_prob[ci])
    cf = (sf.kind == "confident").values
    dc = (sf.kind != "confident").values & sf.house_num_differs.values
    d_conf = lg(sf.ce_prob[cf]).mean() - lg(sf.q3[cf]).mean()
    d_cont = lg(sf.ce_prob[~cf]).mean() - lg(sf.q3[~cf]).mean()
    d_decoy = lg(sf.ce_prob[dc]).mean() - lg(sf.q3[dc]).mean()
    changed = float((np.abs(lg(sf.ce_prob) - lg(sf.q3)) > 0.05).mean())
    ok = (a_all >= base["all"] - 0.001) and (a_con >= base["contested"] - 0.001) and d_conf >= 0 and d_decoy <= 0.1
    print(f"{v:10s} | IN/US AUC all {a_all:.4f} ({a_all - base['all']:+.4f}) contested {a_con:.4f} ({a_con - base['contested']:+.4f}) "
          f"| FR logit shift: confident {d_conf:+.3f} contested {d_cont:+.3f} house-number decoys {d_decoy:+.3f} "
          f"| FR pairs changed {changed:.1%} | {'PASS' if ok else 'FAIL'}")
    if ok and d_conf > best_gain:
        best, best_gain = v, d_conf
open(os.path.join(out, "CHOSEN"), "w").write(best)
print("CHOSEN:", best)
