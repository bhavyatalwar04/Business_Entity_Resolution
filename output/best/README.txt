output/best = v8_full_all (built 2026-09-26 05:33 IST)
Stacker trained on 434,574 full-data fold-0 S1 (complete candidate competition, complete CE coverage): LightGBM full + CE base + CE large + CE large retrained on full data (ce_full); France rows use the France self-trained CE (round 1) in the ce_large slot; France logit -1; expf alpha 1.5.
Held-out India/US macro F0.5: 0.9900 on 434,574 S1; 0.9899 on the 99,964 S1 that v5 (0.9877) and v7a (0.9895) were measured on.
Must beat: v5 public 0.983546.
A/B partner: output/v8b_full_nofrst (same without France self-training; India/US identical).
