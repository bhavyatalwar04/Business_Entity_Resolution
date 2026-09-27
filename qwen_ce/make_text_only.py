"""Qwen3.5 checkpoints ship as vision-language models. Our cross-encoder only reads text, so this writes a text-only
copy: text_config as the config, only model.language_model.* weights (renamed to model.*), no vision tower / MTP head.
usage: python make_text_only.py ~/models/Qwen3.5-4B-Base ~/models/Qwen3.5-4B-Base-text"""
import json, os, shutil, sys

from safetensors import safe_open
from safetensors.torch import save_file

S, D = sys.argv[1], sys.argv[2]
os.makedirs(D, exist_ok=True)
c = json.load(open(f"{S}/config.json"))
t = c["text_config"]
t["architectures"] = ["Qwen3_5ForCausalLM"]
t["tie_word_embeddings"] = c.get("tie_word_embeddings", True)
t["transformers_version"] = c.get("transformers_version")
json.dump(t, open(f"{D}/config.json", "w"), indent=1)
for f in ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]:
    shutil.copy(f"{S}/{f}", f"{D}/{f}")
idx = json.load(open(f"{S}/model.safetensors.index.json"))["weight_map"]
wm, n = {}, 0
for shard in sorted(set(idx.values())):
    out = {}
    with safe_open(f"{S}/{shard}", "pt") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                out["model." + k[len("model.language_model."):]] = f.get_tensor(k)
    n += sum(v.numel() for v in out.values())
    if out:
        save_file(out, f"{D}/{shard}", metadata={"format": "pt"})
        wm.update({k: shard for k in out})
json.dump({"metadata": {}, "weight_map": wm}, open(f"{D}/model.safetensors.index.json", "w"))
print(f"text-only params {n / 1e9:.3f}B -> {D}")
