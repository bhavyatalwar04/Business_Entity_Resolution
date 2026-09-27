#!/bin/bash
# Pack a new CE's outputs into handoff/<name> on the ce-qwen worktree, commit as bhavyatalwar04 (no AI lines), push.
# usage: deliver_model.sh <out_dir> <france_scores_dir> <handoff_name> "<model description>"
set -e
Q=~/work/qwen_ce; W=/tmp/claude-156800066/-home-rudra-120437-work/cf9ce0d1-e0a9-44cd-adb6-eb2674cdfae5/scratchpad/wt_ceqwen
PY=~/.conda/envs/minor-project/bin/python
cd $W && git fetch -q origin ce-qwen && git merge -q --ff-only FETCH_HEAD
cd $Q && nice $PY pack_model.py "$1" "$2" "$W/handoff/$3" "$4" | grep -v '^ *"' || true
cd $W
git add -f handoff/$3
FR=$([ -f handoff/$3/ce_test_france.DONE ] && echo " + all France pairs" || echo "")
TS=$([ -f handoff/$3/ce_test.DONE ] && echo " + contested test" || echo "")
git -c user.name=bhavyatalwar04 -c user.email=bhavyatalwar.bt@gmail.com commit -q -m "$4: fold-0 scores$TS$FR (handoff/$3)"
git push -q origin HEAD:ce-qwen
echo "PUSHED ce-qwen $(git rev-parse --short HEAD) handoff/$3"

# ping the room with the commit hash and the fold-0 comparison vs Qwen3 (laptop #361)
H=$(git rev-parse --short HEAD)
MSG=$(python3 - "$W/handoff/$3" "$H" "$3" "$4" <<'PY'
import json, os, sys
d, h, name, desc = sys.argv[1:5]
m = json.load(open(os.path.join(d, "fold0_metrics.json")))
fr = "yes" if os.path.exists(os.path.join(d, "ce_test_france.DONE")) else "NOT YET"
ts = "yes" if os.path.exists(os.path.join(d, "ce_test.DONE")) else "NOT YET"
t = (f"hpc-server AUTO-PUSH: {desc} -> ce-qwen {h}, handoff/{name} (ce_oof_fold0: yes; ce_test contested: {ts}; France-all: {fr}; .DONE markers set). "
     "Same fold-0 rows vs Qwen3 (v11): " + " | ".join(
         f"{k} AUC {m[k]['auc']:.4f} vs {m[k]['qwen3_auc']:.4f}, logloss {m[k]['logloss']:.4f} vs {m[k]['qwen3_logloss']:.4f}"
         for k in ("all", "handoff", "contested")) +
     f". ENSEMBLE (logit-average with v11's Qwen3), contested AUC {m['contested'].get('avg_with_qwen3_auc', float('nan')):.4f} "
     f"(single {m['contested']['auc']:.4f}, Qwen3 {m['contested']['qwen3_auc']:.4f}); contested logit corr {m['contested'].get('logit_corr_with_qwen3', float('nan')):.3f}")
print(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "send_message", "arguments": {"to": "all", "text": t}}}))
PY
)
curl -s --max-time 20 -X POST https://claude-collab-eta.vercel.app/api/mcp -H "content-type: application/json" \
     -H "accept: application/json, text/event-stream" -H "X-Room-Key: $COLLAB_ROOM_KEY" -H "X-Agent: hpc-server" \
     -d "$MSG" | grep -o 'Sent message #[0-9]*' || echo "room post failed"
