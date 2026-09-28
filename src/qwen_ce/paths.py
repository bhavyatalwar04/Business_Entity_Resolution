"""Default locations for the Qwen cross-encoder scripts (override with environment variables).

WORK    : where inputs/, out*/ and logs/ live (default: this folder; run the scripts from here)
DATA    : challenge data, with train/ and test/ below it (default: <package root>/dataset)
MODELS  : downloaded base models, e.g. MODELS/Qwen3-4B-Base (default: ~/models)
HANDOFF : CE score folders read by the stacker (default: <package root>/handoff)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.environ.get("QWEN_CE_WORK", HERE)
INPUTS = os.path.join(WORK, "inputs")
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "dataset"))
MODELS = os.environ.get("BER_MODELS", os.path.expanduser("~/models"))
HANDOFF = os.environ.get("BER_HANDOFF", os.path.join(ROOT, "handoff"))
