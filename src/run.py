"""CLI entry point: python -m src.run --stage all --split test

Every stage caches its output under artefacts/<split>/ so a single stage
(e.g. --stage decide) can be rerun without redoing the ones before it.
"""
import argparse

STAGES = ["normalize", "block", "features", "cross_encoder", "rank", "decide", "export"]


def parse_args():
    """Parse --stage, --split and --config command-line options."""
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    parser.add_argument("--stage", default="all", choices=["all", *STAGES])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main():
    """Run the requested pipeline stage(s) for the requested split."""
    args = parse_args()
    raise NotImplementedError(f"stage={args.stage} split={args.split}")


if __name__ == "__main__":
    main()
