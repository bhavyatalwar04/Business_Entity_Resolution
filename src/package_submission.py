"""Build <team>_submission.zip in the structure required by the challenge:

<team>_submission.zip
├── output/matching_results.tsv, output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/ (+ src/xlmr_ce, src/qwen_ce), configs/, scripts/, utils/, README.md, requirements.txt}
└── Documentation_template.md

Usage: python -m src.package_submission --team <team_name>
"""
import argparse
import glob
import hashlib
import os
import zipfile

# teammate placeholders that the final pipeline does not use
EXCLUDE_SRC = {"cross_encoder.py", "export.py"}


def main():
    """Validate that every required file exists, then write the zip."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out_dir", default=".")
    # the final leaderboard file (v15_frself_s-1.25, laptop build, LB 0.986509); the zip must hold exactly this file
    ap.add_argument("--expect_md5", default="8b68aed84c39980832ca283d1f050748")
    a = ap.parse_args()
    pkg = "code/business_entity_resolution"
    files = {
        "output/matching_results.tsv": "output/matching_results.tsv",
        "output/candidate_pairs.tsv": "output/candidate_pairs.tsv",
        "Documentation_template.md": "Documentation_template.md",
        f"{pkg}/README.md": "docs/PACKAGE_README.md",
        f"{pkg}/requirements.txt": "requirements.txt",
        f"{pkg}/utils/validate_submission.py": "utils/validate_submission.py",
    }
    for f in sorted(glob.glob("src/*.py")):
        if os.path.basename(f) not in EXCLUDE_SRC:
            files[f"{pkg}/src/{os.path.basename(f)}"] = f
    # cross-encoder sub-packages (code, READMEs, job scripts, pinned GPU requirements, small gate files)
    for sub in ("xlmr_ce", "qwen_ce"):
        for f in sorted(glob.glob(f"src/{sub}/**/*", recursive=True)):
            if os.path.isfile(f) and "__pycache__" not in f:
                files[f"{pkg}/{f}"] = f
    for f in sorted(glob.glob("configs/*.yaml")) + sorted(glob.glob("scripts/*")):
        files[f"{pkg}/{f}"] = f
    missing = [src for src in files.values() if not os.path.exists(src)]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    h = hashlib.md5()
    with open("output/matching_results.tsv", "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    if a.expect_md5 and h.hexdigest() != a.expect_md5:
        raise SystemExit(f"output/matching_results.tsv md5 {h.hexdigest()} != {a.expect_md5} (not the leaderboard file)")
    path = os.path.join(a.out_dir, f"{a.team}_submission.zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, src in files.items():
            z.write(src, arc)
            print(f"  + {arc}")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
