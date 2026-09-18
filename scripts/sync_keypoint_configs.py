#!/usr/bin/env python
"""
Keep every training config's keypoint vocabulary in sync with configs/keypoints.yaml.

configs/keypoints.yaml is the single source of truth for the canonical keypoint names and
their order. Each training config (configs/model*.yaml, configs/zero_shot/*.yaml) repeats the
list under data.keypoint_names plus data.num_keypoints, and convert_dataset.py refuses to run
when they disagree. This script rewrites those two fields in place (text edit, comments and the
rest of the file untouched) so a vocabulary change is one edit to keypoints.yaml.

    python scripts/sync_keypoint_configs.py           # rewrite configs that differ
    python scripts/sync_keypoint_configs.py --check   # exit 1 if any config differs, change nothing
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(ROOT.glob("configs/model*.yaml")) + sorted(ROOT.glob("configs/zero_shot/*.yaml"))


def canonical() -> list[str]:
    k = yaml.safe_load((ROOT / "configs" / "keypoints.yaml").read_text())
    return list(k["keypoints"] if isinstance(k, dict) else k)


def rewrite(path: Path, names: list[str], check: bool) -> bool:
    """Return True if the file differed from `names`."""
    text = path.read_text()
    cfg = yaml.safe_load(text)
    have = list(cfg["data"].get("keypoint_names") or [])
    n_have = cfg["data"].get("num_keypoints")
    if have == names and n_have == len(names):
        return False
    if check:
        return True
    # replace the keypoint_names block: from the key line to the next 2-space-indented key
    m = re.search(r"^  keypoint_names:\n(.*?)(?=^  [A-Za-z_]+:)", text, flags=re.S | re.M)
    if not m:
        raise SystemExit(f"{path}: could not locate data.keypoint_names block")
    block = "  keypoint_names:\n" + "".join(f"  - {n}\n" for n in names)
    text = text[:m.start()] + block + text[m.end():]
    text = re.sub(r"^  num_keypoints: \d+", f"  num_keypoints: {len(names)}", text, count=1, flags=re.M)
    path.write_text(text)
    cfg2 = yaml.safe_load(text)
    assert cfg2["data"]["keypoint_names"] == names and cfg2["data"]["num_keypoints"] == len(names), path
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    names = canonical()
    differ = [p for p in CONFIGS if rewrite(p, names, args.check)]
    for p in CONFIGS:
        print(f"  {'DIFFERS' if p in differ else 'in sync':8s} {p.relative_to(ROOT)}")
    print(f"{len(names)} canonical keypoints; {len(differ)} config(s) {'differ' if args.check else 'rewritten'}")
    sys.exit(1 if args.check and differ else 0)


if __name__ == "__main__":
    main()
