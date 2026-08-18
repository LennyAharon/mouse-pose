"""
Read-only inventory of the converted head-fixed corpus.

Validates that every per-dataset CSV shares one canonical keypoint schema, derives
the three per-dataset keypoint masks (direct / trainable / evaluation), asserts
train/test session disjointness, and reports raw labeled-data directories that the
conversion configs never reference (e.g. kondo's iter7_* sets) without adding them.

Run before freezing a data version, so documentation cannot silently drift:

    python -m mouse_pose.inventory

Writes:
  docs/dataset_inventory.md          human-readable inventory (checked into the repo)
  <data_dir>/dataset_inventory.json  machine-readable manifest (travels with the data)

Masks:
  direct     keypoints with at least one visible=2 label in the train CSV
  trainable  direct, plus _left/_right partners of lateralized direct keypoints
             (hflip augmentation gives these training signal without direct labels)
  eval       keypoints with at least one visible=2 label in the test CSV
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from mouse_pose.paths import load_paths, repo_root
from mouse_pose.registry import load_registry

_paths   = load_paths()
DATA_DIR = Path(_paths["data_dir"])
RAW_DIR  = Path(_paths["raw_dir"])


# ── per-dataset extraction ───────────────────────────────────────────────────

def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if df.index[0] == df.index.name:  # stray header row from some writers
        df = df.iloc[1:]
    return df


def keypoint_names(df: pd.DataFrame) -> list[str]:
    """Canonical keypoint names in column order."""
    return [b for b, c in zip(df.columns.get_level_values(1), df.columns.get_level_values(2))
            if c == "x"]


def sessions(df: pd.DataFrame) -> set[str]:
    """Session names parsed from labeled-data/<dataset>/<session>/<frame> row paths."""
    return {Path(i).parts[2] for i in df.index}


def direct_mask(df: pd.DataFrame) -> set[str]:
    """Keypoints with at least one visible=2 label."""
    vis = df.loc[:, df.columns.get_level_values(2) == "visible"]
    kps = vis.columns.get_level_values(1)
    return {k for k, col in zip(kps, vis.columns) if (df[col] == 2).any()}


def hflip_partner(name: str) -> str | None:
    if name.endswith("_left"):
        return name[:-5] + "_right"
    if name.endswith("_right"):
        return name[:-6] + "_left"
    return None


def trainable_mask(direct: set[str], canonical: list[str]) -> set[str]:
    """Direct keypoints plus hflip partners that exist in the canonical vocabulary."""
    partners = {p for k in direct if (p := hflip_partner(k)) and p in canonical}
    return direct | partners


def unreferenced_raw_sessions(name: str, referenced: set[str]) -> list[str]:
    """Raw labeled-data directories the conversion never consumed."""
    raw_labeled = RAW_DIR / name / "labeled-data"
    if not raw_labeled.is_dir():
        return []
    return sorted(
        d.name for d in raw_labeled.iterdir() if d.is_dir() and d.name not in referenced
    )


def dataset_entry(name: str, canonical: list[str]) -> dict:
    """Build the full inventory entry for one dataset; raises on schema mismatch."""
    train = read_csv(DATA_DIR / f"CollectedData_{name}_train.csv")
    test  = read_csv(DATA_DIR / f"CollectedData_{name}_test.csv")

    for split, df in (("train", train), ("test", test)):
        if keypoint_names(df) != canonical:
            raise ValueError(
                f"{name} {split} CSV keypoint schema deviates from canonical "
                f"({len(keypoint_names(df))} vs {len(canonical)} keypoints, or wrong order)"
            )

    train_sessions = sessions(train)
    test_sessions  = sessions(test)
    overlap        = train_sessions & test_sessions
    if overlap:
        raise ValueError(
            f"{name}: train/test sessions overlap: {sorted(overlap)} — the evaluation "
            f"depends on session disjointness, refusing to continue"
        )

    direct = direct_mask(train)
    config = yaml.safe_load((repo_root() / "configs" / "datasets" / f"{name}.yaml").read_text())

    return {
        "train_frames":    len(train),
        "test_frames":     len(test),
        "train_sessions":  len(train_sessions),
        "test_sessions":   len(test_sessions),
        "direct":          sorted(direct),
        "trainable":       sorted(trainable_mask(direct, canonical)),
        "eval":            sorted(direct_mask(test)),
        "views":           config.get("sessions", {}),
        "excluded":        (config.get("exclude") or {}).get("keypoints") or [],
        "unreferenced_raw": unreferenced_raw_sessions(
            name, train_sessions | test_sessions
        ),
    }


# ── report generation ────────────────────────────────────────────────────────

def make_markdown(canonical: list[str], entries: dict[str, dict]) -> str:
    lines = [
        "# Dataset inventory",
        "",
        "Generated by `python -m mouse_pose.inventory` — do not edit by hand.",
        "",
        f"Canonical vocabulary: **{len(canonical)} keypoints**. "
        "`pupil_center_right` has no direct label in any dataset; it is trainable only "
        "through hflip augmentation and must never enter evaluation or aggregate metrics.",
        "",
        "Every dataset's test sessions are disjoint from its train sessions (asserted on "
        "every run of this command).",
        "",
        "| dataset | train frames | train sessions | test frames | test sessions "
        "| direct | trainable | eval |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in entries.items():
        lines.append(
            f"| `{name}` | {e['train_frames']:,} | {e['train_sessions']} "
            f"| {e['test_frames']:,} | {e['test_sessions']} "
            f"| {len(e['direct'])} | {len(e['trainable'])} | {len(e['eval'])} |"
        )
    total_train = sum(e["train_frames"] for e in entries.values())
    total_test  = sum(e["test_frames"] for e in entries.values())
    lines.append(f"| **total** | **{total_train:,}** | | **{total_test:,}** | | | | |")

    lines += ["", "## Pairwise direct-keypoint overlap", "",
              "| | " + " | ".join(f"`{n}`" for n in entries) + " |",
              "|---|" + "---|" * len(entries)]
    for a, ea in entries.items():
        row = [f"| `{a}`"]
        for b, eb in entries.items():
            row.append(str(len(set(ea["direct"]) & set(eb["direct"]))))
        lines.append(" | ".join(row) + " |")

    for name, e in entries.items():
        lines += ["", f"## `{name}`", ""]
        if e["views"]:
            view_counts = pd.Series(list(e["views"].values())).value_counts().to_dict()
            summary = ", ".join(f"{v} sessions {k}-view" for k, v in sorted(view_counts.items()))
            lines.append(f"- **Views:** {summary}")
        else:
            lines.append("- **Views:** n/a")
        if e["excluded"]:
            lines.append(f"- **Excluded source keypoints:** {', '.join(e['excluded'])}")
        lines.append(f"- **Direct ({len(e['direct'])}):** " + ", ".join(e["direct"]))
        hflip_only = sorted(set(e["trainable"]) - set(e["direct"]))
        if hflip_only:
            lines.append(f"- **Hflip-only trainable:** {', '.join(hflip_only)}")
        if e["unreferenced_raw"]:
            lines.append(
                f"- **⚠ Unreferenced raw labeled-data dirs (deliberately not converted):** "
                + ", ".join(e["unreferenced_raw"])
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the converted corpus and regenerate the dataset inventory.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()

    registry  = load_registry()
    canonical = keypoint_names(read_csv(DATA_DIR / f"CollectedData_{registry[0]}_train.csv"))
    entries   = {name: dataset_entry(name, canonical) for name in registry}

    manifest = {
        "canonical_keypoints": canonical,
        "registry":            registry,
        "datasets":            entries,
    }
    json_path = DATA_DIR / "dataset_inventory.json"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n")

    md_path = repo_root() / "docs" / "dataset_inventory.md"
    md_path.write_text(make_markdown(canonical, entries))

    print(f"canonical keypoints: {len(canonical)}")
    for name, e in entries.items():
        flag = f"  ⚠ unreferenced raw: {e['unreferenced_raw']}" if e["unreferenced_raw"] else ""
        print(f"  {name:15s} train {e['train_frames']:5,}/{e['train_sessions']:3d} sess"
              f"  test {e['test_frames']:5,}/{e['test_sessions']:3d} sess"
              f"  direct {len(e['direct']):2d}  trainable {len(e['trainable']):2d}"
              f"  eval {len(e['eval']):2d}{flag}")
    print(f"→ {md_path}")
    print(f"→ {json_path}")


if __name__ == "__main__":
    main()
