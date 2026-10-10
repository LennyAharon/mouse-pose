"""Adaptation curves: zero-shot / few-shot / all-frames adaptation of a pose model to one dataset.

One grid config (``configs/adaptation/<name>.yaml``) describes everything: the target datasets,
the leave-that-dataset-out Mighty-Mouse trunk and the dedicated models for each, the frame counts
N, the frame draws, the arms (how a cell is initialised and trained) and the masked-label
settings. This module expands that config into cells, builds each cell's Lightning Pose training
overrides, and scores the evaluated cells. It does no I/O at import time and never imports
``lightning_pose``, so it is testable without a GPU and runs unchanged on any machine that has
``paths.yaml`` (local studio, ACCESS, ...).

Arms (``init`` x ``adapter`` x ``anchor``):

- ``init: trunk``  -- start from the dataset's leave-one-out trunk (its head is rebuilt from the
  trunk's config.yaml); ``adapter: lora`` trains LoRA adapters (``lora_lr``) + the head
  (``head_lr``), no adapter = full fine-tuning of every weight at ``lr``; ``anchor: true``
  distils the frozen trunk into the channels the target frame does not label (anchored LoRA).
- ``init: dinov3`` -- DINOv3 backbone + a new head (``head: linear`` or ``nonlinear``) trained
  from scratch on the N frames, every weight at ``lr``.

Scripts: ``scripts/adapt/{plan,run_cell,collect,plot}.py``; skill: ``skills/adaptation-curves/``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ALL = "all"
MASKED = "masked"
INIT_TRUNK, INIT_DINO = "trunk", "dinov3"


@dataclass(frozen=True)
class Cell:
    """One training run of the grid."""

    dataset: str
    arm:     str
    n:       int | str          # frames, or "all"
    draw:    int
    mask:    tuple[str, ...] = ()
    kind:    str = "curve"      # "curve" | "masked"

    @property
    def id(self) -> str:
        """Stable, filesystem-safe identifier, e.g. ``ibl__mm-anchored-lora__tf10__draw0``."""
        parts = [self.dataset, self.arm]
        if self.mask:
            parts.append("mask-" + "+".join(self.mask))
        parts += [f"tf{self.n}", f"draw{self.draw}"]
        return "__".join(parts)


@dataclass
class Grid:
    """A parsed grid config (see ``configs/adaptation/cosyne-2026.yaml`` for the format)."""

    name:        str
    datasets:    dict[str, dict]
    n_frames:    list
    draws:       list[int]
    arms:        dict[str, dict]
    curve_arms:  list[str]
    steps:       dict[str, int]
    val_every:   int
    masked:      dict = field(default_factory=dict)
    dedicated:   dict = field(default_factory=dict)
    out_subdir:  str = "adaptation"
    keep_checkpoints: bool = False
    order:       list = field(default_factory=list)   # launch order of the N blocks and "masked"
    eval_last:   bool = False   # also save the final-step checkpoint and evaluate it (eval_last/)


def load_grid(path: str | Path) -> Grid:
    """Read and validate a grid config.

    Raises:
        ValueError: on an unknown arm, an arm without a valid ``init``, a dataset without a
            trunk, or a non-positive frame count.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    grid = Grid(
        name=cfg["name"], datasets=cfg["datasets"], n_frames=list(cfg["n_frames"]),
        draws=[int(d) for d in cfg["draws"]], arms=cfg["arms"],
        curve_arms=list(cfg["curve_arms"]), steps={k: int(v) for k, v in cfg["steps"].items()},
        val_every=int(cfg.get("val_every", 250)), masked=cfg.get("masked") or {},
        dedicated=cfg.get("dedicated") or {}, out_subdir=cfg.get("out_subdir", "adaptation"),
        keep_checkpoints=bool(cfg.get("keep_checkpoints", False)),
        eval_last=bool(cfg.get("eval_last", False)),
    )
    blocks = grid.n_frames + ([MASKED] if grid.masked.get("settings") else [])
    grid.order = list(cfg.get("order") or blocks)
    if sorted(map(str, grid.order)) != sorted(map(str, blocks)):
        raise ValueError(f"order must list each of {blocks} once, got {grid.order}")
    for name, arm in grid.arms.items():
        if arm.get("init") not in (INIT_TRUNK, INIT_DINO):
            raise ValueError(f"arm {name}: init must be '{INIT_TRUNK}' or '{INIT_DINO}'")
        need = ("rank", "lora_lr", "head_lr") if arm.get("adapter") == "lora" else ("lr",)
        missing = [k for k in need if k not in arm]
        if missing:
            raise ValueError(f"arm {name}: missing {missing}")
    used = set(grid.curve_arms) | set(grid.masked.get("arms", []))
    unknown = used - set(grid.arms)
    if unknown:
        raise ValueError(f"unknown arm(s) {sorted(unknown)}; defined: {sorted(grid.arms)}")
    for ds, spec in grid.datasets.items():
        if not spec.get("trunk"):
            raise ValueError(f"dataset {ds}: no trunk")
    for n in grid.n_frames:
        if n != ALL and int(n) <= 0:
            raise ValueError(f"n_frames must be positive ints or '{ALL}', got {n}")
    return grid


def expand(grid: Grid) -> list[Cell]:
    """All cells of the grid, in launch order: the blocks of ``grid.order`` (each N value, and
    ``masked``; default every N in ``n_frames`` order, then masked), each by dataset then arm.

    ``n: all`` is run once (draw 0): every draw would select the same frames. Arms with
    ``skip_all: true`` (e.g. DINOv3 from scratch, whose all-frames point is the dedicated model)
    get no ``all`` cell.
    """
    blocks = {}
    for n in grid.n_frames:
        draws, cells = ([0] if n == ALL else grid.draws), []
        for ds in grid.datasets:
            for arm in grid.curve_arms:
                if n == ALL and grid.arms[arm].get("skip_all", False):
                    continue
                cells += [Cell(ds, arm, n, d) for d in draws]
        blocks[str(n)] = cells
    m, masked = grid.masked, []
    for ds, settings in (m.get("settings") or {}).items():
        for kps in settings:
            for n in m.get("n_frames", []):
                for arm in m.get("arms", []):
                    for d in ([0] if n == ALL else m.get("draws", grid.draws)):
                        masked.append(Cell(ds, arm, n, d, tuple(kps), "masked"))
    blocks[MASKED] = masked
    order = grid.order or grid.n_frames + [MASKED]
    return [c for b in order for c in blocks[str(b)]]


def cell_dir(results_dir: Path, grid: Grid, cell: Cell) -> Path:
    """``<results_dir>/<out_subdir>/<grid name>/<dataset>/<arm>/[mask-<kps>/]tf<N>-draw<d>``."""
    d = Path(results_dir) / grid.out_subdir / grid.name / cell.dataset / cell.arm
    if cell.mask:
        d = d / ("mask-" + "+".join(cell.mask))
    return d / f"tf{cell.n}-draw{cell.draw}"


def steps_for(grid: Grid, n: int | str) -> int:
    """Training steps of a cell: ``steps.all`` for all frames, else ``steps.few``."""
    return grid.steps[ALL] if n == ALL else grid.steps["few"]


def masked_csv_name(dataset: str, mask: tuple[str, ...]) -> str:
    """Training csv name with ``mask`` hidden (convention of scripts/build_masked_csv.py)."""
    return f"CollectedData_{dataset}_train_mask-{'+'.join(mask)}.csv"


def hide_keypoints(df: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, int]:
    """Copy of a label table with ``keys`` hidden (x/y NaN, visible 0), and the labels hidden.

    Raises:
        ValueError: if a key is not a keypoint of the table.
    """
    names = set(df.columns.get_level_values(1))
    unknown = [k for k in keys if k not in names]
    if unknown:
        raise ValueError(f"not keypoints of this table: {unknown}")
    out, n_hidden = df.copy(), 0
    for k in keys:
        cols = [c for c in out.columns if c[1] == k]
        n_hidden += int(out[[c for c in cols if c[2] == "x"]].notna().sum().sum())
        for c in cols:
            out[c] = 0 if c[2] == "visible" else np.nan
    return out, n_hidden


def best_checkpoint(run_dir: Path) -> Path | None:
    """The run's ``*-best.ckpt`` (else its only checkpoint), or None."""
    ck = Path(run_dir) / "tb_logs" / "test" / "version_0" / "checkpoints"
    best = sorted(ck.glob("*-best.ckpt"))
    return best[0] if best else (sorted(ck.glob("*.ckpt")) or [None])[0]


def trunk_problem(trunk_dir: Path) -> str | None:
    """Why a trunk cannot be adapted from yet, or None when it is ready.

    A running trunk already has a ``*-best.ckpt`` (of its best step so far), so a checkpoint
    alone does not mean the trunk is finished: its ``train_status.json`` must say COMPLETED.
    """
    d = Path(trunk_dir)
    if not (d / "config.yaml").exists():
        return "no config.yaml"
    status = d / "train_status.json"
    if not status.exists() or json.loads(status.read_text()).get("status") != "COMPLETED":
        return "training not COMPLETED"
    if best_checkpoint(d) is None:
        return "no checkpoint"
    return None


def trunk_head_overrides(trunk_cfg: dict) -> list[str]:
    """Overrides that rebuild a trunk's head before its checkpoint is loaded.

    A grouped head (``model.head_groups``) has the same state-dict keys as a plain nonlinear
    head, so it would load silently into the wrong head without these.
    """
    m, ov = trunk_cfg["model"], []
    if m.get("head_hidden_channels"):
        ov.append(f"+model.head_hidden_channels={int(m['head_hidden_channels'])}")
    if m.get("head_groups"):
        groups = ",".join(f"{g}:[{','.join(kps)}]" for g, kps in m["head_groups"].items())
        gain   = str(bool(m.get("head_group_fan_in_gain"))).lower()
        ov += [f"+model.head_groups={{{groups}}}",
               f"+model.head_shared_channels={int(m.get('head_shared_channels') or 0)}",
               f"+model.head_group_fan_in_gain={gain}"]
    return ov


def trunk_datasets(trunk_dir: Path, trunk_cfg: dict) -> list[str]:
    """Datasets the trunk trained on (from the frames of the training csv it saved)."""
    csv = Path(trunk_dir) / trunk_cfg["data"]["csv_file"]
    idx = pd.read_csv(csv, header=[0, 1, 2], index_col=0).index
    return sorted(set(idx.str.split("/").str[1]))


def trained_keypoints(datasets: list[str], inventory: dict) -> list[str]:
    """Union of ``trainable`` keypoints of ``datasets`` (dataset_inventory.json ``datasets``)."""
    sets = [set(inventory[d]["trainable"]) for d in datasets if d in inventory]
    return sorted(set().union(*sets))


def build_overrides(
    grid: Grid,
    cell: Cell,
    data_dir: Path,
    train_csv: str,
    trunk: dict | None,
) -> list[str]:
    """Lightning Pose overrides (for ``litpose train configs/model.yaml``) of one cell.

    Args:
        grid: the grid.
        cell: the cell.
        data_dir: corpus data dir.
        train_csv: training csv file name (the masked one for masked cells).
        trunk: for ``init: trunk`` arms, ``{"checkpoint", "backbone", "head_overrides",
            "keypoints"}`` (keypoints = the trunk's trained keypoints, anchored when the arm has
            ``anchor``).

    Returns:
        override strings, one per argv element (none contains a space).

    Raises:
        ValueError: if a trunk arm gets no trunk, or an override contains a space.
    """
    arm   = grid.arms[cell.arm]
    steps = steps_for(grid, cell.n)
    ov = [f"data.data_dir={data_dir}", f"data.csv_file={train_csv}", "model.losses_to_use=[]",
          f"training.train_frames={1 if cell.n == ALL else int(cell.n)}",
          f"training.rng_seed_data_pt={cell.draw}",
          f"training.min_steps={steps}", f"training.max_steps={steps}",
          "training.unfreezing_step=1", f"training.val_check_interval={grid.val_every}",
          f"training.lr_scheduler_params.multisteplr.milestone_steps=[{steps // 2}]",
          "+training.epoch_repeat=100", "+training.num_workers=2"]
    if grid.eval_last:   # a periodic checkpoint at the final step, evaluated besides the best one
        ov.append(f"+training.ckpt_every_n_steps={steps}")
    if arm["init"] == INIT_TRUNK:
        if trunk is None:
            raise ValueError(f"arm {cell.arm} starts from a trunk but none was given")
        lora = arm.get("adapter") == "lora"   # else full fine-tuning: every weight at `lr`
        ov += [f"model.backbone={trunk['backbone']}",
               f"+model.checkpoint='{trunk['checkpoint']}'",
               f"training.optimizer_params.learning_rate={arm['head_lr' if lora else 'lr']}"]
        if lora:
            r = int(arm["rank"])
            ov += [f"+model.lora.rank={r}", f"+model.lora.alpha={2 * r}",
                   f"+model.lora.lr={arm['lora_lr']}"]
        if arm.get("anchor"):
            kps = "[" + ",".join(f"'{k}'" for k in trunk["keypoints"]) + "]"
            ov += [f"+model.anchor.weight={arm.get('anchor_weight', 1.0)}",
                   "+model.anchor.mode=unlabeled",
                   f"+model.anchor.conf_power={arm.get('anchor_conf_power', 1)}",
                   f"+model.anchor.keypoints={kps}"]
        ov += list(trunk["head_overrides"])
    else:
        ov += [f"model.backbone={arm.get('backbone', 'vits_dinov3')}",
               f"training.optimizer_params.learning_rate={arm['lr']}"]
        if arm.get("head") == "nonlinear":
            width = int(arm.get("head_hidden_channels", 256))
            ov.append(f"+model.head_hidden_channels={width}")
    if any(" " in o for o in ov):
        raise ValueError(f"an override contains a space: {[o for o in ov if ' ' in o]}")
    return ov


def log_checks(arm: dict, log_text: str) -> list[str]:
    """Problems found in a cell's training log (empty = all expected lines present)."""
    problems = []
    if arm["init"] == INIT_TRUNK and "loading weights from" not in log_text:
        problems.append("trunk weights not loaded")
    if arm.get("anchor") and "anchor: frozen teacher attached" not in log_text:
        problems.append("anchor teacher not attached")
    if arm.get("adapter") == "lora":
        pattern = rf"LoRA: wrapped \d+ linear layers \(rank {int(arm['rank'])},"
        if not re.search(pattern, log_text):
            problems.append(f"LoRA (rank {arm.get('rank')}) not applied")
    if arm.get("adapter") != "lora" and "LoRA: wrapped" in log_text:
        problems.append("LoRA applied in a non-LoRA arm")
    if arm.get("head") == "nonlinear" and "nonlinear head:" not in log_text:
        problems.append("nonlinear head not built")
    return problems


# ── scoring ──────────────────────────────────────────────────────────────────


def pixel_errors(pred_csv: Path, label_csv: Path) -> pd.DataFrame:
    """Per (frame, keypoint) pixel error on visible == 2 labels (NaN elsewhere); test frames."""
    pr = pd.read_csv(pred_csv, header=[0, 1, 2], index_col=0)
    gt = pd.read_csv(label_csv, header=[0, 1, 2], index_col=0)
    if gt.index[0] == gt.index.name:
        gt = gt.iloc[1:]
    sp, sg = pr.columns[0][0], gt.columns[0][0]
    idx = gt.index.intersection(pr.index)
    out = {}
    for k in dict.fromkeys(gt.columns.get_level_values(1)):
        if (sg, k, "x") not in gt.columns or (sp, k, "x") not in pr.columns:
            continue
        if (sg, k, "visible") in gt.columns:
            vis = pd.to_numeric(gt.loc[idx, (sg, k, "visible")], errors="coerce") == 2
        else:
            vis = gt.loc[idx, (sg, k, "x")].notna()
        p = pr.loc[idx, [(sp, k, "x"), (sp, k, "y")]].astype(float).to_numpy()
        g = gt.loc[idx, [(sg, k, "x"), (sg, k, "y")]].astype(float).to_numpy()
        out[k] = np.where(vis.values, np.hypot(*(p - g).T), np.nan)
    return pd.DataFrame(out, index=idx)


def summarize(errors: pd.DataFrame, keypoints: list[str]) -> dict:
    """Pooled mean / median px and cell count over ``keypoints`` (only those present)."""
    cols = [k for k in keypoints if k in errors.columns]
    v = errors[cols].to_numpy(dtype=float).ravel() if cols else np.array([])
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean_px": np.nan, "median_px": np.nan, "n_cells": 0}
    return {"mean_px": float(v.mean()), "median_px": float(np.median(v)), "n_cells": int(len(v))}


def keypoint_sets(
    dataset: str,
    inventory: dict,
    trunk_keypoints: list[str],
    mask: tuple[str, ...] = (),
) -> dict:
    """Keypoint sets scored for a target dataset.

    ``all`` = every keypoint the dataset labels (``direct``); ``supported`` = those the
    leave-one-out trunk trained (comparable at N = 0); ``new`` = the rest; ``hidden`` = the
    masked keypoints.
    """
    direct  = sorted(inventory[dataset]["direct"])
    trained = set(trunk_keypoints)
    sets = {"all": direct, "supported": [k for k in direct if k in trained],
            "new": [k for k in direct if k not in trained]}
    if mask:
        sets["hidden"] = list(mask)
    return sets


def run_info(path: Path) -> dict:
    """The cell's run_info.json, or {}."""
    p = Path(path) / "run_info.json"
    return json.loads(p.read_text()) if p.exists() else {}
