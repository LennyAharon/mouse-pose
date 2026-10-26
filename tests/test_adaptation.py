"""Test the mighty_mouse.adaptation module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mighty_mouse.adaptation import (
    ALL,
    Cell,
    base_config,
    build_overrides,
    cell_dir,
    expand,
    flip_partner,
    hide_keypoints,
    keypoint_sets,
    load_grid,
    log_checks,
    masked_csv_name,
    masked_settings,
    pixel_errors,
    steps_for,
    summarize,
    trunk_head_overrides,
    trunk_input_overrides,
    trunk_problem,
)

LORA = {"init": "trunk", "adapter": "lora", "rank": 64, "lora_lr": 2.0e-4, "head_lr": 5.0e-4}
GRID = {
    "name": "t", "n_frames": [10, 50, "all"], "draws": [0, 1, 2],
    "steps": {"few": 2000, "all": 6000}, "val_every": 250,
    "arms": {"mm":   {**LORA, "anchor": True},
             "lora": LORA,
             "ft":   {"init": "trunk", "lr": 5.0e-5},
             "dino": {"init": "dinov3", "head": "nonlinear", "lr": 5.0e-5, "skip_all": True}},
    "curve_arms": ["mm", "dino"],
    "datasets": {"ibl": {"trunk": "x/ibl"}, "kondo": {"trunk": "x/kondo"}},
    "masked": {"n_frames": [10], "draws": [0, 1], "arms": ["mm", "lora"],
               "settings": {"ibl": [["pupil_center_left"], ["wrist_left", "wrist_right"]]}},
}
TRUNK = {"checkpoint": "/r/t.ckpt", "backbone": "vits_dinov3",
         "head_overrides": ["+model.head_hidden_channels=256"],
         "keypoints": ["nose_tip", "ear_top_left"]}


def label_table(rows: list[list[float]], keypoints: list[str]) -> pd.DataFrame:
    """Small label table with (scorer, keypoint, x|y|visible) columns."""
    cols = pd.MultiIndex.from_product([["s"], keypoints, ["x", "y", "visible"]])
    return pd.DataFrame(rows, columns=cols, index=[f"f{i}" for i in range(len(rows))])


@pytest.fixture
def grid(tmp_path: Path):
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(GRID))
    return load_grid(p)


class TestLoadGrid:
    """Test the function load_grid."""

    def test_load_grid_reads_fields(self, grid):
        assert grid.name == "t" and grid.val_every == 250
        assert grid.steps == {"few": 2000, "all": 6000} and grid.out_subdir == "adaptation"

    @pytest.mark.parametrize(("patch", "match"), [
        ({"curve_arms": ["nope"]}, "unknown arm"),
        ({"arms": {"mm": {"init": "x"}}, "curve_arms": ["mm"], "masked": {}}, "init must be"),
        ({"datasets": {"ibl": {}}}, "no trunk"),
        ({"arms": {"mm": {"init": "trunk"}}, "curve_arms": ["mm"], "masked": {}}, "missing"),
        ({"n_frames": [0]}, "positive"),
    ])
    def test_load_grid_invalid_raises(self, tmp_path: Path, patch, match):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({**GRID, **patch}))
        with pytest.raises(ValueError, match=match):
            load_grid(p)


class TestExpand:
    """Test the function expand."""

    def test_expand_counts(self, grid):
        cells = expand(grid)
        curve = [c for c in cells if c.kind == "curve"]
        # N in (10, 50) x 3 draws x 2 datasets x 2 arms, + `all` once per dataset (dino skips it)
        assert len(curve) == 2 * 3 * 2 * 2 + 2
        assert [c for c in curve if c.n == ALL] == [Cell("ibl", "mm", ALL, 0),
                                                    Cell("kondo", "mm", ALL, 0)]
        masked = [c for c in cells if c.kind == "masked"]
        assert len(masked) == 2 * 1 * 2 * 2   # settings x N x arms x draws

    def test_expand_order(self, grid, tmp_path: Path):
        p = tmp_path / "o.yaml"
        p.write_text(yaml.safe_dump({**GRID, "order": [10, "masked", 50, "all"]}))
        cells = expand(load_grid(p))
        kinds = [c.kind if c.kind == "masked" else c.n for c in cells]
        first = {k: kinds.index(k) for k in (10, "masked", 50, ALL)}
        assert first[10] < first["masked"] < first[50] < first[ALL]
        assert sorted(c.id for c in cells) == sorted(c.id for c in expand(grid))   # same cells

    def test_expand_order_incomplete_raises(self, tmp_path: Path):
        p = tmp_path / "o.yaml"
        p.write_text(yaml.safe_dump({**GRID, "order": [10, "masked", "all"]}))
        with pytest.raises(ValueError, match="order must list"):
            load_grid(p)

    def test_expand_draws_all(self, tmp_path: Path):
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump({**GRID, "n_frames": ["all"], "draws_all": [0, 1, 2],
                                     "datasets": {"ibl": {"trunk": "x/ibl"}}, "masked": {}}))
        assert [c.id for c in expand(load_grid(p))] == [
            f"ibl__mm__tfall__draw{d}" for d in (0, 1, 2)]

    def test_expand_ids_unique(self, grid):
        ids = [c.id for c in expand(grid)]
        assert len(ids) == len(set(ids))


class TestMaskedSettings:
    """Test the function masked_settings (and expand with per-setting frame counts)."""

    def test_masked_settings_dict_entry(self, tmp_path: Path):
        settings = {"ibl": [{"keypoints": ["pupil_center_left"], "n_frames": [10, 50]},
                            ["wrist_left", "wrist_right"]]}
        p = tmp_path / "m.yaml"
        p.write_text(yaml.safe_dump({**GRID, "masked": {**GRID["masked"], "settings": settings}}))
        g = load_grid(p)
        assert masked_settings(g) == [("ibl", ("pupil_center_left",), [10, 50]),
                                      ("ibl", ("wrist_left", "wrist_right"), [10])]
        masked = [c for c in expand(g) if c.kind == "masked"]
        assert len(masked) == (2 + 1) * 2 * 2   # (N values) x arms x draws
        assert {c.n for c in masked if c.mask == ("pupil_center_left",)} == {10, 50}


class TestFlipPartner:
    """Test the function flip_partner."""

    def test_flip_partner(self):
        assert flip_partner("pupil_center_left") == "pupil_center_right"
        assert flip_partner("wrist_right") == "wrist_left"
        assert flip_partner("nose_tip") is None


class TestCellDir:
    """Test the function cell_dir (and Cell.id)."""

    def test_cell_dir_masked(self, grid):
        c = Cell("ibl", "lora", 10, 1, ("wrist_left", "wrist_right"), "masked")
        assert c.id == "ibl__lora__mask-wrist_left+wrist_right__tf10__draw1"
        assert cell_dir(Path("/r"), grid, c) == Path(
            "/r/adaptation/t/ibl/lora/mask-wrist_left+wrist_right/tf10-draw1")

    def test_cell_dir_curve(self, grid):
        assert cell_dir(Path("/r"), grid, Cell("ibl", "mm", ALL, 0)) == Path(
            "/r/adaptation/t/ibl/mm/tfall-draw0")


class TestStepsFor:
    """Test the function steps_for."""

    def test_steps_for(self, grid):
        assert steps_for(grid, ALL) == 6000 and steps_for(grid, 25) == 2000

    def test_steps_for_arm_override(self, grid):
        arm = {"steps": {"few": 4000}}
        assert steps_for(grid, 10, arm) == 4000 and steps_for(grid, ALL, arm) == 6000


class TestBaseConfig:
    """Test the function base_config."""

    def test_base_config(self):
        assert base_config({}) == "configs/model.yaml"
        zoom = "configs/model_zoominout.yaml"
        assert base_config({"config": zoom}) == zoom


class TestBuildOverrides:
    """Test the function build_overrides."""

    def test_build_overrides_anchored_lora(self, grid):
        ov = build_overrides(grid, Cell("ibl", "mm", 10, 2), Path("/d"),
                             "CollectedData_ibl_train.csv", TRUNK)
        for o in ["training.train_frames=10", "training.rng_seed_data_pt=2",
                  "training.max_steps=2000", "training.val_check_interval=250",
                  "training.lr_scheduler_params.multisteplr.milestone_steps=[1000]",
                  "+model.checkpoint='/r/t.ckpt'", "+model.lora.rank=64",
                  "+model.lora.alpha=128", "+model.anchor.keypoints=['nose_tip','ear_top_left']",
                  "+model.head_hidden_channels=256",
                  "training.optimizer_params.learning_rate=0.0005"]:
            assert o in ov, o

    def test_build_overrides_input_size(self, grid):
        trunk = {**TRUNK, "input_overrides": ["data.image_resize_dims.height=384",
                                              "data.image_resize_dims.width=384"]}
        ov = build_overrides(grid, Cell("ibl", "mm", ALL, 1), Path("/d"), "x.csv", trunk)
        assert {"data.image_resize_dims.height=384", "data.image_resize_dims.width=384"} <= set(ov)
        assert "training.rng_seed_data_pt=1" in ov
        assert not any("image_resize_dims" in o for o in
                       build_overrides(grid, Cell("ibl", "mm", 10, 0), Path("/d"), "x.csv", TRUNK))

    def test_build_overrides_eval_last(self, grid):
        assert not any("ckpt_every_n_steps" in o for o in
                       build_overrides(grid, Cell("ibl", "mm", 10, 0), Path("/d"), "x.csv", TRUNK))
        grid.eval_last = True
        ov = build_overrides(grid, Cell("ibl", "mm", ALL, 0), Path("/d"), "x.csv", TRUNK)
        assert "+training.ckpt_every_n_steps=6000" in ov

    def test_build_overrides_plain_lora_has_no_anchor(self, grid):
        ov = build_overrides(grid, Cell("ibl", "lora", 10, 0), Path("/d"), "x.csv", TRUNK)
        assert "+model.lora.rank=64" in ov
        assert not any(o.startswith("+model.anchor") for o in ov)

    def test_build_overrides_full_ft(self, grid):
        ov = build_overrides(grid, Cell("ibl", "ft", 25, 1), Path("/d"), "x.csv", TRUNK)
        assert "+model.checkpoint='/r/t.ckpt'" in ov and "+model.head_hidden_channels=256" in ov
        assert "training.optimizer_params.learning_rate=5e-05" in ov
        assert not any("lora" in o or "anchor" in o for o in ov)

    def test_build_overrides_arm_steps_and_extra(self, grid):
        grid.arms["mm"] = {**grid.arms["mm"], "steps": {"few": 4000},
                           "overrides": ["training.optimizer_params.weight_decay=0.01"]}
        ov = build_overrides(grid, Cell("ibl", "mm", 10, 0), Path("/d"), "x.csv", TRUNK)
        assert "training.max_steps=4000" in ov
        assert "training.lr_scheduler_params.multisteplr.milestone_steps=[2000]" in ov
        assert ov[-1] == "training.optimizer_params.weight_decay=0.01"

    def test_build_overrides_video(self, grid):
        grid.arms["vid"] = {**grid.arms["mm"], "video": True, "video_weight": 0.5}
        trunk = {**TRUNK, "video_keypoints": ["ear_top_left"]}
        ov = build_overrides(grid, Cell("ibl", "vid", 10, 0), Path("/d"), "x.csv", trunk,
                             Path("/d/videos_unlabelled/ibl"))
        assert "model.losses_to_use=[anchor_video]" in ov and "model.losses_to_use=[]" not in ov
        assert "data.video_dir=/d/videos_unlabelled/ibl" in ov
        assert "+model.anchor.video_weight=0.5" in ov
        assert "+model.anchor.video_keypoints=['ear_top_left']" in ov

    def test_build_overrides_video_all_channels(self, grid):
        grid.arms["vid"] = {**grid.arms["mm"], "video": True, "video_keypoints": "all"}
        ov = build_overrides(grid, Cell("ibl", "vid", 10, 0), Path("/d"), "x.csv", TRUNK,
                             Path("/d/v"))
        assert not any("video_keypoints" in o for o in ov)

    def test_build_overrides_video_no_unlabelled_keypoints_raises(self, grid):
        grid.arms["vid"] = {**grid.arms["mm"], "video": True}
        with pytest.raises(ValueError, match="no keypoints left"):
            build_overrides(grid, Cell("ibl", "vid", 10, 0), Path("/d"), "x.csv",
                            {**TRUNK, "video_keypoints": []}, Path("/d/v"))
        with pytest.raises(ValueError, match="video needs"):
            build_overrides(grid, Cell("ibl", "vid", 10, 0), Path("/d"), "x.csv", TRUNK)

    def test_build_overrides_all_frames(self, grid):
        ov = build_overrides(grid, Cell("ibl", "mm", ALL, 0), Path("/d"), "x.csv", TRUNK)
        assert "training.train_frames=1" in ov and "training.max_steps=6000" in ov

    def test_build_overrides_dino(self, grid):
        ov = build_overrides(grid, Cell("kondo", "dino", 50, 0), Path("/d"), "x.csv", None)
        assert not any("checkpoint" in o or "lora" in o or "anchor" in o for o in ov)
        assert "+model.head_hidden_channels=256" in ov
        assert "training.optimizer_params.learning_rate=5e-05" in ov

    def test_build_overrides_trunk_arm_without_trunk_raises(self, grid):
        with pytest.raises(ValueError, match="starts from a trunk"):
            build_overrides(grid, Cell("ibl", "mm", 10, 0), Path("/d"), "x.csv", None)

    def test_build_overrides_space_raises(self, grid):
        with pytest.raises(ValueError, match="contains a space"):
            build_overrides(grid, Cell("ibl", "lora", 10, 0), Path("/my data"), "x.csv", TRUNK)


class TestTrunkProblem:
    """Test the function trunk_problem."""

    def make_trunk(self, root: Path, status: str | None, ckpt: str | None) -> Path:
        ck = root / "tb_logs" / "test" / "version_0" / "checkpoints"
        ck.mkdir(parents=True)
        (root / "config.yaml").write_text("model: {}\n")
        if status:
            (root / "train_status.json").write_text(f'{{"status": "{status}"}}')
        if ckpt:
            (ck / ckpt).touch()
        return root

    def test_trunk_problem_ready(self, tmp_path: Path):
        trunk = self.make_trunk(tmp_path, "COMPLETED", "e=1-step=9-best.ckpt")
        assert trunk_problem(trunk) is None

    def test_trunk_problem_still_training(self, tmp_path: Path):
        # a running trunk already has a -best.ckpt of its best step so far
        trunk = self.make_trunk(tmp_path, "TRAINING", "e=1-step=750-best.ckpt")
        assert trunk_problem(trunk) == "training not COMPLETED"

    def test_trunk_problem_missing(self, tmp_path: Path):
        assert trunk_problem(tmp_path / "nope") == "no config.yaml"
        assert trunk_problem(self.make_trunk(tmp_path, "COMPLETED", None)) == "no checkpoint"


class TestTrunkHeadOverrides:
    """Test the function trunk_head_overrides."""

    def test_trunk_head_overrides_linear(self):
        assert trunk_head_overrides({"model": {}}) == []

    def test_trunk_head_overrides_nonlinear(self):
        assert trunk_head_overrides({"model": {"head_hidden_channels": 256}}) == [
            "+model.head_hidden_channels=256"]

    def test_trunk_head_overrides_groups(self):
        m = {"head_hidden_channels": 256, "head_shared_channels": 0,
             "head_group_fan_in_gain": True, "head_groups": {"eye": ["a", "b"], "paw": ["c"]}}
        assert trunk_head_overrides({"model": m}) == [
            "+model.head_hidden_channels=256", "+model.head_groups={eye:[a,b],paw:[c]}",
            "+model.head_shared_channels=0", "+model.head_group_fan_in_gain=true"]


class TestTrunkInputOverrides:
    """Test the function trunk_input_overrides."""

    def test_trunk_input_overrides_384(self):
        cfg = {"data": {"image_resize_dims": {"height": 384, "width": 384}}}
        assert trunk_input_overrides(cfg) == ["data.image_resize_dims.height=384",
                                              "data.image_resize_dims.width=384"]

    def test_trunk_input_overrides_missing(self):
        assert trunk_input_overrides({"data": {}}) == []


class TestHideKeypoints:
    """Test the function hide_keypoints."""

    def test_hide_keypoints_hides_and_counts(self):
        df = label_table([[1.0, 2.0, 2, 3.0, 4.0, 2], [np.nan, np.nan, 0, 5.0, 6.0, 1]],
                         ["a", "b"])
        out, n = hide_keypoints(df, ["b"])
        assert n == 2
        assert out[("s", "b", "x")].isna().all() and (out[("s", "b", "visible")] == 0).all()
        assert out[("s", "a", "x")].iloc[0] == 1.0
        assert out[("s", "a", "visible")].tolist() == [2, 0]
        assert df[("s", "b", "x")].iloc[0] == 3.0   # input untouched

    def test_hide_keypoints_unknown_raises(self):
        with pytest.raises(ValueError, match="not keypoints"):
            hide_keypoints(label_table([[1.0, 2.0, 2]], ["a"]), ["z"])


class TestMaskedCsvName:
    """Test the function masked_csv_name."""

    def test_masked_csv_name(self):
        assert masked_csv_name("ibl", ("a", "b")) == "CollectedData_ibl_train_mask-a+b.csv"


class TestPixelErrors:
    """Test the function pixel_errors."""

    def test_pixel_errors_scores_visible_2_only(self, tmp_path: Path):
        gt = label_table([[0.0, 0.0, 2, 0.0, 0.0, 1], [0.0, 0.0, 2, 1.0, 1.0, 2]], ["a", "b"])
        pr = pd.DataFrame([[3.0, 4.0, 0.9, 9.0, 9.0, 0.9], [0.0, 1.0, 0.9, 1.0, 1.0, 0.9]],
                          columns=pd.MultiIndex.from_product([["m"], ["a", "b"],
                                                              ["x", "y", "likelihood"]]),
                          index=["f0", "f1"])
        gt.to_csv(tmp_path / "gt.csv")
        pr.to_csv(tmp_path / "pr.csv")
        e = pixel_errors(tmp_path / "pr.csv", tmp_path / "gt.csv")
        assert e["a"].tolist() == [5.0, 1.0]
        assert np.isnan(e.loc["f0", "b"]) and e.loc["f1", "b"] == 0.0   # visible 1 not scored


class TestSummarize:
    """Test the function summarize."""

    def test_summarize_pools_cells(self):
        e = pd.DataFrame({"a": [1.0, 3.0, np.nan], "b": [10.0, np.nan, np.nan]})
        assert summarize(e, ["a", "b"]) == {"mean_px": 14 / 3, "median_px": 3.0, "n_cells": 3}

    def test_summarize_no_keypoints(self):
        out = summarize(pd.DataFrame({"a": [1.0]}), ["zzz"])
        assert out["n_cells"] == 0 and np.isnan(out["mean_px"])


class TestKeypointSets:
    """Test the function keypoint_sets."""

    def test_keypoint_sets(self):
        inv = {"ibl": {"direct": ["nose_tip", "tongue_end_left", "pupil_center_left"]}}
        sets = keypoint_sets("ibl", inv, ["nose_tip", "pupil_center_left", "ear_top_left"],
                             ("pupil_center_left",))
        assert sets == {"all": ["nose_tip", "pupil_center_left", "tongue_end_left"],
                        "supported": ["nose_tip", "pupil_center_left"],
                        "new": ["tongue_end_left"], "hidden": ["pupil_center_left"],
                        "shown": ["nose_tip", "tongue_end_left"]}


class TestLogChecks:
    """Test the function log_checks."""

    ARM = {"init": "trunk", "adapter": "lora", "rank": 64, "anchor": True}

    def test_log_checks_ok(self):
        log = ("loading weights from x\nanchor: frozen teacher attached\n"
               "LoRA: wrapped 72 linear layers (rank 64, alpha 128)")
        assert log_checks(self.ARM, log) == []

    def test_log_checks_missing_lines(self):
        assert set(log_checks(self.ARM, "")) == {"trunk weights not loaded",
                                                 "anchor teacher not attached",
                                                 "LoRA (rank 64) not applied"}

    def test_log_checks_video(self):
        arm = {**self.ARM, "video": True}
        log = ("loading weights from x\nanchor: frozen teacher attached\n"
               "LoRA: wrapped 72 linear layers (rank 64, alpha 128)")
        assert log_checks(arm, log) == ["video anchor not applied"]
        assert log_checks(arm, log + "\nanchor loss on unlabeled video: weight 1") == [
            "video anchor not restricted to the unlabelled keypoints"]
        ok = log + "\nanchor loss on unlabeled video: weight 1, 38 keypoints"
        assert log_checks(arm, ok) == []
        assert log_checks({**arm, "video_keypoints": "all"},
                          log + "\nanchor loss on unlabeled video: weight 1") == []

    def test_log_checks_dino(self):
        assert log_checks({"init": "dinov3", "head": "nonlinear"}, "LoRA: wrapped 72") == [
            "LoRA applied in a non-LoRA arm", "nonlinear head not built"]
