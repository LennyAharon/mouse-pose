"""
Single source of truth for which datasets are registered in the combined corpus.

Used by scripts/build_dataset.py (as the default --datasets set) and mouse_pose/train.py (as the
set evaluated against after each training run). The list itself lives in
configs/dataset_registry.yaml, where the list position is the dataset id (samplers, per-dataset
heads and checkpoints identify datasets by index): append new datasets at the end, never
reorder. This does NOT derive from configs/datasets/*.yaml; adding a dataset is still a manual
step (see scripts/preprocessing/README.md, stage 2), since not every configs/datasets/<name>.yaml
is necessarily ready to be part of default runs yet.
"""

from mouse_pose.registry import load_registry

ALL_DATASETS = load_registry()
