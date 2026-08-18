"""
Single source of truth for the ordered dataset registry.

The list position in configs/dataset_registry.yaml IS the dataset id — samplers,
per-dataset heads, and checkpoints identify datasets by index — so every consumer
(sweep evaluation, plotting, Lightning Pose's `data.dataset_names` override, the
inventory command) must read the same file rather than keep its own list.
"""

from pathlib import Path

import yaml

from mouse_pose.paths import repo_root

REGISTRY_FILE = repo_root() / "configs" / "dataset_registry.yaml"


def load_registry(registry_file: Path = REGISTRY_FILE) -> list[str]:
    """Return the ordered dataset names; the list index is the dataset id."""
    names = yaml.safe_load(registry_file.read_text())["datasets"]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate dataset names in {registry_file}: {names}")
    return names
