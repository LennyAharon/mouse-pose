"""
Single source of truth for which datasets are registered in the combined corpus.

This does NOT derive from configs/datasets/*.yaml; adding a dataset here is a manual step
(stage 2 of onboarding a dataset) since not every configs/datasets/<name>.yaml is necessarily
ready to be part of default runs yet.

Invariant: a dataset is in the registry (ALL_DATASETS) if and only if its
configs/datasets/<name>.yaml is tracked in git (a collaborator's default runs would otherwise fail
on the missing config).
"""

from mighty_mouse.registry import load_registry

ALL_DATASETS = load_registry()

# dataset -> short name in merged-CSV tags, in tag order (append new datasets at the end; same
# order as configs/dataset_registry.yaml). From upstream main (#30); scripts/evaluation/zeroshot_score.py
# and eval_suite.py use these.
SHORT_NAMES = {
    "facemap":       "face",
    "ibl":           "ibl",
    "cheese-2d":     "cheese",
    "cazettes-side": "caz",
    "kondo":         "kondo",
    "hantman-mv":    "hmv",
    "cheese-3d":     "c3d",
    "kaufman":       "kauf",
    "aind-vbn-body": "vbnb",
    "aind-vbn-face": "vbnf",
    "cavelli":       "cav",
    "tyssowski":     "tys",
    "kaufman-water": "kaufw",
}


def tag_of(datasets: list[str]) -> str:
    """Merged-CSV tag of a set of datasets: their short names joined by "+", in SHORT_NAMES order.

    Args:
        datasets: dataset names (order and repeats do not matter)

    Returns:
        e.g. "face+ibl+cheese" for ["cheese-2d", "facemap", "ibl"]

    Raises:
        KeyError: if a dataset has no short name
    """
    unknown = sorted(set(datasets) - set(SHORT_NAMES))
    if unknown:
        raise KeyError(f"no short name for {unknown}: add it to SHORT_NAMES")
    return "+".join(short for name, short in SHORT_NAMES.items() if name in datasets)
