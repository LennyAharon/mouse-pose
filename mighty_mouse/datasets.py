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
