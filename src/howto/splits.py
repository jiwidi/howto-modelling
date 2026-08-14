"""Leakage-resistant cross-validation splits.

Random row splits inflate results badly on this task: paraphrases and replay
variants of one template share structure, so a random split leaks the answer
shape into the test fold. We group by template family, which is the plan's
"template-disjoint" challenge split (section 5.5) -- the model is always
evaluated on command shapes whose family it never saw in training.
"""

from __future__ import annotations

from collections import defaultdict

from sklearn.model_selection import GroupKFold

from .spec import TaskSpec


def group_key(spec: TaskSpec) -> str:
    """Group tasks by the *shape* of their reference solution.

    Grouping by template_family worked while every task came from the ten
    procedural generators. It breaks on an LLM-authored corpus, where every task
    carries the single family "llm_authored": GroupKFold would then drop 90% of
    the data into one test fold and train on the remainder.

    The structural signature (utility set + pipeline depth + file count) is the
    right grain either way -- two tasks that solve with the same utilities in
    the same pipeline arrangement teach the same lesson, so they must not be
    split across train and test.
    """
    from .authoring import signature  # local import keeps splits.py light

    family = spec.provenance.template_family
    if family != "llm_authored":
        return f"proc:{family}"
    return f"sig:{signature(spec)}"


def make_folds(specs: list[TaskSpec], n_splits: int = 5) -> list[dict[str, list[int]]]:
    """Return [{'train': [...idx], 'test': [...idx]}] with disjoint families."""
    groups = [group_key(s) for s in specs]
    n_groups = len(set(groups))
    if n_groups < 2:
        raise ValueError("need at least two template families to build grouped folds")
    n_splits = min(n_splits, n_groups)
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_idx, test_idx in splitter.split(specs, groups=groups):
        folds.append({"train": train_idx.tolist(), "test": test_idx.tolist()})
    return folds


def fold_report(specs: list[TaskSpec], folds: list[dict]) -> list[dict]:
    out = []
    for i, fold in enumerate(folds):
        train_fams = sorted({group_key(specs[j]) for j in fold["train"]})
        test_fams = sorted({group_key(specs[j]) for j in fold["test"]})
        assert not set(train_fams) & set(test_fams), "family leaked across the split"
        out.append(
            {
                "fold": i,
                "n_train": len(fold["train"]),
                "n_test": len(fold["test"]),
                "train_families": train_fams,
                "test_families": test_fams,
            }
        )
    return out


def by_family(specs: list[TaskSpec]) -> dict[str, list[TaskSpec]]:
    out: dict[str, list[TaskSpec]] = defaultdict(list)
    for s in specs:
        out[group_key(s)].append(s)
    return dict(out)
