"""Procedural generator for executable task specifications.

Environment-first (research plan section 5.3 step 2): we build a filesystem
state and its postconditions, then phrase the natural-language request for it.
Generating a command first and paraphrasing it produces artificially explicit
instructions, so we never do that.

Every family produces variants sharing one user_request but differing in
filenames-that-do-not-appear-in-the-request, content, ordering and distractors.
A command only becomes a verified positive if it passes on all variants
(plan section 5.3 step 6), which kills solutions that memorised the fixture.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Callable

from .spec import OraclePostconditions, Provenance, TaskSpec

LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
NOUNS = [
    "disk", "memory", "socket", "cache", "queue", "worker", "index", "session",
    "token", "buffer", "thread", "packet", "shard", "cursor", "daemon",
]
VERBS = ["failed", "ready", "retrying", "expired", "flushed", "stalled", "started"]
DIRS = ["logs", "var", "srv", "data", "reports", "archive", "inbox", "spool"]
STEMS = ["app", "server", "cron", "audit", "gateway", "batch", "sync", "worker"]
NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]
CITIES = ["lisbon", "oslo", "quito", "dakar", "hanoi", "lima", "sofia", "perth"]


def _rng(family: str, concept_seed: int, variant: int) -> random.Random:
    digest = hashlib.sha256(f"{family}:{concept_seed}:{variant}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _distractors(rng: random.Random, base: str, count: int = 2) -> dict[str, str]:
    """Irrelevant files that must not change the answer."""
    out: dict[str, str] = {}
    for _ in range(count):
        name = f"{base}/{rng.choice(STEMS)}_{rng.randrange(100, 999)}.bak"
        out[name] = f"{rng.choice(NOUNS)} {rng.choice(VERBS)}\n"
    return out


# --- families ---------------------------------------------------------------
# Each returns (files, dirs, user_request, oracle_kwargs, accepted, tags, risk)


def f_filter_level(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("filter_level", concept_seed, variant)
    crng = _rng("filter_level", concept_seed, 0)
    directory, stem = crng.choice(DIRS), crng.choice(STEMS)
    level = crng.choice(["WARN", "ERROR"])
    path = f"{directory}/{stem}.log"

    lines = []
    for _ in range(rng.randrange(8, 16)):
        lvl = rng.choice(LEVELS)
        lines.append(f"{lvl} {rng.choice(NOUNS)} {rng.choice(VERBS)}")
    if not any(l.startswith(level) for l in lines):
        lines.insert(rng.randrange(len(lines)), f"{level} {rng.choice(NOUNS)} stalled")

    files = {path: "\n".join(lines) + "\n"}
    files.update(_distractors(rng, directory))
    expected = "\n".join(l for l in lines if l.startswith(level + " ")) + "\n"
    return TaskSpec(
        setup_files=files,
        setup_dirs=[directory],
        user_request=f"Print only the {level} messages from {path}.",
        oracle=OraclePostconditions(stdout_exact=expected, filesystem_unchanged=True),
        accepted_commands=[f"grep '^{level} ' {path}", f"awk '/^{level} /' {path}"],
        difficulty_tags=["filtering", "single_utility"],
        provenance=Provenance(template_family="filter_level", seed=concept_seed),
    ).finalize()


def f_count_extension(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("count_extension", concept_seed, variant)
    crng = _rng("count_extension", concept_seed, 0)
    directory = crng.choice(DIRS)
    ext = crng.choice(["json", "csv", "txt", "yaml"])
    other = crng.choice([e for e in ["json", "csv", "txt", "yaml"] if e != ext])

    files: dict[str, str] = {}
    n_target = rng.randrange(2, 7)
    for _ in range(n_target):
        files[f"{directory}/{rng.choice(NAMES)}_{rng.randrange(10, 99)}.{ext}"] = "x\n"
    for _ in range(rng.randrange(1, 4)):
        files[f"{directory}/{rng.choice(CITIES)}_{rng.randrange(10, 99)}.{other}"] = "y\n"
    n_target = len([k for k in files if k.endswith(f".{ext}")])

    return TaskSpec(
        setup_files=files,
        setup_dirs=[directory],
        user_request=f"How many .{ext} files are in the {directory} directory? Print just the number.",
        oracle=OraclePostconditions(stdout_exact=f"{n_target}\n", filesystem_unchanged=True),
        accepted_commands=[
            f"ls {directory}/*.{ext} | wc -l",
            f"find {directory} -maxdepth 1 -name '*.{ext}' | wc -l",
        ],
        difficulty_tags=["counting", "globbing", "pipeline"],
        provenance=Provenance(template_family="count_extension", seed=concept_seed),
    ).finalize()


def f_find_extension(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("find_extension", concept_seed, variant)
    crng = _rng("find_extension", concept_seed, 0)
    root = crng.choice(DIRS)
    ext = crng.choice(["log", "conf", "md"])
    sub = f"{root}/{crng.choice(STEMS)}"

    files: dict[str, str] = {}
    targets: list[str] = []
    for _ in range(rng.randrange(2, 5)):
        p = f"{root}/{rng.choice(NAMES)}{rng.randrange(1, 40)}.{ext}"
        files[p] = "data\n"
        targets.append(p)
    for _ in range(rng.randrange(1, 4)):
        p = f"{sub}/{rng.choice(CITIES)}{rng.randrange(1, 40)}.{ext}"
        files[p] = "data\n"
        targets.append(p)
    for _ in range(2):
        files[f"{sub}/{rng.choice(CITIES)}.tmp"] = "noise\n"

    return TaskSpec(
        setup_files=files,
        setup_dirs=[root, sub],
        user_request=(
            f"List the paths of every .{ext} file under {root}, including "
            "files in subdirectories."
        ),
        oracle=OraclePostconditions(
            stdout_lines_unordered=sorted(f"./{t}" for t in set(targets)),
            filesystem_unchanged=True,
        ),
        accepted_commands=[
            f"find {root} -type f -name '*.{ext}'",
            f"find ./{root} -type f -name '*.{ext}'",
        ],
        difficulty_tags=["recursion", "find", "path_depth"],
        provenance=Provenance(template_family="find_extension", seed=concept_seed),
    ).finalize()


def f_csv_column(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("csv_column", concept_seed, variant)
    crng = _rng("csv_column", concept_seed, 0)
    directory = crng.choice(DIRS)
    path = f"{directory}/{crng.choice(STEMS)}.csv"
    headers = ["id", "name", "region", "score"]
    col = crng.randrange(2, 5)  # 1-indexed, never the id column

    rows = []
    for i in range(rng.randrange(4, 9)):
        rows.append([str(i + 1), rng.choice(NAMES), rng.choice(CITIES), str(rng.randrange(1, 100))])
    content = ",".join(headers) + "\n" + "\n".join(",".join(r) for r in rows) + "\n"
    expected = "\n".join(r[col - 1] for r in rows) + "\n"

    return TaskSpec(
        setup_files={path: content},
        setup_dirs=[directory],
        user_request=(
            f"Print the {headers[col - 1]} column from {path}, one value per line, "
            "without the header row."
        ),
        oracle=OraclePostconditions(stdout_exact=expected, filesystem_unchanged=True),
        accepted_commands=[
            f"tail -n +2 {path} | cut -d, -f{col}",
            f"awk -F, 'NR>1{{print ${col}}}' {path}",
        ],
        difficulty_tags=["field_extraction", "delimiter", "pipeline"],
        provenance=Provenance(template_family="csv_column", seed=concept_seed),
    ).finalize()


def f_count_matches(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("count_matches", concept_seed, variant)
    crng = _rng("count_matches", concept_seed, 0)
    directory = crng.choice(DIRS)
    path = f"{directory}/{crng.choice(STEMS)}.txt"
    word = crng.choice(NOUNS)

    lines = []
    hits = 0
    for _ in range(rng.randrange(6, 15)):
        if rng.random() < 0.4:
            lines.append(f"{rng.choice(VERBS)} {word} handler")
            hits += 1
        else:
            lines.append(f"{rng.choice(VERBS)} {rng.choice([n for n in NOUNS if n != word])}")
    if hits == 0:
        lines.append(f"restarted {word} handler")
        hits = 1

    return TaskSpec(
        setup_files={path: "\n".join(lines) + "\n"},
        setup_dirs=[directory],
        user_request=f"Count how many lines in {path} contain the word '{word}'. Print only the count.",
        oracle=OraclePostconditions(stdout_exact=f"{hits}\n", filesystem_unchanged=True),
        accepted_commands=[f"grep -c '{word}' {path}", f"grep -c -- '{word}' {path}"],
        difficulty_tags=["counting", "filtering"],
        provenance=Provenance(template_family="count_matches", seed=concept_seed),
    ).finalize()


def f_top_frequency(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("top_frequency", concept_seed, variant)
    crng = _rng("top_frequency", concept_seed, 0)
    directory = crng.choice(DIRS)
    path = f"{directory}/{crng.choice(STEMS)}_access.txt"
    top_n = crng.choice([2, 3])

    # Build a strict frequency ordering so the expected output is unambiguous.
    pool = rng.sample(CITIES, 5)
    counts = {city: 12 - 2 * i for i, city in enumerate(pool)}
    entries = [c for c, n in counts.items() for _ in range(n)]
    rng.shuffle(entries)
    expected = "\n".join(pool[:top_n]) + "\n"

    return TaskSpec(
        setup_files={path: "\n".join(entries) + "\n"},
        setup_dirs=[directory],
        user_request=(
            f"Print the {top_n} most frequent values in {path}, most frequent first, "
            "one per line and without the counts."
        ),
        oracle=OraclePostconditions(stdout_exact=expected, filesystem_unchanged=True),
        accepted_commands=[
            f"sort {path} | uniq -c | sort -rn | head -n {top_n} | awk '{{print $2}}'",
            f"sort {path} | uniq -c | sort -k1,1nr | head -{top_n} | sed 's/^ *[0-9]* //'",
        ],
        difficulty_tags=["pipeline", "sorting", "aggregation", "multi_utility"],
        provenance=Provenance(template_family="top_frequency", seed=concept_seed),
    ).finalize()


def f_head_lines(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("head_lines", concept_seed, variant)
    crng = _rng("head_lines", concept_seed, 0)
    directory = crng.choice(DIRS)
    path = f"{directory}/{crng.choice(STEMS)}_report.txt"
    n = crng.randrange(2, 6)

    lines = [f"{rng.choice(NOUNS)}-{rng.randrange(100, 999)}" for _ in range(rng.randrange(8, 20))]
    return TaskSpec(
        setup_files={path: "\n".join(lines) + "\n"},
        setup_dirs=[directory],
        user_request=f"Show the first {n} lines of {path}.",
        oracle=OraclePostconditions(
            stdout_exact="\n".join(lines[:n]) + "\n", filesystem_unchanged=True
        ),
        accepted_commands=[f"head -n {n} {path}", f"sed -n '1,{n}p' {path}"],
        difficulty_tags=["slicing", "single_utility"],
        provenance=Provenance(template_family="head_lines", seed=concept_seed),
    ).finalize()


def f_sorted_unique(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("sorted_unique", concept_seed, variant)
    crng = _rng("sorted_unique", concept_seed, 0)
    directory = crng.choice(DIRS)
    path = f"{directory}/{crng.choice(STEMS)}_hosts.txt"

    pool = rng.sample(CITIES, rng.randrange(3, 6))
    entries = [rng.choice(pool) for _ in range(rng.randrange(8, 16))]
    entries += pool  # guarantee every value appears
    rng.shuffle(entries)
    expected = "\n".join(sorted(set(entries))) + "\n"

    return TaskSpec(
        setup_files={path: "\n".join(entries) + "\n"},
        setup_dirs=[directory],
        user_request=f"Print the unique lines of {path} in alphabetical order.",
        oracle=OraclePostconditions(stdout_exact=expected, filesystem_unchanged=True),
        accepted_commands=[f"sort -u {path}", f"sort {path} | uniq"],
        difficulty_tags=["sorting", "dedup"],
        provenance=Provenance(template_family="sorted_unique", seed=concept_seed),
    ).finalize()


def f_replace_in_file(concept_seed: int, variant: int) -> TaskSpec:
    """Local-write family: the oracle is a post-state file hash, not stdout."""
    rng = _rng("replace_in_file", concept_seed, variant)
    crng = _rng("replace_in_file", concept_seed, 0)
    directory = crng.choice(DIRS)
    path = f"{directory}/{crng.choice(STEMS)}.conf"
    old, new = crng.sample(CITIES, 2)

    lines = []
    for _ in range(rng.randrange(4, 9)):
        key = rng.choice(NOUNS)
        value = old if rng.random() < 0.5 else rng.choice(NAMES)
        lines.append(f"{key}={value}")
    if not any(l.endswith(f"={old}") for l in lines):
        lines.append(f"{rng.choice(NOUNS)}={old}")
    content = "\n".join(lines) + "\n"
    after = content.replace(old, new)

    return TaskSpec(
        setup_files={path: content},
        setup_dirs=[directory],
        user_request=f"In {path}, replace every occurrence of '{old}' with '{new}', editing the file in place.",
        risk_class="local_write",
        allowed_side_effects=["stdout", "stderr", "filesystem_write"],
        forbidden_side_effects=["filesystem_delete", "network", "privilege_change"],
        oracle=OraclePostconditions(
            filesystem_unchanged=False,
            expected_files={path: hashlib.sha256(after.encode()).hexdigest()},
        ),
        accepted_commands=[
            f"sed -i 's/{old}/{new}/g' {path}",
            f"sed -i -e 's/{old}/{new}/g' {path}",
        ],
        difficulty_tags=["in_place_edit", "substitution", "side_effect"],
        provenance=Provenance(template_family="replace_in_file", seed=concept_seed),
    ).finalize()


def f_largest_file(concept_seed: int, variant: int) -> TaskSpec:
    rng = _rng("largest_file", concept_seed, variant)
    crng = _rng("largest_file", concept_seed, 0)
    directory = crng.choice(DIRS)

    files: dict[str, str] = {}
    sizes: dict[str, int] = {}
    names = rng.sample(NAMES, rng.randrange(3, 6))
    for i, name in enumerate(names):
        size = 40 + i * 37 + rng.randrange(0, 11)
        p = f"{directory}/{name}.dat"
        files[p] = "z" * size + "\n"
        sizes[p] = size
    biggest = max(sizes, key=lambda k: sizes[k])

    return TaskSpec(
        setup_files=files,
        setup_dirs=[directory],
        user_request=(
            f"Which file in {directory} is the largest? Print just its name "
            "(no size, no path prefix)."
        ),
        oracle=OraclePostconditions(
            stdout_exact=biggest.split("/")[-1] + "\n", filesystem_unchanged=True
        ),
        accepted_commands=[
            f"ls -S {directory} | head -n 1",
            f"du -b {directory}/* | sort -rn | head -n 1 | awk '{{print $2}}' | xargs basename",
        ],
        difficulty_tags=["sorting", "metadata", "pipeline"],
        provenance=Provenance(template_family="largest_file", seed=concept_seed),
    ).finalize()


FAMILIES: dict[str, Callable[[int, int], TaskSpec]] = {
    "filter_level": f_filter_level,
    "count_extension": f_count_extension,
    "find_extension": f_find_extension,
    "csv_column": f_csv_column,
    "count_matches": f_count_matches,
    "top_frequency": f_top_frequency,
    "head_lines": f_head_lines,
    "sorted_unique": f_sorted_unique,
    "replace_in_file": f_replace_in_file,
    "largest_file": f_largest_file,
}


def build(family: str, concept_seed: int, variant: int = 0) -> TaskSpec:
    return FAMILIES[family](concept_seed, variant)


def variants(spec: TaskSpec, count: int) -> list[TaskSpec]:
    """Replay instances of the same task concept with randomized fixtures.

    LLM-authored tasks have no procedural generator behind them, so they cannot
    be re-instantiated with fresh fixtures; they get no replay variants and
    rely on cross-command agreement instead (see authoring.py).
    """
    fam, seed = spec.provenance.template_family, spec.provenance.seed
    if fam not in FAMILIES:
        return []
    return [build(fam, seed, v) for v in range(1, count + 1)]


def generate_tasks(n: int, seed: int = 0) -> list[TaskSpec]:
    """Round-robin across families so every fold of a grouped CV split is usable."""
    families = sorted(FAMILIES)
    out: list[TaskSpec] = []
    seen: set[str] = set()
    concept = seed
    while len(out) < n:
        for fam in families:
            if len(out) >= n:
                break
            spec = build(fam, concept, 0)
            if spec.task_id in seen:
                continue
            seen.add(spec.task_id)
            out.append(spec)
        concept += 1
    return out


def family_distribution(specs: list[TaskSpec]) -> Counter:
    return Counter(s.provenance.template_family for s in specs)
