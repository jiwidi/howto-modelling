"""LLM-authored executable tasks — how we get thousands of *distinct* samples.

The procedural generator in tasks.py gives airtight oracles but only ten task
shapes: asking it for 5,000 samples yields 500 near-identical variants of each,
which teaches a student almost nothing new. This module trades a little oracle
strength for a lot of diversity.

The model invents the whole task (workspace, request, and two or more textually
different solutions). It is never trusted about what the command *does*: the
sandbox executes the first accepted command to DERIVE the postconditions, then
requires an independent second command to reproduce them exactly. A task where
the two disagree is discarded -- that cross-check is what makes an unverified
generator usable at all.

Guarantee ladder, weakest to strongest:
  1. model claims a command works                      (never trusted alone)
  2. command executes and we record what it did        (derived oracle)
  3. a second, textually different command agrees      (accepted here)
  4. + re-instantiation with randomized fixtures       (procedural tasks only)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass

from .config import ProviderSpec
from .spec import OraclePostconditions, Provenance, TaskSpec
from .teachers import _complete, _parse_json
from .verify import _post_state, clean_command, static_risk, syntax_ok

# Diversity scaffolding. Each authoring call gets a distinct combination so the
# model is pushed away from its favourite handful of grep/find tasks.
DOMAINS = [
    "web server access logs", "application error logs", "CSV exports",
    "JSON-lines event streams", "configuration files", "source code trees",
    "backup archives", "package manifests", "CI build output", "test reports",
    "database dumps", "markdown documentation", "shell dotfiles",
    "systemd unit files", "cron schedules", "TSV data tables",
    "email mbox files", "firewall rules", "DNS zone files", "inventory lists",
    "git-style patch files", "YAML manifests", "INI settings", "log rotation dirs",
    "temperature sensor readings", "financial transaction records",
    "user account listings", "process snapshots", "disk usage reports",
    "translation string files",
]

SKILLS = [
    "filtering lines by a pattern", "counting matches", "extracting a column",
    "sorting numerically", "deduplicating values", "summing a numeric column",
    "finding files by name", "finding files by size", "computing a frequency ranking",
    "joining two files", "replacing text in place", "reformatting whitespace",
    "slicing a line range", "splitting on a delimiter", "case-insensitive search",
    "inverting a match", "extracting the last field", "counting words",
    "listing unique prefixes", "checking a file exists", "comparing two files",
    "renaming by pattern", "computing min and max", "grouping and counting",
    "stripping comments", "extracting a substring", "counting lines per file",
    "filtering by line length", "printing every Nth line", "reversing line order",
]

UTILITY_HINTS = [
    "grep", "awk", "sed", "sort", "uniq", "cut", "find", "wc", "head", "tail",
    "tr", "paste", "join", "comm", "diff", "xargs", "basename", "dirname", "du",
]

DIFFICULTY = [
    ("single_utility", "one utility, no pipeline"),
    ("two_stage", "a two-stage pipeline"),
    ("multi_stage", "a three-or-more stage pipeline"),
]

AUTHOR_SYSTEM = """You invent small, self-contained shell exercises for a training set.

Return STRICT JSON only:
{
  "setup_files": {"relative/path.txt": "file contents\\n", ...},
  "user_request": "one natural sentence asking for the result",
  "accepted_commands": ["<solution 1>", "<solution 2>"],
  "risk_class": "read_only" | "local_write",
  "difficulty_tags": ["...", "..."]
}

Hard requirements:
- 1 to 4 setup files, each under 40 short lines. Inline the real contents.
- The task must be fully determined by the files you provide. No network, no
  sudo, no /etc, no absolute paths, no commands depending on the current date,
  hostname, process list, or file timestamps.
- accepted_commands must contain at least TWO solutions that are textually
  different (different utilities or a different pipeline shape) but produce
  BYTE-IDENTICAL stdout and identical filesystem effects.
- Commands run with the workspace as the working directory; use relative paths.
- Output must be deterministic: no unsorted directory iteration whose order
  could vary, no randomness.
- The user_request must NOT contain the command itself, and must be phrased the
  way a person would ask, naming any file it refers to.
- Prefer GNU/POSIX-portable flags."""


@dataclass
class AuthorRecipe:
    domain: str
    skill: str
    utility: str
    difficulty: str
    difficulty_desc: str
    index: int

    def prompt(self) -> str:
        return (
            f"Invent one shell exercise.\n"
            f"Subject matter: {self.domain}.\n"
            f"Core operation: {self.skill}.\n"
            f"Shape: {self.difficulty_desc}.\n"
            f"Try to involve `{self.utility}` if it fits naturally; if it does "
            f"not, use whatever is right.\n"
            f"Make it concrete and specific -- invent realistic file names and "
            f"contents rather than generic 'file1.txt' placeholders. "
            f"Variation id {self.index}: make this materially different from "
            f"other exercises on the same subject."
        )


def recipes(n: int, seed: int = 0) -> list[AuthorRecipe]:
    """Sample distinct (domain, skill, utility, difficulty) combinations.

    The cross product is 30*30*19*3 = 51,300, comfortably larger than any corpus
    we build, so we sample without replacement rather than cycling indices --
    cycling made combinations recur every few hundred attempts.
    """
    rng = random.Random(seed)
    space = [
        (d, s, u, k)
        for d in range(len(DOMAINS))
        for s in range(len(SKILLS))
        for u in range(len(UTILITY_HINTS))
        for k in range(len(DIFFICULTY))
    ]
    picks = rng.sample(space, min(n, len(space)))
    out: list[AuthorRecipe] = []
    for i, (d, s, u, k) in enumerate(picks):
        diff, desc = DIFFICULTY[k]
        out.append(
            AuthorRecipe(
                domain=DOMAINS[d],
                skill=SKILLS[s],
                utility=UTILITY_HINTS[u],
                difficulty=diff,
                difficulty_desc=desc,
                index=i,
            )
        )
    return out


def _sanity(payload: dict) -> str | None:
    """Cheap structural rejections before we pay for a sandbox run."""
    files = payload.get("setup_files")
    if not isinstance(files, dict) or not (1 <= len(files) <= 6):
        return "bad_setup_files"
    for path, content in files.items():
        if not isinstance(path, str) or not isinstance(content, str):
            return "bad_file_entry"
        if path.startswith("/") or ".." in path:
            return "unsafe_path"
        if len(content) > 20_000:
            return "file_too_large"
    request = payload.get("user_request")
    if not isinstance(request, str) or not (10 <= len(request) <= 400):
        return "bad_request"
    cmds = payload.get("accepted_commands")
    if not isinstance(cmds, list) or len(cmds) < 2:
        return "needs_two_commands"
    cleaned = [clean_command(c) for c in cmds if isinstance(c, str)]
    cleaned = [c for c in cleaned if c]
    if len(cleaned) < 2 or cleaned[0] == cleaned[1]:
        return "commands_not_distinct"
    for c in cleaned[:2]:
        if not syntax_ok(c):
            return "syntax_error"
        if static_risk(c) == "high":
            return "dangerous_command"
    payload["accepted_commands"] = cleaned
    return None


def signature(spec: TaskSpec) -> str:
    """Structural fingerprint for dedup: utilities + pipeline shape.

    Two tasks whose reference solution uses the same utilities in the same
    pipeline arrangement teach the same lesson even if the file names differ,
    so they collapse to one signature.
    """
    cmd = spec.accepted_commands[0] if spec.accepted_commands else ""
    tokens = re.findall(r"[a-zA-Z_][\w-]*", cmd)
    utils = [t for t in tokens if t in set(UTILITY_HINTS) | {"ls", "cat", "echo", "sed"}]
    stages = cmd.count("|")
    return f"{'+'.join(sorted(set(utils)))}#{stages}#{len(spec.setup_files)}"


def realize(payload: dict, recipe: AuthorRecipe) -> tuple[TaskSpec | None, str]:
    """Execute the proposed solutions and derive the oracle from what happened."""
    reason = _sanity(payload)
    if reason:
        return None, reason

    cmd_a, cmd_b = payload["accepted_commands"][:2]
    risk = payload.get("risk_class", "read_only")
    draft = TaskSpec(
        setup_files=payload["setup_files"],
        user_request=payload["user_request"].strip(),
        risk_class=risk if risk in ("read_only", "local_write") else "read_only",
        allowed_side_effects=(
            ["stdout", "stderr", "filesystem_write"]
            if risk == "local_write"
            else ["stdout", "stderr"]
        ),
        forbidden_side_effects=["network", "privilege_change"]
        + ([] if risk == "local_write" else ["filesystem_delete"]),
        accepted_commands=[cmd_a, cmd_b],
        difficulty_tags=[str(t) for t in (payload.get("difficulty_tags") or [])][:4]
        + [recipe.difficulty],
        provenance=Provenance(
            generator="llm_authored",
            template_family="llm_authored",
            seed=recipe.index,
        ),
    ).finalize()

    # 1. Run the primary solution and record everything it did.
    res_a, files_a = _post_state(draft, cmd_a)
    if res_a.sandbox_error or res_a.timed_out:
        return None, "primary_timeout_or_sandbox_error"
    if res_a.exit_code != 0:
        return None, "primary_nonzero_exit"
    if not res_a.stdout.strip() and not res_a.filesystem_changed:
        return None, "primary_no_observable_effect"

    # 2. Determinism: the same command twice must do the same thing.
    res_a2, files_a2 = _post_state(draft, cmd_a)
    if res_a2.stdout != res_a.stdout or files_a2 != files_a:
        return None, "nondeterministic"

    # 3. Derive the oracle from observed behaviour, not from model claims.
    baseline = {
        path: hashlib.sha256(content.encode()).hexdigest()
        for path, content in draft.setup_files.items()
    }
    changed = {p: h for p, h in files_a.items() if baseline.get(p) != h}
    oracle = OraclePostconditions(
        stdout_exact=res_a.stdout,
        exit_code=res_a.exit_code,
        filesystem_unchanged=not res_a.filesystem_changed,
        expected_files=changed if res_a.filesystem_changed else {},
        absent_files=sorted(set(baseline) - set(files_a)),
    )
    draft.oracle = oracle

    # 4. The independent second solution must reproduce it exactly.
    res_b, files_b = _post_state(draft, cmd_b)
    if res_b.stdout != res_a.stdout or res_b.exit_code != res_a.exit_code:
        return None, "commands_disagree_stdout"
    if files_b != files_a:
        return None, "commands_disagree_filesystem"

    forbidden = set(draft.forbidden_side_effects) & set(res_a.observed_side_effects)
    if forbidden:
        return None, f"forbidden_side_effect:{sorted(forbidden)}"

    return draft.finalize(), "ok"


def author_one(spec: ProviderSpec, recipe: AuthorRecipe) -> tuple[TaskSpec | None, str]:
    try:
        raw = _complete(spec, AUTHOR_SYSTEM, recipe.prompt(), max_tokens=2400)
    except Exception as exc:  # a flaky API call must not kill the run
        return None, f"api_error:{type(exc).__name__}"
    payload = _parse_json(raw)
    if not payload:
        return None, "unparseable_json"
    task, reason = realize(payload, recipe)
    if task is not None:
        task.provenance.proposer_models = [spec.model]
    return task, reason
