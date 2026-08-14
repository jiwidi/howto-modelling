"""Deterministic verification: postconditions + complete side-effect accounting.

Correctness is decided here, never by an LLM. A candidate is a verified positive
only when it satisfies the postconditions on the canonical task AND on every
randomized replay variant, with no forbidden side effect on any of them.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

from .config import LIMITS
from .sandbox import run_command
from .spec import ExecutionResult, TaskSpec, Verdict
from .tasks import variants

FENCE = re.compile(r"^\s*```(?:bash|sh|shell)?\s*|\s*```\s*$", re.MULTILINE)
PREFIX = re.compile(r"^\s*(?:\$|#|>)\s+")


def clean_command(raw: str) -> str:
    """Strip the formatting a chat model tends to add around a command."""
    text = FENCE.sub("", raw or "").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    # Keep the first non-comment line: the deployment contract is one command.
    for line in lines:
        candidate = PREFIX.sub("", line).strip()
        if candidate and not candidate.startswith("#"):
            return candidate
    return PREFIX.sub("", lines[0]).strip()


def syntax_ok(command: str) -> bool:
    """bash -n parse check; cheap static gate before we spend a sandbox run."""
    if not command.strip():
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=True) as fh:
        fh.write(command + "\n")
        fh.flush()
        proc = subprocess.run(
            ["/bin/bash", "-n", fh.name], capture_output=True, timeout=10, check=False
        )
    return proc.returncode == 0


DANGEROUS = re.compile(
    r"(?:^|[;&|`(]\s*)\s*(?:sudo|su\b|shutdown|reboot|mkfs|dd\s+if=|chown\s+-R\s+/"
    r"|rm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+/(?:\s|$)|:\(\)\s*\{|curl|wget|nc\b|ssh\b)",
    re.IGNORECASE,
)


def static_risk(command: str) -> str:
    """Advisory AST-free risk screen, mirroring the shipped safety checker."""
    if DANGEROUS.search(command):
        return "high"
    if re.search(r"\brm\b|\bmv\b|>\s*\S|\bsed\b[^|]*-i|\btruncate\b|\bchmod\b", command):
        return "write"
    return "read_only"


def _check_stdout(spec: TaskSpec, res: ExecutionResult) -> tuple[int, int, str]:
    """Returns (passed_checks, total_checks, first_failure_reason)."""
    passed = total = 0
    reason = ""
    o = spec.oracle

    if o.stdout_exact is not None:
        total += 1
        if res.stdout == o.stdout_exact:
            passed += 1
        elif res.stdout.strip() == o.stdout_exact.strip():
            passed += 1  # trailing-newline difference only
        else:
            reason = reason or "stdout_mismatch"

    if o.stdout_lines_unordered is not None:
        total += 1
        got = sorted(l for l in res.stdout.splitlines() if l.strip())
        want = sorted(o.stdout_lines_unordered)
        # ./x and x denote the same path; normalise before comparing.
        norm = lambda xs: sorted(x[2:] if x.startswith("./") else x for x in xs)
        if norm(got) == norm(want):
            passed += 1
        else:
            reason = reason or "stdout_set_mismatch"

    if o.exit_code is not None:
        total += 1
        if res.exit_code == o.exit_code:
            passed += 1
        else:
            reason = reason or ("timeout" if res.timed_out else "exit_code")

    return passed, total, reason


def _check_filesystem(
    spec: TaskSpec, res: ExecutionResult, workspace_files: dict[str, str]
) -> tuple[int, int, str]:
    passed = total = 0
    reason = ""
    o = spec.oracle

    if o.filesystem_unchanged:
        total += 1
        if not res.filesystem_changed:
            passed += 1
        else:
            reason = reason or "unexpected_filesystem_change"

    for relpath, want_hash in o.expected_files.items():
        total += 1
        got = workspace_files.get(relpath)
        if got == want_hash:
            passed += 1
        else:
            reason = reason or "wrong_file_content"

    for relpath in o.absent_files:
        total += 1
        if relpath not in workspace_files:
            passed += 1
        else:
            reason = reason or "file_should_be_absent"

    return passed, total, reason


def _post_state(spec: TaskSpec, command: str) -> tuple[ExecutionResult, dict[str, str]]:
    """Run once and also hash the surviving workspace files for file oracles."""
    import shutil

    from .sandbox import _snapshot, build_bwrap_argv, materialize

    tmp = Path(tempfile.mkdtemp(prefix="howto-v-"))
    workdir = tmp / "work"
    workdir.mkdir()
    try:
        materialize(spec, workdir)
        before = _snapshot(workdir)
        res = ExecutionResult(command=command)
        argv = build_bwrap_argv(workdir) + ["--", "/bin/bash", "-c", command]
        try:
            from .sandbox import _preexec_limits

            proc = subprocess.run(
                argv,
                capture_output=True,
                timeout=LIMITS.timeout_s,
                preexec_fn=_preexec_limits,
                check=False,
            )
            res.exit_code = proc.returncode
            res.stdout = proc.stdout[: LIMITS.max_output_bytes].decode("utf-8", "replace")
            res.stderr = proc.stderr[: LIMITS.max_output_bytes].decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            res.timed_out = True
        except OSError as exc:
            res.sandbox_error = str(exc)

        after = _snapshot(workdir)
        res.files_added = sorted(set(after) - set(before))
        res.files_removed = sorted(set(before) - set(after))
        res.files_modified = sorted(
            k for k in set(before) & set(after) if before[k] != after[k]
        )
        effects = []
        if res.stdout:
            effects.append("stdout")
        if res.stderr:
            effects.append("stderr")
        if res.files_added or res.files_modified:
            effects.append("filesystem_write")
        if res.files_removed:
            effects.append("filesystem_delete")
        res.observed_side_effects = effects

        file_hashes: dict[str, str] = {}
        for path in workdir.rglob("*"):
            if path.is_file() and not path.is_symlink():
                rel = str(path.relative_to(workdir))
                file_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        return res, file_hashes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify(spec: TaskSpec, command: str, replay_seeds: int | None = None) -> Verdict:
    """Full verdict for one candidate command across canonical + replay variants."""
    command = clean_command(command)
    v = Verdict(command=command)
    if not command:
        v.failure_category = "empty_output"
        v.syntax_valid = False
        return v
    if not syntax_ok(command):
        v.syntax_valid = False
        v.failure_category = "syntax_error"
        return v

    n_replays = LIMITS.replay_seeds if replay_seeds is None else replay_seeds
    instances = [spec] + variants(spec, n_replays)
    v.seeds_total = len(instances)

    fractions: list[float] = []
    for inst in instances:
        res, files = _post_state(inst, command)
        v.results.append(res)

        forbidden = set(inst.forbidden_side_effects) & set(res.observed_side_effects)
        if forbidden:
            v.forbidden_side_effect = True
            v.failure_category = "forbidden_side_effect"
            v.detail = f"observed {sorted(forbidden)}"

        p1, t1, r1 = _check_stdout(inst, res)
        p2, t2, r2 = _check_filesystem(inst, res, files)
        total = t1 + t2
        frac = (p1 + p2) / total if total else 0.0
        fractions.append(frac)
        if frac >= 1.0 and not forbidden:
            v.seeds_passed += 1
        elif v.failure_category == "none":
            v.failure_category = r1 or r2 or "unknown"
            v.detail = v.detail or f"stdout={res.stdout[:200]!r}"

    v.postconditions_passed = sum(fractions) / len(fractions)
    v.passed = v.seeds_passed == v.seeds_total and not v.forbidden_side_effect
    if v.passed:
        v.failure_category = "none"
    elif v.seeds_passed > 0 and v.failure_category in ("none", "unknown"):
        v.failure_category = "fixture_overfit"  # passed some seeds only
    return v


def validate_task(spec: TaskSpec) -> tuple[bool, str]:
    """A task is only admitted if its own reference solutions agree everywhere.

    Two textually different accepted commands must both satisfy the oracle on
    the canonical task and all replay variants. This catches broken generators
    before any model output is scored against them.
    """
    if len(spec.accepted_commands) < 2:
        return False, "needs_two_reference_commands"
    for cmd in spec.accepted_commands:
        v = verify(spec, cmd)
        if not v.passed:
            return False, f"reference_failed:{cmd!r}:{v.failure_category}:{v.detail[:120]}"
    return True, "ok"
