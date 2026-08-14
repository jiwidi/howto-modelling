"""Rootless sandbox execution with complete before/after filesystem snapshots.

Research-plan section 5.5: the scorer must observe *every* change, not just
stdout and a couple of files, otherwise commands that pass stdout while
corrupting the workspace are scored as correct.

Isolation uses bubblewrap: no network (--unshare-all), read-only system paths,
a single disposable writable workspace, and rlimit-based resource caps.
"""

from __future__ import annotations

import hashlib
import os
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import LIMITS
from .spec import ExecutionResult, TaskSpec

BWRAP = shutil.which("bwrap")
WORKDIR_IN_SANDBOX = "/work"

# System paths bound read-only when they exist (distro layouts differ).
_RO_CANDIDATES = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")


class SandboxUnavailable(RuntimeError):
    pass


def _snapshot(root: Path) -> dict[str, tuple]:
    """Hash every path under root: content, mode, type, symlink target."""
    snap: dict[str, tuple] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + list(filenames):
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            try:
                st = p.lstat()
                if p.is_symlink():
                    snap[rel] = ("symlink", os.readlink(p), st.st_mode & 0o7777)
                elif p.is_dir():
                    snap[rel] = ("dir", "", st.st_mode & 0o7777)
                else:
                    digest = hashlib.sha256(p.read_bytes()).hexdigest()
                    snap[rel] = ("file", digest, st.st_mode & 0o7777)
            except OSError as exc:  # unreadable entry is itself a change signal
                snap[rel] = ("error", str(exc), 0)
    return snap


def _diff(before: dict, after: dict) -> tuple[list[str], list[str], list[str]]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return added, removed, modified


def materialize(spec: TaskSpec, root: Path) -> None:
    """Build the task's initial filesystem state under root."""
    for d in spec.setup_dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    for relpath, content in spec.setup_files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for relpath, mode in spec.setup_modes.items():
        target = root / relpath
        if target.exists():
            target.chmod(mode)


def _uid_task_count() -> int:
    """RLIMIT_NPROC is enforced per-UID across the whole machine and counts
    *threads*, not processes, so the fork-bomb cap has to sit above everything
    this login already runs -- otherwise bwrap itself fails to clone (EAGAIN)."""
    uid = os.getuid()
    count = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            count += len(os.listdir(f"/proc/{entry.name}/task"))
        except OSError:
            continue
    return count


_NPROC_BASELINE = _uid_task_count()


def _preexec_limits() -> None:
    mb = 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS.max_file_size_mb * mb,) * 2)
    hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
    want = _NPROC_BASELINE + LIMITS.max_processes
    if hard != resource.RLIM_INFINITY:
        want = min(want, hard)
    resource.setrlimit(resource.RLIMIT_NPROC, (want, hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setsid()


def build_bwrap_argv(host_workdir: Path) -> list[str]:
    if not BWRAP:
        raise SandboxUnavailable(
            "bubblewrap (bwrap) not found; refusing to execute model-generated "
            "commands without isolation."
        )
    argv = [
        BWRAP,
        "--unshare-all",  # includes network namespace: no egress
        "--die-with-parent",
        "--new-session",  # blocks TIOCSTI terminal injection
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--bind", str(host_workdir), WORKDIR_IN_SANDBOX,
        "--chdir", WORKDIR_IN_SANDBOX,
    ]
    for path in _RO_CANDIDATES:
        if os.path.islink(path) or os.path.isdir(path):
            argv += ["--ro-bind", path, path]
    for key, value in LIMITS.env.items():
        argv += ["--setenv", key, value]
    return argv


def run_command(spec: TaskSpec, command: str) -> ExecutionResult:
    """Execute one candidate command in a fresh sandbox and record all effects."""
    result = ExecutionResult(command=command)
    tmp = Path(tempfile.mkdtemp(prefix="howto-"))
    workdir = tmp / "work"
    workdir.mkdir()
    try:
        materialize(spec, workdir)
        before = _snapshot(workdir)

        argv = build_bwrap_argv(workdir) + ["--", "/bin/bash", "-c", command]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                timeout=LIMITS.timeout_s,
                preexec_fn=_preexec_limits,
                check=False,
            )
            result.exit_code = proc.returncode
            result.stdout = proc.stdout[: LIMITS.max_output_bytes].decode(
                "utf-8", "replace"
            )
            result.stderr = proc.stderr[: LIMITS.max_output_bytes].decode(
                "utf-8", "replace"
            )
        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.exit_code = None
        except SandboxUnavailable:
            raise
        except OSError as exc:
            result.sandbox_error = str(exc)

        after = _snapshot(workdir)
        result.files_added, result.files_removed, result.files_modified = _diff(
            before, after
        )

        effects: list[str] = []
        if result.stdout:
            effects.append("stdout")
        if result.stderr:
            effects.append("stderr")
        if result.files_added or result.files_modified:
            effects.append("filesystem_write")
        if result.files_removed:
            effects.append("filesystem_delete")
        result.observed_side_effects = effects
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def selftest() -> bool:
    """Confirm isolation actually holds before any model output is executed."""
    spec = TaskSpec(setup_files={"probe.txt": "hello\n"}).finalize()
    ok = run_command(spec, "cat probe.txt").stdout == "hello\n"
    # Network must be unreachable inside the namespace.
    net = run_command(spec, "getent hosts example.com >/dev/null 2>&1 && echo UP || echo DOWN")
    isolated = "DOWN" in net.stdout
    # The host filesystem must not be writable.
    host = run_command(spec, "touch /usr/howto-probe 2>/dev/null && echo WRITABLE || echo RO")
    ro = "RO" in host.stdout
    return ok and isolated and ro
