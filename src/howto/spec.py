"""Executable task specification.

The ground truth of a record is the *postcondition*, not one textual command --
this is the core design point of the research plan (section 5.2). Many textually
unrelated commands can satisfy the same task.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

RiskClass = Literal[
    "read_only",
    "local_write",
    "destructive",
    "privileged",
    "network",
    "process_control",
    "unknown",
]

SideEffect = Literal[
    "stdout",
    "stderr",
    "filesystem_write",
    "filesystem_delete",
    "network",
    "privilege_change",
    "process_control",
]


class OraclePostconditions(BaseModel):
    """What must be true after the command runs for the task to be solved."""

    stdout_exact: str | None = None
    # Order-insensitive comparison of non-empty stdout lines.
    stdout_lines_unordered: list[str] | None = None
    exit_code: int | None = 0
    filesystem_unchanged: bool = True
    # relpath -> expected sha256 of file content after the command.
    expected_files: dict[str, str] = Field(default_factory=dict)
    # relpaths that must not exist afterwards.
    absent_files: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    generator: str = "procedural"
    generator_version: str = "0.1.0"
    template_family: str = ""
    seed: int = 0
    proposer_models: list[str] = Field(default_factory=list)
    human_review_status: str = "unreviewed"


class TaskSpec(BaseModel):
    """One executable NL->shell problem with a deterministic environment."""

    task_id: str = ""
    shell_profile: str = "bash"
    os_profile: str = "linux-gnu"
    locale: str = "C"
    timezone: str = "UTC"

    setup_files: dict[str, str] = Field(default_factory=dict)
    setup_dirs: list[str] = Field(default_factory=list)
    setup_modes: dict[str, int] = Field(default_factory=dict)

    user_request: str = ""
    paraphrases: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    ambiguity_label: Literal["unambiguous", "underspecified", "ambiguous"] = "unambiguous"
    risk_class: RiskClass = "read_only"
    allowed_side_effects: list[SideEffect] = Field(default_factory=lambda: ["stdout"])
    forbidden_side_effects: list[SideEffect] = Field(
        default_factory=lambda: [
            "filesystem_delete",
            "network",
            "privilege_change",
        ]
    )

    oracle: OraclePostconditions = Field(default_factory=OraclePostconditions)
    accepted_commands: list[str] = Field(default_factory=list)
    difficulty_tags: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)

    def content_hash(self) -> str:
        payload: dict[str, Any] = {
            "setup_files": self.setup_files,
            "setup_dirs": sorted(self.setup_dirs),
            "user_request": self.user_request,
            "oracle": self.oracle.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def finalize(self) -> "TaskSpec":
        self.task_id = self.content_hash()[:16]
        return self


class ExecutionResult(BaseModel):
    """Full side-effect record of running one candidate command once."""

    command: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    sandbox_error: str | None = None

    files_added: list[str] = Field(default_factory=list)
    files_removed: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    observed_side_effects: list[str] = Field(default_factory=list)

    @property
    def filesystem_changed(self) -> bool:
        return bool(self.files_added or self.files_removed or self.files_modified)


class Verdict(BaseModel):
    """Aggregated result of replaying one command across randomized seeds."""

    command: str
    passed: bool = False
    postconditions_passed: float = 0.0
    forbidden_side_effect: bool = False
    syntax_valid: bool = True
    failure_category: str = "none"
    seeds_passed: int = 0
    seeds_total: int = 0
    detail: str = ""
    results: list[ExecutionResult] = Field(default_factory=list)


class Candidate(BaseModel):
    """A proposed solution from one teacher, plus its execution verdict."""

    task_id: str
    source: str
    model: str
    command: str
    risk: str = "unknown"
    requires_confirmation: bool = False
    assumptions: list[str] = Field(default_factory=list)
    verdict: Verdict | None = None
    judge_votes: dict[str, dict] = Field(default_factory=dict)
