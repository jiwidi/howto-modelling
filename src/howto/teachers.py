"""Closed-weight teachers and blind judges (OpenAI + Anthropic).

Two distinct roles, deliberately kept separate:

* proposer  -- generates candidate commands for a task. Execution decides whether
               a proposal becomes training data; the model never self-certifies.
* judge     -- given the command AND its execution evidence, returns a strict
               JSON verdict. Used only to measure judge-vs-execution agreement
               (plan section 11 report B), never to overrule the executor.

NOTE ON TERMS: the research plan flags that standard OpenAI/Anthropic terms
restrict using outputs to train competing models. This PoC uses them because it
was explicitly requested for validation; clear it contractually (or swap in the
self-hosted open-weight committee) before any published training run.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import PROVIDERS, ProviderSpec
from .spec import Candidate, TaskSpec, Verdict

PROPOSE_SYSTEM = (
    "You are a shell expert. Given a task description and the initial state of a "
    "workspace directory, reply with the single bash command that accomplishes it.\n"
    "Respond with STRICT JSON only:\n"
    '{"command": "<one line of bash>", "risk": "read_only|local_write|destructive|'
    'privileged|network|process_control", "requires_confirmation": true|false, '
    '"assumptions": ["..."]}\n'
    "The command runs with the workspace as the working directory. Use relative paths. "
    "No sudo, no network, no markdown fences."
)

JUDGE_SYSTEM = (
    "You are a blind evaluator of shell commands. You receive a request, the "
    "workspace state, a candidate command and its real execution evidence. "
    "Decide whether the candidate genuinely satisfies the request.\n"
    "Respond with STRICT JSON only:\n"
    '{"semantically_correct": true|false, "safe_under_constraints": true|false, '
    '"reasonable_assumptions": true|false, "failure_category": "none|wrong_utility|'
    'wrong_flag|quoting|pipeline|path|portability|side_effect|other", '
    '"confidence": 0.0-1.0}'
)


def _client(spec: ProviderSpec):
    if spec.provider == "gemini":
        from google import genai

        from .config import VERTEX_LOCATION, VERTEX_PROJECT

        # Inference only: this client is used exclusively for generateContent.
        return genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
        )
    if spec.provider == "openai":
        from openai import OpenAI

        # This project's key is scoped to the EU data region; the global endpoint
        # rejects it with a 401 "outside project geography" error.
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "https://eu.api.openai.com/v1"),
        )
    from anthropic import Anthropic

    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20), reraise=True)
def _complete(spec: ProviderSpec, system: str, user: str, max_tokens: int = 900) -> str:
    client = _client(spec)
    if spec.provider == "gemini":
        from google.genai import types

        resp = client.models.generate_content(
            model=spec.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                temperature=1.0,
            ),
        )
        return resp.text or ""
    if spec.provider == "openai":
        # GPT-5.x reasoning models: no temperature knob, max_completion_tokens.
        # That budget also covers reasoning tokens, so a tight cap can leave an
        # empty message; keep headroom and hold reasoning effort down.
        resp = client.chat.completions.create(
            model=spec.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max(max_tokens, 4000),
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    # Anthropic has no JSON mode; prefilling the assistant turn with "{" forces
    # the reply to continue as JSON instead of prose or a fenced block.
    resp = client.messages.create(
        model=spec.model,
        max_tokens=max_tokens,
        system=system,
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": "{"},
        ],
    )
    body = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return "{" + body if body and not body.lstrip().startswith("{") else body


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def workspace_view(spec: TaskSpec, max_bytes: int = 900) -> str:
    """A compact, deterministic rendering of the initial state."""
    lines = ["Initial workspace contents (paths are relative to the working directory):"]
    for path in sorted(spec.setup_files):
        content = spec.setup_files[path]
        preview = content if len(content) <= max_bytes else content[:max_bytes] + "...<truncated>"
        lines.append(f"--- {path} ---\n{preview.rstrip()}")
    return "\n".join(lines)


def propose(spec: ProviderSpec, task: TaskSpec, style_hint: str) -> Candidate:
    user = (
        f"Platform: {task.os_profile}, shell: {task.shell_profile}, LC_ALL=C, TZ=UTC.\n\n"
        f"{workspace_view(task)}\n\n"
        f"Request: {task.user_request}\n\n"
        f"Style guidance: {style_hint}"
    )
    raw = _complete(spec, PROPOSE_SYSTEM, user)
    data = _parse_json(raw)
    return Candidate(
        task_id=task.task_id,
        source=spec.key,
        model=spec.model,
        command=str(data.get("command", "")).strip(),
        risk=str(data.get("risk", "unknown")),
        requires_confirmation=bool(data.get("requires_confirmation", False)),
        assumptions=[str(a) for a in (data.get("assumptions") or [])][:4],
    )


def judge(spec: ProviderSpec, task: TaskSpec, command: str, verdict: Verdict) -> dict:
    evidence = verdict.results[0] if verdict.results else None
    user = (
        f"Request: {task.user_request}\n\n{workspace_view(task, 400)}\n\n"
        f"Candidate command: {command}\n"
        f"Exit code: {getattr(evidence, 'exit_code', None)}\n"
        f"stdout: {getattr(evidence, 'stdout', '')[:800]!r}\n"
        f"stderr: {getattr(evidence, 'stderr', '')[:400]!r}\n"
        f"Files added: {getattr(evidence, 'files_added', [])}\n"
        f"Files modified: {getattr(evidence, 'files_modified', [])}\n"
        f"Files removed: {getattr(evidence, 'files_removed', [])}\n"
        f"Allowed side effects: {task.allowed_side_effects}\n"
        f"Forbidden side effects: {task.forbidden_side_effects}"
    )
    return _parse_json(_complete(spec, JUDGE_SYSTEM, user, max_tokens=500))


@dataclass
class Committee:
    """Teacher/judge panel selected on the command line."""

    keys: tuple[str, ...]
    proposals_per_model: int = 2
    max_workers: int = 8

    @property
    def specs(self) -> list[ProviderSpec]:
        return [PROVIDERS[k] for k in self.keys]

    def propose_all(self, task: TaskSpec) -> list[Candidate]:
        jobs = [
            (spec, hint)
            for spec in self.specs
            for hint in spec.style_hints[: self.proposals_per_model]
        ]
        out: list[Candidate] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(propose, s, task, h) for s, h in jobs]
            for fut in futures:
                try:
                    out.append(fut.result())
                except Exception as exc:  # a dead teacher must not kill the run
                    out.append(
                        Candidate(
                            task_id=task.task_id,
                            source="error",
                            model="error",
                            command="",
                            assumptions=[f"proposal_failed: {exc!s}[:200]"],
                        )
                    )
        return out

    def judge_all(self, task: TaskSpec, command: str, verdict: Verdict) -> dict[str, dict]:
        votes: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                spec.key: pool.submit(judge, spec, task, command, verdict)
                for spec in self.specs
            }
            for key, fut in futures.items():
                try:
                    votes[key] = fut.result()
                except Exception as exc:
                    votes[key] = {"error": str(exc)[:200]}
        return votes
