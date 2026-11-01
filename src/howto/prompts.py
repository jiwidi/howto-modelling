"""Stable prompt policies shared by training, export, and release gates."""

from __future__ import annotations


TRAINING_PROMPT_POLICY_ID = "howto.training.command-generation.v1"
TRAINING_SYSTEM_PROMPT = (
    "You translate a natural-language request into a single POSIX/bash command line.\n"
    "Rules:\n"
    "- Output ONLY the command. No prose, no markdown fences, no explanation.\n"
    "- The command runs with the working directory set to the task workspace.\n"
    "- Do not use sudo, network access, or destructive operations unless explicitly asked.\n"
)
# Literal by design: changing even the final newline must update the policy ID
# and this reviewed digest rather than silently inheriting a new hash.
TRAINING_SYSTEM_PROMPT_SHA256 = (
    "66b3e1858ab9bf993f0a232ceb079fc256ff41e8177cfd3e949021e55c3a89e1"
)

# Exact Platform::system_prompt(false) strings deployed by HowTo. Policy IDs
# must be versioned whenever either string changes.
PRODUCTION_PROMPT_POLICIES: dict[str, dict[str, str]] = {
    "macos": {
        "policy_id": "howto.command-generation.macos.non-retry.v1",
        "system_prompt_sha256": (
            "a31c2650c55e6aad3c54d81fb7430601ee4dfd5a3f023ca31ae43e8c6d733ab8"
        ),
        "system_prompt": (
            "You are a shell command generator. Output exactly one physical line: "
            "a single shell command that accomplishes the user's request. No prose, "
            "markdown, explanation, comments, or prompt symbol. The target is macOS "
            "with zsh and the BSD command-line tools shipped by Apple. Prefer built-in "
            "macOS tools and avoid GNU/Linux-only flags.\n"
        ),
    },
    "linux": {
        "policy_id": "howto.command-generation.linux.non-retry.v1",
        "system_prompt_sha256": (
            "d17f6a13075035601f90dbd8d0943da39ae8fb5b328a4d7dee2b27e33b789cc9"
        ),
        "system_prompt": (
            "You are a shell command generator. Output exactly one physical line: "
            "a single shell command that accomplishes the user's request. No prose, "
            "markdown, explanation, comments, or prompt symbol. The target is GNU/Linux "
            "with bash and GNU coreutils. Avoid macOS-only commands and BSD-only flags.\n"
        ),
    },
}

# Release archives are built for all four targets. Quality and performance must
# be measured on each native binary; targets sharing an OS reuse its prompt.
PRODUCTION_RUNTIME_TARGETS: dict[str, str] = {
    "macos-arm64": "macos",
    "macos-x86_64": "macos",
    "linux-arm64": "linux",
    "linux-x86_64": "linux",
}
