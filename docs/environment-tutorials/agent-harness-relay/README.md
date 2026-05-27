# Agent Harness Relay Artifacts

This directory holds one-task SWE-bench artifacts for comparing Gym rollout output with optional NeMoRelay capture.

- `hermes/`: runbook, regeneration script, and artifacts for Hermes on `django__django-13741`.
- `openclaw/`: runbook, regeneration script, and artifacts for OpenClaw on `django__django-13741`.
- `opencode/`: captured artifacts only; kept as reference data because OpenCode hook coverage is less clear for this POC.

Each harness stores generated JSON/JSONL payloads under its `artifacts/` directory so the docs remain readable while the data stays easy to copy into Phoenix or other ATIF consumers.
