# Codex Agent

This response-api agent runs the Codex CLI through Gym. It is intended to let the
same SWE-bench task used by the Relay harness POC run with Codex while preserving
Gym's normal rollout, patch collection, and optional SWE-bench verification flow.

The first smoke dataset is `django__django-13741`:

```bash
ng_collect_rollouts \
  +agent_name=codex_agent \
  +input_jsonl_fpath=responses_api_agents/codex_agent/data/django_13741_smoke.jsonl \
  +output_jsonl_fpath=outputs/codex_agent/django_13741_smoke.jsonl \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

By default the agent uses `codex exec --json --sandbox workspace-write
--ephemeral` and sets `approval_policy="never"` through a Codex config override.
It does not force a model, so Codex can use the local account/profile default.
Provider and model settings should come from the local Codex config/profile or
from `model`, `openai_api_key`, `openai_base_url`, and `config_overrides` in
`configs/codex_agent.yaml`.

When `nemo_relay.enabled=true`, the agent uses Relay's native Codex hook path
instead of `codex exec --json`: Gym launches top-level Codex in a PTY and wraps
it with `nemo-relay run --agent codex`. This produces Relay ATOF/ATIF artifacts
with model, shell, and patch events while preserving Gym rollout metadata.
