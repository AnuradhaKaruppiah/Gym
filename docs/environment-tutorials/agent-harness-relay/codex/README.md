# Codex Relay Artifacts

Codex runs `django__django-13741` through Gym with native NeMo Relay hook
capture. Gym launches the Codex CLI, Relay wraps it with
`nemo-relay run --agent codex`, and Relay writes the primary artifacts.

## Primary Files

- `artifacts/with-relay.nemo-relay.atof.jsonl`: lossless Relay event stream.
- `artifacts/with-relay.nemo-relay.atif.json`: Relay-projected ATIF trajectory.
- `artifacts/summary.json`: compact run metadata and event counts.

## Secondary Context

- `artifacts/with-relay.gym-rollout.jsonl`: Gym rollout output from the
  Relay-enabled run.
- [RUNBOOK.md](RUNBOOK.md): setup and rerun commands.

## Relay Settings

Relay mode uses the Codex hook integration, not Gym's `codex exec --json`
stdout. The wrapper starts top-level Codex in a PTY with the SWE task as the
prompt, enables Codex hooks, and routes model traffic through the Relay gateway.

For this capture, `OPENAI_API_KEY` was unset when starting Gym so Relay could
forward Codex's normal login auth instead of substituting the API-key route.
That produced a full trajectory with model activity, Bash calls, and
`apply_patch` calls.

## Snapshot

The Relay-enabled Codex run produced a Django patch, ran the targeted auth form
tests, and finished naturally. SWE-bench verification was disabled for artifact
capture, so the Gym rollout reward remains `0.0`.

The checked-in ATOF has 6,847 events, including 114 Bash tool events and 8
`apply_patch` tool events. The ATIF projection has 63 steps with tool calls and
observations suitable for Phoenix visualization.
