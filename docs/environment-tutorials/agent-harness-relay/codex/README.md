# Codex Relay Artifacts

Codex runs `django__django-13741` through the Gym Codex wrapper with optional
Python NeMoRelay exporters around `codex exec --json`. The Relay artifacts are
the primary output: ATOF preserves the raw Codex event stream, and ATIF gives the
normalized trajectory currently projected from those events.

## Primary Files

- `artifacts/with-relay.nemo-relay.atof.jsonl`: lossless Relay event stream.
- `artifacts/with-relay.nemo-relay.atif.json`: Relay-projected ATIF trajectory.
- `artifacts/summary.json`: compact run metadata and counts.

## Secondary Context

- `artifacts/with-relay.gym-rollout.jsonl`: Gym rollout output from the
  Relay-enabled run.
- [RUNBOOK.md](RUNBOOK.md): setup and rerun commands.

## Relay Settings

NeMo Relay's Codex documentation recommends the transparent CLI wrapper
(`nemo-relay run -- codex`) for local Codex sessions. That path requires
`codex-cli >= 0.129.0`, enables Codex hooks with `features.hooks = true`,
injects hook forwarding, and routes model traffic through the Relay gateway.

This Gym POC uses a direct adapter path instead: `nemo_relay.enabled=true`
registers Python ATOF/ATIF exporters in the Gym Codex agent and records the
`codex exec --json` stream into Relay. That keeps the run self-contained inside
Gym while preserving Codex-native `command_execution` and `file_change` records
in ATOF. It does not depend on Codex hook support or the Relay gateway provider
alias.

OpenInference export is an optional sibling view. Leave
`nemo_relay.openinference.enabled=false` for file-only ATOF/ATIF capture, or
enable it with an OTLP endpoint such as Phoenix when live span telemetry is
useful.

## Snapshot

The Relay-enabled Codex run produced a patch for the SWE-bench task and finished
naturally. SWE-bench verification was disabled for this capture, so the checked
in rollout has `reward=0.0`.

The ATOF file has 218 events, including 184 raw Codex JSON marks, 78 completed
command executions, and 4 completed file changes. The current ATIF projection is
32 user/agent steps. Codex command and file-change records are preserved in ATOF,
but they are not yet projected as ATIF tool steps.
