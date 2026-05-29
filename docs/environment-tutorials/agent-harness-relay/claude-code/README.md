# Claude Code Relay Artifacts

Claude Code runs `django__django-13741` through the Gym Claude Code wrapper with
an optional NeMoFlow/NeMoRelay wrapper around the Claude CLI. The Relay artifacts
are the primary output: ATOF preserves the raw event stream, and ATIF gives the
normalized trajectory currently projected from those events.

## Primary Files

- `artifacts/with-relay.nemo-relay.atof.jsonl`: lossless Relay event stream.
- `artifacts/with-relay.nemo-relay.atif.json`: Relay-projected ATIF trajectory.
- `artifacts/summary.json`: compact run metadata and counts.

## Secondary Context

- `artifacts/with-relay.gym-rollout.jsonl`: Gym rollout output from the
  Relay-enabled run.
- `artifacts/with-relay.swebench-report.json`: SWE-bench verifier report.
- [RUNBOOK.md](RUNBOOK.md): setup and rerun commands.

## Snapshot

The checked-in Relay-enabled Claude Code run resolves the SWE-bench task with
`reward=1.0` and `swebench_resolved=true`.

Relay-enabled Claude Code currently emits 320 ATOF events and a 12-step ATIF
trajectory. The ATIF has 4 same-step correlated tool observations and no
standalone tool observation steps. The tool sequence is `Grep`, `Grep`, `Read`,
and `Edit`.

The Gym rollout and SWE-bench report are useful supporting context, but the
Relay ATOF/ATIF pair is the artifact shape to use for unified trajectory
analysis across harnesses.
