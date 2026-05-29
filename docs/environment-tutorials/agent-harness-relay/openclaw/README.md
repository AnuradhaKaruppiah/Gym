# OpenClaw Relay Artifacts

OpenClaw runs `django__django-13741` with an optional NeMoFlow plugin loaded by
the harness. The Relay artifacts are the primary output: ATOF keeps the raw event
stream, and ATIF gives a normalized trajectory independent of OpenClaw's native
session format.

## Primary Files

- `artifacts/with-relay.nemo-relay.atof.jsonl`: lossless Relay event stream.
- `artifacts/with-relay.nemo-relay.atif.json`: Relay-projected ATIF trajectory.
- `artifacts/summary.json`: compact run metadata and counts.

## Secondary Context

- `artifacts/with-relay.openclaw.session.jsonl`: native OpenClaw session log from
  the Relay-enabled run.
- `artifacts/without-relay.gym-reconstructed.atif.json`: ATIF reconstructed
  after the fact from Gym rollout output.
- `artifacts/without-relay.openclaw.session.jsonl`: native OpenClaw baseline log.
- [regenerate.py](regenerate.py): refreshes the checked-in artifacts from
  rollout outputs.
- [RUNBOOK.md](RUNBOOK.md): setup and rerun commands.

## Snapshot

Both OpenClaw runs resolve the SWE-bench task with `reward=1.0`.

Relay-enabled OpenClaw currently emits 27 ATOF events and a 16-step ATIF
trajectory. The checked-in run denies external web tools, so the trace stays
focused on repository-local work. The ATOF has the full 5 tool calls; the ATIF is
valid but currently groups those tool results into one standalone observation
step, which is the next OpenClaw-specific integration issue to inspect.

The OpenClaw native session log is useful supporting context, but the Relay
ATOF/ATIF pair is the unified artifact shape to carry into visualization,
validation, and cross-harness evaluation.
