# OpenCode Relay Artifacts

OpenCode runs `django__django-13741` with an optional NeMoFlow plugin inside the
OpenCode runtime. The Relay artifacts are the primary output: ATOF keeps the raw
event stream, and ATIF gives a normalized trajectory independent of OpenCode's
native runtime details.

## Primary Files

- `artifacts/with-relay.nemo-relay.atof.jsonl`: lossless Relay event stream.
- `artifacts/with-relay.nemo-relay.atif.json`: Relay-projected ATIF trajectory.
- `artifacts/with-relay.nemo-relay.explore.atif.json`: Relay-projected ATIF
  trajectory for the OpenCode `explore` subagent.
- `artifacts/summary.json`: compact run metadata and counts.

## Secondary Context

- `artifacts/without-relay.gym-reconstructed.atif.json`: ATIF reconstructed
  after the fact from Gym rollout output.
- [RUNBOOK.md](RUNBOOK.md): setup and rerun notes.

## Snapshot

Both OpenCode runs resolve the SWE-bench task with `reward=1.0`.

The checked-in Relay bundle includes native OpenCode NeMoFlow capture, with ATOF
as the lossless source trace and ATIF as the portable trajectory projection.
The reconstructed baseline is retained only as supporting context.
