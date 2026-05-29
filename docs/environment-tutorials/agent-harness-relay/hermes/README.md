# Hermes Relay Artifacts

Hermes runs `django__django-13741` with optional Python NeMoRelay capture around
the Gym/Hermes callbacks. The Relay artifacts are the primary output: ATOF keeps
the lower-level event stream, and ATIF gives the normalized trajectory for
viewers and validators.

## Primary Files

- `artifacts/with-relay.nemo-relay.atof.jsonl`: lossless Relay event stream.
- `artifacts/with-relay.nemo-relay.atif.json`: Relay-projected ATIF trajectory.
- `artifacts/summary.json`: compact run metadata and counts.

## Secondary Context

- `artifacts/without-relay.gym-reconstructed.atif.json`: ATIF reconstructed
  after the fact from Gym response output.
- [regenerate.py](regenerate.py): refreshes the checked-in artifacts from
  rollout outputs.
- [RUNBOOK.md](RUNBOOK.md): setup and rerun commands.

## Snapshot

Both Hermes runs resolve the SWE-bench task with `reward=1.0`.

Relay-enabled Hermes currently emits 110 ATOF events and a 66-step ATIF
trajectory. The ATIF has 21 same-step correlated tool observations and no
standalone tool observation steps.

The reconstructed baseline remains useful when debugging Gym response output,
but the Relay ATOF/ATIF pair is the artifact shape to use for unified trajectory
analysis across harnesses.
