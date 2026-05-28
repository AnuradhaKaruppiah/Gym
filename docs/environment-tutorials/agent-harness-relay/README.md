# Agent Harness Relay Artifacts

This directory holds one-task SWE-bench artifacts for comparing Gym rollout output with optional NeMoRelay capture.

## Quick Orientation

This POC asks a narrow question: what extra trajectory data do we get when an agent harness runs under Gym with NeMoRelay capture enabled? The main value is that different agent harnesses can emit a shared telemetry shape, making trajectories easier to compare, validate, visualize, and feed into downstream eval tooling.

| Concept | What it means here | Deeper reference |
|---|---|---|
| ATIF | Agent Trajectory Interchange Format. This is the normalized trajectory JSON used for downstream viewers, validators, and eval tooling. | [Harbor trajectory schema models](https://github.com/harbor-framework/harbor/tree/main/src/harbor/models/trajectories) |
| ATOF | Agent Trajectory Observability Format. This is the lossless lower-level event stream emitted during execution; it can be used for replay and projected into ATIF. | [ATOF event format](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/packages/nvidia_nat_atif/atof-event-format.md) |
| NeMoRelay | Optional runtime/observability layer used here to capture agent-harness events without making Gym depend on Relay directly. | [GitHub repo](https://github.com/NVIDIA/NeMo-Relay), [Fern docs](https://nvidia-nemo-relay.docs.buildwithfern.com/nemo/relay/observability-plugin/about) |

In this directory, each harness has a baseline artifact reconstructed from Gym output and, where available, Relay-generated ATOF/ATIF artifacts from the same `django__django-13741` SWE-bench task.

To inspect an ATIF trajectory visually, see [Viewing ATIF In Phoenix](phoenix.atif.md).

## Harness Bundles

- `hermes/`: runbook, regeneration script, and artifacts for Hermes on `django__django-13741`.
- `hermes-offrails/`: preserved unconstrained Hermes trace showing why trajectory evaluation matters.
- `openclaw/`: runbook, regeneration script, and artifacts for OpenClaw on `django__django-13741`.
- `opencode/`: captured artifacts for OpenCode on `django__django-13741`.

Each harness stores generated JSON/JSONL payloads under its `artifacts/` directory so the docs remain readable while the data stays easy to copy into Phoenix or other ATIF consumers.
