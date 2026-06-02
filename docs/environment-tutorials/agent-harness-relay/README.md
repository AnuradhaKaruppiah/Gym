# NeMoRelay Artifacts For Agent Harnesses

This POC shows how Gym can optionally enable a NeMoFlow/NeMoRelay plugin inside
an agent harness and get richer, unified trajectory artifacts without making the
harness output format the integration contract.

The Relay artifacts are the primary files:

- `with-relay.nemo-relay.atof.jsonl`: lossless ATOF event stream from the run.
- `with-relay.nemo-relay.atif.json`: ATIF trajectory projected from Relay events.

Baseline files are included as secondary context for debugging and migration:

- `without-relay.gym-reconstructed.atif.json`: best-effort ATIF reconstructed
  from Gym rollout output.
- Harness-native logs, when available, such as OpenClaw session JSONL.

## POC Shape

All harnesses run the same SWE-bench Verified task, `django__django-13741`, and
leave a resolved workspace patch. The goal is to show that Relay can provide a
consistent observability layer across them.

| Harness | Relay integration | Primary artifact bundle |
|---|---|---|
| Hermes | Python adapter-level NeMoRelay capture around Hermes callbacks | [hermes/](hermes/) |
| OpenClaw | Optional NeMoFlow plugin loaded by the OpenClaw harness | [openclaw/](openclaw/) |
| OpenCode | Optional NeMoFlow plugin inside the OpenCode runtime | [opencode/](opencode/) |
| Codex | Native NeMo Relay Codex hook wrapper launched from the Gym wrapper | [codex/](codex/) |
| Claude Code | Optional NeMoFlow wrapper around the Claude Code CLI in the Gym wrapper | [claude-code/](claude-code/) |

ATOF is the source trace to use when losslessness matters. ATIF is the normalized
trajectory to feed into viewers, validators, and eval tooling. The checked-in
ATIF files were projected from Relay events after the current correlation fixes.
To inspect ATIF visually, see [Viewing ATIF In Phoenix](phoenix.atif.md).

## Discussion-Ready Outcome

The POC now covers five Gym harnesses with a consistent artifact contract:
enable Relay at the harness boundary, keep ATOF/ATIF as local durable files, and
treat backend uploads as optional views. Codex also exercises an optional
OpenInference export to Phoenix from the same Relay capture path, which shows
that the file-first contract can support multiple downstream visualizations
without making a live telemetry server mandatory.

Codex now uses Relay's native hook path and captures model, Bash, and
`apply_patch` activity in both ATOF and ATIF. The checked-in Codex rollout has
SWE-bench verification disabled, matching the other artifact-focused captures.

## Layout

Each harness directory keeps the readable overview in `README.md`, the run
commands in `RUNBOOK.md`, and generated payloads in `artifacts/`.

`hermes-offrails/` is a preserved off-rails trace that demonstrates why richer
trajectory capture is useful, but it is not one of the main POC harnesses.
