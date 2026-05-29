# Codex Relay Runbook

Commands and notes for regenerating the Codex `django__django-13741` artifacts.
The overview stays in `README.md`; this file keeps operational detail out of the
primary page.

## Set Up Gym

Start from a Gym checkout with Python 3.12+, `uv`, Codex CLI, and Apptainer
available. The stock Relay Codex wrapper requires `codex-cli >= 0.129.0`; this
Gym direct-exporter path only requires a Codex CLI with `exec --json`.

```bash
export GYM_SOURCE_DIR=/path/to/Gym
cd "${GYM_SOURCE_DIR}"

uv venv --python 3.12
source .venv/bin/activate
uv sync

.venv/bin/ng_run +help=true
.venv/bin/ng_collect_rollouts +help=true

codex --version
apptainer --version
```

## Set Up Relay

This POC uses the local Python `nemo_relay` package directly from a NeMo Relay
checkout. Rebuild the native Python extension after switching Relay branches.

```bash
export NEMO_RELAY_SOURCE_DIR=/path/to/NeMo-Relay
cd "${NEMO_RELAY_SOURCE_DIR}"
.venv/bin/maturin develop --skip-install
```

The stock Relay Codex path is `nemo-relay run -- codex`, which enables Codex
hooks and gateway routing. The Gym Codex path below is intentionally direct:
Gym launches `codex exec --json`, registers Python Relay exporters, and writes
ATOF/ATIF under the configured output directory without requiring Codex hooks.

## Configure Paths

```bash
export GYM_OUTPUT_DIR=/path/to/gym-output
export SWEBENCH_IMAGE_DIR="${GYM_OUTPUT_DIR}/images"
export CODEX_RELAY_OUTPUT_DIR="${GYM_OUTPUT_DIR}/codex-nemo-relay"
export CODEX_WORKSPACE_ROOT="${GYM_OUTPUT_DIR}/codex-relay-workspaces"

mkdir -p "${CODEX_RELAY_OUTPUT_DIR}" "${CODEX_WORKSPACE_ROOT}"
```

Expected SWE-bench image:

```bash
ls "${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.django_1776_django-13741.sif"
```

## Relay-Enabled Gym/Codex

Terminal 1:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_run \
  '+config_paths=[responses_api_agents/codex_agent/configs/codex_agent.yaml]' \
  +head_server.host=127.0.0.1 \
  +head_server.port=11075 \
  +openai_api_key=dummy \
  +openai_base_url=null \
  +error_on_almost_servers=false \
  "+codex_agent.responses_api_agents.codex_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+codex_agent.responses_api_agents.codex_agent.workspace_root=${CODEX_WORKSPACE_ROOT}" \
  +codex_agent.responses_api_agents.codex_agent.verify_swebench=false \
  +codex_agent.responses_api_agents.codex_agent.nemo_relay.enabled=true \
  "+codex_agent.responses_api_agents.codex_agent.nemo_relay.python_path=${NEMO_RELAY_SOURCE_DIR}/python" \
  "+codex_agent.responses_api_agents.codex_agent.nemo_relay.output_dir=${CODEX_RELAY_OUTPUT_DIR}"
```

Optional Phoenix/OpenInference view:

```bash
  +codex_agent.responses_api_agents.codex_agent.nemo_relay.openinference.enabled=true \
  +codex_agent.responses_api_agents.codex_agent.nemo_relay.openinference.endpoint=http://127.0.0.1:6006/v1/traces \
  +codex_agent.responses_api_agents.codex_agent.nemo_relay.openinference.project_name=oi-gym-codex-relay
```

Terminal 2:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_collect_rollouts \
  +head_server.host=127.0.0.1 \
  +head_server.port=11075 \
  +agent_name=codex_agent \
  +input_jsonl_fpath=responses_api_agents/codex_agent/data/django_13741_smoke.jsonl \
  "+output_jsonl_fpath=${GYM_OUTPUT_DIR}/codex_django_13741_relay_rollout.jsonl" \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

## Artifact Copy

Copy the generated Relay files into this tutorial bundle:

```bash
cp "${CODEX_RELAY_OUTPUT_DIR}"/django__django-13741_*/codex.atof.jsonl \
  docs/environment-tutorials/agent-harness-relay/codex/artifacts/with-relay.nemo-relay.atof.jsonl

cp "${CODEX_RELAY_OUTPUT_DIR}"/django__django-13741_*/codex-*.atif.json \
  docs/environment-tutorials/agent-harness-relay/codex/artifacts/with-relay.nemo-relay.atif.json

cp "${GYM_OUTPUT_DIR}/codex_django_13741_relay_rollout.jsonl" \
  docs/environment-tutorials/agent-harness-relay/codex/artifacts/with-relay.gym-rollout.jsonl
```
