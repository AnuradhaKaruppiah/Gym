# OpenCode Relay Runbook

Commands and notes for regenerating the OpenCode `django__django-13741`
artifacts. The overview stays in `README.md`; this file keeps operational detail
out of the primary page.

## Set Up Gym

Start from a Gym checkout with Python 3.12+, `uv`, Bun, and Apptainer available.

```bash
export GYM_SOURCE_DIR=/path/to/Gym
cd "${GYM_SOURCE_DIR}"

uv venv --python 3.12
source .venv/bin/activate
uv sync

.venv/bin/ng_run +help=true
.venv/bin/ng_collect_rollouts +help=true

bun --version
apptainer --version
```

## Set Up Relay/OpenCode

This POC uses a local NeMoRelay checkout with the OpenCode NeMoFlow integration.
Refresh the local Bun install after switching Relay branches so OpenCode sees the
current native node binding.

```bash
export NEMO_RELAY_SOURCE_DIR=/path/to/NeMo-Relay
cd "${NEMO_RELAY_SOURCE_DIR}/third_party/opencode"
bun install --force
```

The OpenCode command below runs from the local Relay checkout:

```bash
export OPENCODE_COMMAND="bun run --cwd ${NEMO_RELAY_SOURCE_DIR}/third_party/opencode/packages/opencode --conditions=browser ./src/index.ts"
```

## Configure Paths

```bash
export GYM_OUTPUT_DIR=/path/to/gym-output
export SWEBENCH_IMAGE_DIR="${GYM_OUTPUT_DIR}/images"
export NVIDIA_ENV_FILE=/path/to/nvidia.env
export NEMO_FLOW_ATOF_DIR="${GYM_OUTPUT_DIR}/opencode-nemo-flow/atof"
export NEMO_FLOW_ATIF_DIR="${GYM_OUTPUT_DIR}/opencode-nemo-flow/atif"

mkdir -p "${NEMO_FLOW_ATOF_DIR}" "${NEMO_FLOW_ATIF_DIR}"
```

Expected SWE-bench image:

```bash
ls "${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.django_1776_django-13741.sif"
```

## Relay-Enabled Gym/OpenCode

Terminal 1:

```bash
cd "${GYM_SOURCE_DIR}"

set -a
source "${NVIDIA_ENV_FILE}"
set +a

export NEMO_FLOW_ENABLED=1
export NEMO_FLOW_ATOF_DIR
export NEMO_FLOW_ATIF_DIR

.venv/bin/ng_run \
  '+config_paths=[responses_api_agents/opencode_agent/configs/opencode_agent.yaml]' \
  +head_server.host=127.0.0.1 \
  +head_server.port=11064 \
  +openai_api_key=dummy \
  +openai_base_url=null \
  +error_on_almost_servers=false \
  "+opencode_agent.responses_api_agents.opencode_agent.command=${OPENCODE_COMMAND}" \
  "+opencode_agent.responses_api_agents.opencode_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+opencode_agent.responses_api_agents.opencode_agent.workspace_root=${GYM_OUTPUT_DIR}/opencode-relay-workspaces" \
  +opencode_agent.responses_api_agents.opencode_agent.verify_swebench=true \
  "+opencode_agent.responses_api_agents.opencode_agent.swebench_results_root=${GYM_OUTPUT_DIR}/opencode-relay-swebench-verifier"
```

Terminal 2:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_collect_rollouts \
  +head_server.host=127.0.0.1 \
  +head_server.port=11064 \
  +agent_name=opencode_agent \
  +input_jsonl_fpath=responses_api_agents/opencode_agent/data/django_13741_smoke.jsonl \
  "+output_jsonl_fpath=${GYM_OUTPUT_DIR}/opencode_django_13741_relay_rollout.jsonl" \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

## Baseline Context

If you also need the secondary baseline artifact, rerun OpenCode without the
`NEMO_FLOW_*` environment variables and write to a separate rollout JSONL. Then
use the local artifact-generation script or notebook used for this POC to refresh
`artifacts/without-relay.gym-reconstructed.atif.json`.
