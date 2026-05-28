# OpenClaw Relay Runbook

Commands for regenerating the OpenClaw `django__django-13741` artifacts. The
overview stays in `README.md`; this file keeps the shell steps out of the way.

## Set Up Gym

Start from a Gym checkout with Python 3.12+, `uv`, and Node/npm available.

```bash
export GYM_SOURCE_DIR=/path/to/Gym
cd "${GYM_SOURCE_DIR}"

uv venv --python 3.12
source .venv/bin/activate
uv sync

.venv/bin/ng_run +help=true
.venv/bin/ng_collect_rollouts +help=true

node --version
npm --version
npx --version
apptainer --version
```

## Set Up Relay Plugin

OpenClaw uses the NeMoRelay OpenClaw plugin, not the Python package.

```bash
export NEMO_RELAY_OPENCLAW_PLUGIN_PATH=/path/to/NeMo-Relay/integrations/openclaw
test -f "${NEMO_RELAY_OPENCLAW_PLUGIN_PATH}/package.json"
```

If the local checkout has not been built yet:

```bash
cd /path/to/NeMo-Relay
npm ci --ignore-scripts
npm run build --workspace=nemo-relay-openclaw
test -f integrations/openclaw/dist/index.js
```

## Configure Paths

```bash
export GYM_OUTPUT_DIR=/path/to/gym-output
export SWEBENCH_IMAGE_DIR="${GYM_OUTPUT_DIR}/images"
export NVIDIA_ENV_FILE=/path/to/nvidia.env
export NEMO_RELAY_OPENCLAW_PLUGIN_PATH=/path/to/NeMo-Relay/integrations/openclaw
```

Expected SWE-bench image:

```bash
ls "${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.django_1776_django-13741.sif"
```

If needed, download it:

```bash
mkdir -p "${SWEBENCH_IMAGE_DIR}" "${GYM_OUTPUT_DIR}/apptainer-cache" "${GYM_OUTPUT_DIR}/apptainer-tmp"

APPTAINER_CACHEDIR="${GYM_OUTPUT_DIR}/apptainer-cache" \
APPTAINER_TMPDIR="${GYM_OUTPUT_DIR}/apptainer-tmp" \
apptainer pull \
  "${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.django_1776_django-13741.sif" \
  docker://swebench/sweb.eval.x86_64.django_1776_django-13741:latest
```

## Baseline Gym/OpenClaw

Terminal 1:

```bash
cd "${GYM_SOURCE_DIR}"

set -a
source "${NVIDIA_ENV_FILE}"
set +a

.venv/bin/ng_run \
  '+config_paths=[responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml]' \
  +head_server.host=127.0.0.1 \
  +head_server.port=11011 \
  '+nvidia_api_key=${oc.env:NVIDIA_API_KEY}' \
  '+nvidia_base_url=${oc.env:NVIDIA_BASE_URL}' \
  +openai_api_key=dummy \
  +openai_base_url=null \
  +error_on_almost_servers=false \
  '+openclaw_agent.responses_api_agents.openclaw_agent.command=npx -y openclaw@2026.5.26' \
  "+openclaw_agent.responses_api_agents.openclaw_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+openclaw_agent.responses_api_agents.openclaw_agent.workspace_root=${GYM_OUTPUT_DIR}/openclaw-workspaces" \
  +openclaw_agent.responses_api_agents.openclaw_agent.verify_swebench=true \
  "+openclaw_agent.responses_api_agents.openclaw_agent.swebench_results_root=${GYM_OUTPUT_DIR}/openclaw-swebench-verifier" \
  '+openclaw_agent.responses_api_agents.openclaw_agent.openclaw_config.tools.deny=[web_search,web_fetch]'
```

Terminal 2:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_collect_rollouts \
  +head_server.host=127.0.0.1 \
  +head_server.port=11011 \
  +agent_name=openclaw_agent \
  +input_jsonl_fpath=responses_api_agents/openclaw_agent/data/django_13741_smoke.jsonl \
  "+output_jsonl_fpath=${GYM_OUTPUT_DIR}/openclaw_django_13741_rollout.jsonl" \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

Stop the server before starting the Relay run.

## Relay-Enabled Gym/OpenClaw

Terminal 1:

```bash
cd "${GYM_SOURCE_DIR}"

set -a
source "${NVIDIA_ENV_FILE}"
set +a

.venv/bin/ng_run \
  '+config_paths=[responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml]' \
  +head_server.host=127.0.0.1 \
  +head_server.port=11012 \
  '+nvidia_api_key=${oc.env:NVIDIA_API_KEY}' \
  '+nvidia_base_url=${oc.env:NVIDIA_BASE_URL}' \
  +openai_api_key=dummy \
  +openai_base_url=null \
  +error_on_almost_servers=false \
  '+openclaw_agent.responses_api_agents.openclaw_agent.command=npx -y openclaw@2026.5.26' \
  "+openclaw_agent.responses_api_agents.openclaw_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+openclaw_agent.responses_api_agents.openclaw_agent.workspace_root=${GYM_OUTPUT_DIR}/openclaw-relay-workspaces" \
  +openclaw_agent.responses_api_agents.openclaw_agent.verify_swebench=true \
  "+openclaw_agent.responses_api_agents.openclaw_agent.swebench_results_root=${GYM_OUTPUT_DIR}/openclaw-relay-swebench-verifier" \
  +openclaw_agent.responses_api_agents.openclaw_agent.nemo_flow.enabled=true \
  "+openclaw_agent.responses_api_agents.openclaw_agent.nemo_flow.plugin_local_path=${NEMO_RELAY_OPENCLAW_PLUGIN_PATH}" \
  '+openclaw_agent.responses_api_agents.openclaw_agent.openclaw_config.tools.deny=[web_search,web_fetch]'
```

Terminal 2:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_collect_rollouts \
  +head_server.host=127.0.0.1 \
  +head_server.port=11012 \
  +agent_name=openclaw_agent \
  +input_jsonl_fpath=responses_api_agents/openclaw_agent/data/django_13741_smoke.jsonl \
  "+output_jsonl_fpath=${GYM_OUTPUT_DIR}/openclaw_django_13741_relay_rollout.jsonl" \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

## Refresh Checked-In Artifacts

```bash
cd "${GYM_SOURCE_DIR}"

python3 docs/environment-tutorials/agent-harness-relay/openclaw/regenerate.py \
  --tmp-root "${GYM_OUTPUT_DIR}" \
  --output-dir "${GYM_SOURCE_DIR}/docs/environment-tutorials/agent-harness-relay/openclaw/artifacts"
```
