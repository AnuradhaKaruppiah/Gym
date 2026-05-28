# Hermes Relay Runbook

Commands for regenerating the Hermes `django__django-13741` artifacts. The
overview stays in `README.md`; this file keeps the shell steps out of the way.

## Set Up Gym

```bash
export GYM_SOURCE_DIR=/path/to/Gym
cd "${GYM_SOURCE_DIR}"

uv venv --python 3.12
source .venv/bin/activate
uv sync

.venv/bin/ng_run +help=true
.venv/bin/ng_collect_rollouts +help=true
apptainer --version
```

## Set Up Relay

Hermes uses the NeMoRelay Python package when capture is enabled.

```bash
export NEMO_RELAY_PYTHON_PATH=/path/to/NeMo-Relay/python

cd "${GYM_SOURCE_DIR}"
source .venv/bin/activate
uv pip install -e "$(dirname "${NEMO_RELAY_PYTHON_PATH}")"

python -c "import nemo_relay; print(nemo_relay.__file__)"
```

## Configure Paths

```bash
export GYM_OUTPUT_DIR=/path/to/gym-output
export SWEBENCH_IMAGE_DIR="${GYM_OUTPUT_DIR}/images"
export NVIDIA_ENV_FILE=/path/to/nvidia.env
export NEMO_RELAY_PYTHON_PATH=/path/to/NeMo-Relay/python
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

## Baseline Gym/Hermes

Terminal 1:

```bash
cd "${GYM_SOURCE_DIR}"

set -a
source "${NVIDIA_ENV_FILE}"
set +a

.venv/bin/ng_run \
  '+config_paths=[responses_api_agents/hermes_agent/configs/hermes_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml]' \
  +head_server.host=127.0.0.1 \
  +head_server.port=11031 \
  +hermes_agent.responses_api_agents.hermes_agent.resources_server=null \
  '+policy_base_url=${oc.env:NVIDIA_BASE_URL}' \
  '+policy_api_key=${oc.env:NVIDIA_API_KEY}' \
  +policy_model_name=nvidia/qwen/qwen-235b \
  +error_on_almost_servers=false \
  "+hermes_agent.responses_api_agents.hermes_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+hermes_agent.responses_api_agents.hermes_agent.workspace_root=${GYM_OUTPUT_DIR}/hermes-workspaces" \
  +hermes_agent.responses_api_agents.hermes_agent.verify_swebench=true \
  "+hermes_agent.responses_api_agents.hermes_agent.swebench_results_root=${GYM_OUTPUT_DIR}/hermes-swebench-verifier" \
  +hermes_agent.responses_api_agents.hermes_agent.temperature=0.2 \
  '+hermes_agent.responses_api_agents.hermes_agent.enabled_toolsets=[terminal,file,code_execution]'
```

Terminal 2:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_collect_rollouts \
  +head_server.host=127.0.0.1 \
  +head_server.port=11031 \
  +agent_name=hermes_agent \
  +input_jsonl_fpath=responses_api_agents/hermes_agent/data/django_13741_smoke.jsonl \
  "+output_jsonl_fpath=${GYM_OUTPUT_DIR}/hermes_django_13741_rollout.jsonl" \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

Stop the server before starting the Relay run.

## Relay-Enabled Gym/Hermes

Terminal 1:

```bash
cd "${GYM_SOURCE_DIR}"

set -a
source "${NVIDIA_ENV_FILE}"
set +a

.venv/bin/ng_run \
  '+config_paths=[responses_api_agents/hermes_agent/configs/hermes_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml]' \
  +head_server.host=127.0.0.1 \
  +head_server.port=11032 \
  +hermes_agent.responses_api_agents.hermes_agent.resources_server=null \
  '+policy_base_url=${oc.env:NVIDIA_BASE_URL}' \
  '+policy_api_key=${oc.env:NVIDIA_API_KEY}' \
  +policy_model_name=nvidia/qwen/qwen-235b \
  +error_on_almost_servers=false \
  "+hermes_agent.responses_api_agents.hermes_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+hermes_agent.responses_api_agents.hermes_agent.workspace_root=${GYM_OUTPUT_DIR}/hermes-relay-workspaces" \
  +hermes_agent.responses_api_agents.hermes_agent.verify_swebench=true \
  "+hermes_agent.responses_api_agents.hermes_agent.swebench_results_root=${GYM_OUTPUT_DIR}/hermes-relay-swebench-verifier" \
  +hermes_agent.responses_api_agents.hermes_agent.temperature=0.2 \
  '+hermes_agent.responses_api_agents.hermes_agent.enabled_toolsets=[terminal,file,code_execution]' \
  +hermes_agent.responses_api_agents.hermes_agent.nemo_relay.enabled=true \
  "+hermes_agent.responses_api_agents.hermes_agent.nemo_relay.python_path=${NEMO_RELAY_PYTHON_PATH}" \
  "+hermes_agent.responses_api_agents.hermes_agent.nemo_relay.output_dir=${GYM_OUTPUT_DIR}/hermes-nemo-relay"
```

Terminal 2:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_collect_rollouts \
  +head_server.host=127.0.0.1 \
  +head_server.port=11032 \
  +agent_name=hermes_agent \
  +input_jsonl_fpath=responses_api_agents/hermes_agent/data/django_13741_smoke.jsonl \
  "+output_jsonl_fpath=${GYM_OUTPUT_DIR}/hermes_django_13741_relay_rollout.jsonl" \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

## Refresh Checked-In Artifacts

```bash
cd "${GYM_SOURCE_DIR}"

python3 docs/environment-tutorials/agent-harness-relay/hermes/regenerate.py \
  --tmp-root "${GYM_OUTPUT_DIR}" \
  --output-dir "${GYM_SOURCE_DIR}/docs/environment-tutorials/agent-harness-relay/hermes/artifacts"
```
