# Hermes django__django-13741 ATIF Comparison

This directory compares artifacts for the same SWE-bench Verified task with and without NeMoRelay.

## Files

- `without-relay.gym-reconstructed.atif.json`: post-hoc ATIF reconstructed from Gym rollout output.
- `with-relay.nemo-relay.atif.json`: ATIF emitted from adapter-level NeMoRelay capture around Hermes callbacks.
- `with-relay.nemo-relay.atof.jsonl`: raw ATOF events behind the Relay ATIF.
- `summary.json`: compact comparison metadata.
- `regenerate.py`: refreshes the files in this directory from Gym rollout artifacts.

## Current Snapshot

Both runs resolved `django__django-13741` with `reward=1.0`.

Baseline Gym/Hermes:

- 29 reconstructed ATIF steps
- 10 assistant message items
- 9 function calls
- 9 function-call outputs

Relay-enabled Gym/Hermes:

- 64 ATOF events
- 50 ATIF steps
- ATIF source split: 13 agent, 13 user, 24 system
- 13 turns used by the Hermes agent

Hermes is a useful third comparison point: the baseline Gym response already has structured tool call/output items, while Relay adds raw ATOF plus normalized ATIF. The current Hermes ATIF is noisier than OpenClaw because the adapter projects Hermes callbacks and assistant messages rather than consuming a native harness session log.

## Configure Paths

The commands need a Gym checkout, an output directory, model credentials, the SWE-bench image directory, and the NeMoRelay Python package path for the Relay run.

```bash
export GYM_SOURCE_DIR=/path/to/Gym
export GYM_OUTPUT_DIR=/path/to/gym-output
export SWEBENCH_IMAGE_DIR="${GYM_OUTPUT_DIR}/images"
export NVIDIA_ENV_FILE=/path/to/nvidia.env
export NEMO_RELAY_PYTHON_PATH=/path/to/NeMo-Relay/python
```

Expected SWE-bench image:

```bash
ls "${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.django_1776_django-13741.sif"
```

If that file does not exist, download it with Apptainer:

```bash
mkdir -p "${SWEBENCH_IMAGE_DIR}" "${GYM_OUTPUT_DIR}/apptainer-cache" "${GYM_OUTPUT_DIR}/apptainer-tmp"

APPTAINER_CACHEDIR="${GYM_OUTPUT_DIR}/apptainer-cache" \
APPTAINER_TMPDIR="${GYM_OUTPUT_DIR}/apptainer-tmp" \
apptainer pull \
  "${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.django_1776_django-13741.sif" \
  docker://swebench/sweb.eval.x86_64.django_1776_django-13741:latest
```

If `NVIDIA_BASE_URL` and `NVIDIA_API_KEY` are already exported in your shell, you can skip the `source "${NVIDIA_ENV_FILE}"` lines below.

## Run Baseline Gym/Hermes

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
  '+policy_base_url=${oc.env:NVIDIA_BASE_URL}' \
  '+policy_api_key=${oc.env:NVIDIA_API_KEY}' \
  +policy_model_name=nvidia/qwen/qwen-235b \
  +error_on_almost_servers=false \
  "+hermes_agent.responses_api_agents.hermes_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+hermes_agent.responses_api_agents.hermes_agent.workspace_root=${GYM_OUTPUT_DIR}/hermes-workspaces" \
  +hermes_agent.responses_api_agents.hermes_agent.verify_swebench=true \
  "+hermes_agent.responses_api_agents.hermes_agent.swebench_results_root=${GYM_OUTPUT_DIR}/hermes-swebench-verifier" \
  +hermes_agent.responses_api_agents.hermes_agent.temperature=0.2
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

Stop the server in terminal 1 before starting the Relay run.

## Run Relay-Enabled Gym/Hermes

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
  '+policy_base_url=${oc.env:NVIDIA_BASE_URL}' \
  '+policy_api_key=${oc.env:NVIDIA_API_KEY}' \
  +policy_model_name=nvidia/qwen/qwen-235b \
  +error_on_almost_servers=false \
  "+hermes_agent.responses_api_agents.hermes_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+hermes_agent.responses_api_agents.hermes_agent.workspace_root=${GYM_OUTPUT_DIR}/hermes-relay-workspaces" \
  +hermes_agent.responses_api_agents.hermes_agent.verify_swebench=true \
  "+hermes_agent.responses_api_agents.hermes_agent.swebench_results_root=${GYM_OUTPUT_DIR}/hermes-relay-swebench-verifier" \
  +hermes_agent.responses_api_agents.hermes_agent.temperature=0.2 \
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

## Regenerate This Directory

After rerunning the baseline and Relay commands, refresh the checked-in comparison artifacts:

```bash
cd "${GYM_SOURCE_DIR}"

python3 data/swe_task/hermes/regenerate.py \
  --tmp-root "${GYM_OUTPUT_DIR}" \
  --output-dir "${GYM_SOURCE_DIR}/data/swe_task/hermes"
```

The script rewrites:

- `without-relay.gym-reconstructed.atif.json`
- `with-relay.nemo-relay.atif.json`
- `with-relay.nemo-relay.atof.jsonl`
- `summary.json`
