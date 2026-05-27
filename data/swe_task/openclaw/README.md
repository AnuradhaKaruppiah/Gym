# OpenClaw django__django-13741 ATIF Comparison

This directory compares artifacts for the same SWE-bench Verified task with and without NeMoRelay.

## Files

- `without-relay.gym-reconstructed.atif.json`: post-hoc ATIF reconstructed from Gym rollout output.
- `without-relay.openclaw.session.jsonl`: native OpenClaw session log from the baseline run.
- `with-relay.nemo-relay.atif.json`: ATIF emitted from NeMoRelay/NeMoFlow capture.
- `with-relay.openclaw.session.jsonl`: native OpenClaw session log from the Relay-enabled run.
- `summary.json`: compact comparison metadata.
- `regenerate.py`: refreshes the files in this directory from Gym rollout artifacts.

## Current Snapshot

Both runs resolved `django__django-13741` with `reward=1.0`.

Baseline Gym/OpenClaw:

- 3 reconstructed ATIF steps
- 16 native OpenClaw session events
- Native event types: custom, message, model_change, session, thinking_level_change
- 1 turn used by the OpenClaw agent

Relay-enabled Gym/OpenClaw:

- 20 ATIF steps
- ATIF source split: 6 agent, 6 user, 8 system
- 16 native OpenClaw session events
- 1 turn used by the OpenClaw agent

OpenClaw is a useful comparison point because the baseline already emits a compact native session JSONL, while Relay turns the enabled run into a normalized ATIF trajectory that can be compared across harnesses.

## Configure Paths

The commands need a Gym checkout, an output directory, model credentials, the SWE-bench image directory, and the NeMoRelay OpenClaw plugin path for the Relay run.

```bash
export GYM_SOURCE_DIR=/path/to/Gym
export GYM_OUTPUT_DIR=/path/to/gym-output
export SWEBENCH_IMAGE_DIR="${GYM_OUTPUT_DIR}/images"
export NVIDIA_ENV_FILE=/path/to/nvidia.env
export NEMO_RELAY_OPENCLAW_PLUGIN_PATH=/path/to/NeMo-Relay/integrations/openclaw
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

## Run Baseline Gym/OpenClaw

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
  '+openclaw_agent.responses_api_agents.openclaw_agent.command=npx -y openclaw@2026.5.22' \
  "+openclaw_agent.responses_api_agents.openclaw_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+openclaw_agent.responses_api_agents.openclaw_agent.workspace_root=${GYM_OUTPUT_DIR}/openclaw-workspaces" \
  +openclaw_agent.responses_api_agents.openclaw_agent.verify_swebench=true \
  "+openclaw_agent.responses_api_agents.openclaw_agent.swebench_results_root=${GYM_OUTPUT_DIR}/openclaw-swebench-verifier"
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

Stop the server in terminal 1 before starting the Relay run.

## Run Relay-Enabled Gym/OpenClaw

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
  '+openclaw_agent.responses_api_agents.openclaw_agent.command=npx -y openclaw@2026.5.22' \
  "+openclaw_agent.responses_api_agents.openclaw_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+openclaw_agent.responses_api_agents.openclaw_agent.workspace_root=${GYM_OUTPUT_DIR}/openclaw-relay-workspaces" \
  +openclaw_agent.responses_api_agents.openclaw_agent.verify_swebench=true \
  "+openclaw_agent.responses_api_agents.openclaw_agent.swebench_results_root=${GYM_OUTPUT_DIR}/openclaw-relay-swebench-verifier" \
  +openclaw_agent.responses_api_agents.openclaw_agent.nemo_flow.enabled=true \
  "+openclaw_agent.responses_api_agents.openclaw_agent.nemo_flow.plugin_local_path=${NEMO_RELAY_OPENCLAW_PLUGIN_PATH}"
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

## Regenerate This Directory

After rerunning the baseline and Relay commands, refresh the checked-in comparison artifacts:

```bash
cd "${GYM_SOURCE_DIR}"

python3 data/swe_task/openclaw/regenerate.py \
  --tmp-root "${GYM_OUTPUT_DIR}" \
  --output-dir "${GYM_SOURCE_DIR}/data/swe_task/openclaw"
```

The script rewrites:

- `without-relay.gym-reconstructed.atif.json`
- `without-relay.openclaw.session.jsonl`
- `with-relay.nemo-relay.atif.json`
- `with-relay.openclaw.session.jsonl`
- `summary.json`
