# OpenClaw django__django-13741 ATIF Comparison

This directory compares artifacts for the same SWE-bench Verified task with and without NeMoRelay.

## Files

- `artifacts/without-relay.gym-reconstructed.atif.json`: post-hoc ATIF reconstructed from Gym rollout output.
- `artifacts/without-relay.openclaw.session.jsonl`: native OpenClaw session log from the baseline run.
- `artifacts/with-relay.nemo-relay.atof.jsonl`: raw ATOF event stream emitted by NeMoRelay.
- `artifacts/with-relay.nemo-relay.atif.json`: ATIF emitted from NeMoRelay capture.
- `artifacts/with-relay.openclaw.session.jsonl`: native OpenClaw session log from the Relay-enabled run.
- `artifacts/summary.json`: compact comparison metadata.
- `regenerate.py`: refreshes the files in `artifacts/` from Gym rollout outputs.

## Current Snapshot

Both runs resolved `django__django-13741` with `reward=1.0`.

Baseline Gym/OpenClaw:

- 3 reconstructed ATIF steps
- 14 native OpenClaw session events
- Native event types: custom, message, model_change, session, thinking_level_change
- 1 turn used by the OpenClaw agent

Relay-enabled Gym/OpenClaw:

- 23 raw ATOF events
- ATOF category split: 2 agent, 10 llm, 3 mark, 8 tool
- 16 ATIF steps
- ATIF source split: 5 agent, 5 user, 6 system
- 14 native OpenClaw session events
- 1 turn used by the OpenClaw agent

OpenClaw is a useful comparison point because the baseline already emits a compact native session JSONL, while Relay turns the enabled run into a normalized ATIF trajectory that can be compared across harnesses.
The checked-in Relay run denies external web tools (`web_search` and `web_fetch`), so the trace stays focused on repository-local tool use. The Relay ATIF is the current projector output and still demonstrates the RELAY-169 semantic projection issue, so the ATOF is included as the lossless source trace for comparison.

## Set Up Gym Environment

Start from a Gym checkout with Python 3.12+, `uv`, and Node/npm available. Node is needed because the OpenClaw run below launches OpenClaw.

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
```

Install Apptainer separately if it is not already available:

```bash
apptainer --version
```

## Set Up NeMoRelay

OpenClaw uses the NeMoRelay OpenClaw plugin, not the Python package. For this
POC, point at the plugin from a local NeMoRelay checkout:

```bash
export NEMO_RELAY_OPENCLAW_PLUGIN_PATH=/path/to/NeMo-Relay/integrations/openclaw
test -f "${NEMO_RELAY_OPENCLAW_PLUGIN_PATH}/package.json"
```

If the local checkout has not been built yet, build the plugin once from the
NeMoRelay repository root:

```bash
npm ci --ignore-scripts
npm run build --workspace=nemo-relay-openclaw
test -f integrations/openclaw/dist/index.js
```

The Relay-enabled command passes the plugin path as
`nemo_flow.plugin_local_path`; the OpenClaw adapter adds it to OpenClaw's plugin
load paths and enables ATOF and ATIF output under the run artifact directory.

The adapter also has a `nemo_flow.plugin_package` setting for package-based
installs (`npm:nemo-relay-openclaw@0.3.0`), but this POC uses the local plugin
path so PR changes can be tested before a package publish.

## Configure Paths

The commands need a Gym checkout, an output directory, model credentials, the SWE-bench image directory, and the NeMoRelay OpenClaw plugin path for the Relay run.

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

## Regenerate This Directory

After rerunning the baseline and Relay commands, refresh the checked-in comparison artifacts:

```bash
cd "${GYM_SOURCE_DIR}"

python3 docs/environment-tutorials/agent-harness-relay/openclaw/regenerate.py \
  --tmp-root "${GYM_OUTPUT_DIR}" \
  --output-dir "${GYM_SOURCE_DIR}/docs/environment-tutorials/agent-harness-relay/openclaw/artifacts"
```

The script rewrites:

- `artifacts/without-relay.gym-reconstructed.atif.json`
- `artifacts/without-relay.openclaw.session.jsonl`
- `artifacts/with-relay.nemo-relay.atof.jsonl`
- `artifacts/with-relay.nemo-relay.atif.json`
- `artifacts/with-relay.openclaw.session.jsonl`
- `artifacts/summary.json`
