# Hermes django__django-13741 ATIF Comparison

This directory compares artifacts for the same SWE-bench Verified task with and without NeMoRelay.

## Files

- `artifacts/without-relay.gym-reconstructed.atif.json`: post-hoc ATIF reconstructed from Gym rollout output.
- `artifacts/with-relay.nemo-relay.atif.json`: ATIF emitted from adapter-level NeMoRelay capture around Hermes callbacks.
- `artifacts/with-relay.nemo-relay.atof.jsonl`: raw ATOF events behind the Relay ATIF.
- `artifacts/summary.json`: compact comparison metadata.
- `regenerate.py`: refreshes the files in `artifacts/` from Gym rollout outputs.

## Current Snapshot

The current regenerated snapshot uses the same `django__django-13741` task with
Hermes constrained to local SWE-style tools: `terminal`, `file`, and
`code_execution`. This avoids browser/web-search behavior and produces a real
workspace patch.

Baseline Gym/Hermes:

- reward `1.0`; SWE-bench resolved
- 29 reconstructed ATIF steps
- 10 assistant message items
- 9 function calls
- 9 function-call outputs

Relay-enabled Gym/Hermes:

- reward `1.0`; SWE-bench resolved
- 35 ATOF events
- 27 ATIF steps
- ATIF source split: 7 agent, 7 user, 13 system
- 7 turns used by the Hermes agent
- tool calls captured in Relay ATIF: `search_files`, `read_file`, `patch`

Hermes is useful for comparison because the baseline Gym response already has structured tool call/output items, while Relay adds raw ATOF plus normalized ATIF. The current Hermes ATIF is noisier than OpenClaw because the adapter projects Hermes callbacks and assistant messages rather than consuming a native harness session log.

## Set Up Gym Environment

Start from a Gym checkout with Python 3.12+ and `uv` available.

```bash
export GYM_SOURCE_DIR=/path/to/Gym
cd "${GYM_SOURCE_DIR}"

uv venv --python 3.12
source .venv/bin/activate
uv sync

.venv/bin/ng_run +help=true
.venv/bin/ng_collect_rollouts +help=true
```

Install Apptainer separately if it is not already available:

```bash
apptainer --version
```

## Set Up NeMoRelay

Hermes uses the NeMoRelay Python package only when Relay capture is enabled. For
this POC, install a local NeMoRelay checkout into the Gym virtual environment so
Gym does not need a hard dependency:

```bash
export NEMO_RELAY_PYTHON_PATH=/path/to/NeMo-Relay/python

cd "${GYM_SOURCE_DIR}"
source .venv/bin/activate
uv pip install -e "$(dirname "${NEMO_RELAY_PYTHON_PATH}")"

python -c "import nemo_relay; print(nemo_relay.__file__)"
```

The Relay-enabled command also passes this path as `nemo_relay.python_path` for
local source checkout runs.

If you want to use the published Python package instead, install it into the
Gym virtual environment and omit the `nemo_relay.python_path` override:

```bash
cd "${GYM_SOURCE_DIR}"
source .venv/bin/activate

uv pip install nemo-relay
python -c "import nemo_relay; print(nemo_relay.__file__)"
```

## Configure Paths

The commands need a Gym checkout, an output directory, model credentials, the SWE-bench image directory, and the NeMoRelay Python package path for the Relay run.

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

## Regenerate This Directory

After rerunning the baseline and Relay commands, refresh the checked-in comparison artifacts:

```bash
cd "${GYM_SOURCE_DIR}"

python3 docs/environment-tutorials/agent-harness-relay/hermes/regenerate.py \
  --tmp-root "${GYM_OUTPUT_DIR}" \
  --output-dir "${GYM_SOURCE_DIR}/docs/environment-tutorials/agent-harness-relay/hermes/artifacts"
```

The script rewrites:

- `artifacts/without-relay.gym-reconstructed.atif.json`
- `artifacts/with-relay.nemo-relay.atif.json`
- `artifacts/with-relay.nemo-relay.atof.jsonl`
- `artifacts/summary.json`
