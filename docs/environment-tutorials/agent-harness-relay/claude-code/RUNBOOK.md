# Claude Code Relay Runbook

Commands and notes for regenerating the Claude Code `django__django-13741`
artifacts. The overview stays in `README.md`; this file keeps operational detail
out of the primary page.

## Set Up Gym

Start from a Gym checkout with Python 3.12+, `uv`, Claude Code, and Apptainer
available.

```bash
export GYM_SOURCE_DIR=/path/to/Gym
cd "${GYM_SOURCE_DIR}"

uv venv --python 3.12
source .venv/bin/activate
uv sync

.venv/bin/ng_run +help=true
.venv/bin/ng_collect_rollouts +help=true

claude --version
apptainer --version
```

## Set Up Relay

This POC uses a local NeMo Relay checkout. Rebuild the native binary after
switching Relay branches.

```bash
export NEMO_RELAY_SOURCE_DIR=/path/to/NeMo-Relay
cd "${NEMO_RELAY_SOURCE_DIR}"
cargo build
```

## Configure Paths

```bash
export GYM_OUTPUT_DIR=/path/to/gym-output
export SWEBENCH_IMAGE_DIR="${GYM_OUTPUT_DIR}/images"
export CLAUDE_CODE_RELAY_OUTPUT_DIR="${GYM_OUTPUT_DIR}/claude-code-nemo-relay"
export CLAUDE_CODE_WORKSPACE_ROOT="${GYM_OUTPUT_DIR}/claude-code-relay-workspaces"
export CLAUDE_CODE_SWEBENCH_RESULTS_ROOT="${GYM_OUTPUT_DIR}/claude-code-swebench-verifier"

mkdir -p \
  "${CLAUDE_CODE_RELAY_OUTPUT_DIR}" \
  "${CLAUDE_CODE_WORKSPACE_ROOT}" \
  "${CLAUDE_CODE_SWEBENCH_RESULTS_ROOT}"
```

Expected SWE-bench image:

```bash
ls "${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.django_1776_django-13741.sif"
```

## Relay-Enabled Gym/Claude Code

Terminal 1:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_run \
  '+config_paths=[responses_api_agents/claude_code_agent/configs/claude_code_agent.yaml]' \
  +head_server.host=127.0.0.1 \
  +head_server.port=11097 \
  '+anthropic_api_key=${oc.env:ANTHROPIC_API_KEY}' \
  +anthropic_base_url=null \
  +openai_api_key=dummy \
  +openai_base_url=null \
  +error_on_almost_servers=false \
  '+claude_code_agent.responses_api_agents.claude_code_agent.command=claude' \
  '+claude_code_agent.responses_api_agents.claude_code_agent.model=claude-sonnet-4-6' \
  "+claude_code_agent.responses_api_agents.claude_code_agent.container_formatter=${SWEBENCH_IMAGE_DIR}/swebench_sweb.eval.x86_64.\{instance_id\}.sif" \
  "+claude_code_agent.responses_api_agents.claude_code_agent.workspace_root=${CLAUDE_CODE_WORKSPACE_ROOT}" \
  +claude_code_agent.responses_api_agents.claude_code_agent.verify_swebench=true \
  "+claude_code_agent.responses_api_agents.claude_code_agent.swebench_results_root=${CLAUDE_CODE_SWEBENCH_RESULTS_ROOT}" \
  +claude_code_agent.responses_api_agents.claude_code_agent.nemo_flow.enabled=true \
  "+claude_code_agent.responses_api_agents.claude_code_agent.nemo_flow.command=${NEMO_RELAY_SOURCE_DIR}/target/debug/nemo-relay" \
  "+claude_code_agent.responses_api_agents.claude_code_agent.nemo_flow.output_dir=${CLAUDE_CODE_RELAY_OUTPUT_DIR}"
```

Terminal 2:

```bash
cd "${GYM_SOURCE_DIR}"

.venv/bin/ng_collect_rollouts \
  +head_server.host=127.0.0.1 \
  +head_server.port=11097 \
  +agent_name=claude_code_agent \
  +input_jsonl_fpath=responses_api_agents/claude_code_agent/data/django_13741_smoke.jsonl \
  "+output_jsonl_fpath=${GYM_OUTPUT_DIR}/claude_code_django_13741_relay_verified_rollout.jsonl" \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

## Artifact Copy

Copy the generated Relay files into this tutorial bundle:

```bash
cp "${CLAUDE_CODE_RELAY_OUTPUT_DIR}"/django__django-13741_*/claude-code.atof.jsonl \
  docs/environment-tutorials/agent-harness-relay/claude-code/artifacts/with-relay.nemo-relay.atof.jsonl

cp "${CLAUDE_CODE_RELAY_OUTPUT_DIR}"/django__django-13741_*/claude-code-*.atif.json \
  docs/environment-tutorials/agent-harness-relay/claude-code/artifacts/with-relay.nemo-relay.atif.json

cp "${GYM_OUTPUT_DIR}/claude_code_django_13741_relay_verified_rollout.jsonl" \
  docs/environment-tutorials/agent-harness-relay/claude-code/artifacts/with-relay.gym-rollout.jsonl
```
