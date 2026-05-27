# OpenCode Agent

Runs the standalone `opencode run --format=json` CLI as a NeMo Gym
`responses_api_agent`.

This adapter is intentionally eval-first. It wraps the external OpenCode CLI at
Gym's `/run` and `/v1/responses` layer, parses OpenCode JSONL events into Gym
Responses output items, and optionally forwards the result to a Gym resources
server for verification.

## Configuration

```yaml
opencode_agent:
  responses_api_agents:
    opencode_agent:
      entrypoint: app.py
      resources_server: null
      model_server: null
      concurrency: 8
      command: opencode
      model: nvidia/opus-frontier
      openai_api_key: ${openai_api_key}
      openai_base_url: ${openai_base_url}
      work_dir: null
      workspace_root: outputs/opencode_agent/workspaces
      container_formatter: null
      apptainer_command: apptainer
      setup_timeout: 900
      timeout: 900
      thinking: true
      system_prompt: null
      extra_args: []
      nemo_relay:
        enabled: false
        plugin_package: nemo-flow-opencode
        server_module_path: null
        wrapper_filename: nemo-relay-opencode-plugin.mjs
        output_dir: null
        log_filename: opencode-plugin.log
        atof_filename: opencode.atof.jsonl
        atif_filename_template: "opencode-{session_id}.atif.json"
        agent_name: opencode
        mode: overwrite
      verify_swebench: false
      swebench_setup_dir: null
      swebench_results_root: outputs/opencode_agent/swebench-verifier
      swebench_verifier_timeout: 1200
      swebench_model_name: opencode_agent
      opencode_config:
        provider:
          nvidia:
            npm: "@ai-sdk/openai-compatible"
            name: NVIDIA
            options:
              baseURL: "{env:NVIDIA_BASE_URL}"
              apiKey: "{env:NVIDIA_API_KEY}"
            models:
              opus-frontier:
                id: aws/anthropic/claude-opus-4-5
                name: Claude 4.5 Opus
                reasoning: true
                options:
                  reasoning_effort: high
      datasets:
        - name: django_13741_smoke
          type: example
          jsonl_fpath: responses_api_agents/opencode_agent/data/django_13741_smoke.jsonl
          license: Apache 2.0
```

## POC Notes

- `resources_server: null` is useful while proving the CLI wrapper loads and
  can execute prompts.
- `command` can be a plain executable (`opencode`) or a command prefix such as
  `npx -y opencode-ai` for local smoke tests without a global install.
- `container_formatter` optionally points at SWE-bench SIF images. When set,
  the adapter copies `/testbed` out of the image, checks out the task base
  commit, runs OpenCode in that host workspace, and captures `git diff`.
- `opencode_config` is written to an isolated per-run OpenCode config home.
  This is required for provider aliases such as `nvidia/opus-frontier`.
- `nemo_relay.enabled=true` injects NeMo Relay into the isolated OpenCode
  config. For local PR testing, set `nemo_relay.server_module_path` to the
  public plugin module, for example
  `/path/to/nemo-relay/integrations/opencode/server.js`; the adapter writes a
  per-run wrapper plugin so OpenCode can load it by `file://` URL and receive
  the configured ATOF/ATIF output paths. Without `server_module_path`, the
  adapter falls back to the package name in `plugin_package`.
- For benchmark scoring, set `resources_server` to a verifier server or extend
  this agent with benchmark-specific verification.
- For SWE-bench scoring, set `verify_swebench=true` and provide
  `container_formatter`. The adapter writes a SWE-bench prediction JSONL from
  the OpenCode patch, runs `swebench.harness.run_local_evaluation` in the task
  SIF, and returns `reward=1.0` only when the report marks the instance
  resolved.
- `data/django_13741_smoke.jsonl` is a one-row Django SWE-bench Verified task
  adapted from the Harbor OpenCode smoke test.
- The verifier stores per-run artifacts under `swebench_results_root`, including
  the prediction JSONL, patch diff, copied SWE-bench report, and test logs.

## Optional NeMoRelay Capture Without Hard Dependency

NeMoRelay capture is opt-in. The OpenCode agent does not import NeMoRelay from
Python or require the plugin files when `nemo_relay.enabled=false`.

When enabled, the agent writes a per-run OpenCode plugin wrapper into the
isolated OpenCode config directory. That wrapper points OpenCode at the
configured NeMoRelay plugin module and passes the run-specific ATOF/ATIF output
paths. The Gym response metadata then includes the discovered Relay artifacts:

- `nemo_relay_output_dir`
- `nemo_relay_atof_path`
- `nemo_relay_atif_paths`
- `nemo_relay_log_path`

Minimal local override:

```bash
+opencode_agent.responses_api_agents.opencode_agent.nemo_relay.enabled=true \
+opencode_agent.responses_api_agents.opencode_agent.nemo_relay.server_module_path=/path/to/nemo-relay/integrations/opencode/server.js \
+opencode_agent.responses_api_agents.opencode_agent.nemo_relay.output_dir=/tmp/opencode-nemo-relay
```
