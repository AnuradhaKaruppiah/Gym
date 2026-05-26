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
