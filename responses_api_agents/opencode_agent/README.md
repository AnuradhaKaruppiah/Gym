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
      model: openai/gpt-4o-mini
      openai_api_key: ${openai_api_key}
      openai_base_url: ${openai_base_url}
      work_dir: null
      timeout: 900
      thinking: true
      system_prompt: null
      extra_args: []
```

## POC Notes

- `responses_server: null` is useful while proving the CLI wrapper loads and
  can execute prompts.
- For benchmark scoring, set `resources_server` to a verifier server or extend
  this agent with benchmark-specific verification.
- For the SWE-bench/OpenCode POC, the next step is to materialize the
  SWE-bench workspace for one task, run OpenCode inside that workspace, collect
  a patch/logs, and run the SWE-bench verifier.
