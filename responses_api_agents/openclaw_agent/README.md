# OpenClaw Agent

Runs standalone `openclaw agent --local --json` as a NeMo Gym
`responses_api_agent`.

This adapter follows the OpenClaw Harbor integration in
`NVIDIA/NeMo-Agent-Toolkit#1945`: it writes an isolated `openclaw.json`, runs
OpenClaw against a task workspace, captures the CLI JSON envelope and optional
session JSONL, then returns Gym-compatible response metadata.

## POC Notes

- The first proof-of-life kept `nemo_flow.enabled=false` and proved OpenClaw can
  solve the same one-row Django SWE-bench task through Gym.
- OpenClaw requires Node 22+. If `node_bin_dir` is unset, the adapter
  opportunistically prepends the newest local `~/.nvm/versions/node/v22*/bin`.
- For the NVIDIA OpenAI-compatible endpoint, the OpenClaw model selector has the
  shape `nvidia/<api-model-id>`; the first `nvidia` selects the OpenClaw
  provider and the remaining path is the model id sent to the API. The POC
  default is `nvidia/nvidia/qwen/qwen-235b`.
- `container_formatter` uses the same SWE-bench SIF convention as the OpenCode
  POC. The adapter copies `/testbed` to a host workspace, checks out the base
  commit, and collects `git diff`.
- `verify_swebench=true` reuses the OpenCode POC's SWE-bench verifier helper to
  compute `reward=1.0` only when the SWE-bench report marks the instance
  resolved.
- `nemo_flow.enabled=true` configures the OpenClaw native NeMoRelay plugin shape.
  Set `nemo_flow.plugin_local_path` to a local NeMoRelay OpenClaw plugin
  checkout to load the plugin through OpenClaw `plugins.load.paths`; the POC
  verified direct ATIF export with this path.
