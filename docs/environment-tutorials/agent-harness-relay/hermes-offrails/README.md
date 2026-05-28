# Hermes Off-Rails Trajectory

This directory preserves an earlier Relay-captured Hermes run for
`django__django-13741` where the agent solved the task poorly from an execution
policy perspective. The run is intentionally kept as a trajectory-evaluation
example, not as a recommended configuration.

## What Happened

Hermes was run with its default tool exposure (`enabled_toolsets: null`), which
made browser and web-search tools available. On this SWE-bench task, the agent
drifted into external research instead of staying focused on the local checkout
and using repository tools.

The preserved Relay ATIF shows:

- 122 ATIF steps
- 31 agent steps, 31 user steps, 60 system steps
- 13 `web_search` calls
- browser tool calls including `browser_snapshot`, `browser_scroll`,
  `browser_navigate`, and `web_extract`
- only 1 `execute_code` call

The later constrained Hermes run in `../hermes/` fixes this by using:

```bash
'hermes_agent.responses_api_agents.hermes_agent.enabled_toolsets=[terminal,file,code_execution]'
```

That constrained run uses local repository tools such as `search_files`,
`read_file`, and `patch`, and resolves the SWE-bench task.

## Why Keep This

The final score alone is not enough to understand an agent-harness run. A task
can appear to make progress while relying on the wrong capabilities, producing a
trajectory that is not representative of the execution environment we intended
to evaluate.

This artifact demonstrates why trajectory evaluation matters:

- It reveals whether the agent used the intended capability set.
- It distinguishes local workspace work from external web research.
- It makes tool-policy regressions visible even when the run produces plausible
  prose or a patch.
- It gives reviewers a concrete before/after comparison against the constrained
  Hermes trace.

## Files

- `artifacts/with-relay.offrails.nemo-relay.atif.json`: Relay ATIF from the
  unconstrained Hermes run.

## Phoenix Reference

This trace was also uploaded to the local Phoenix database as:

```text
gym-hermes-offrails-with-relay-20260527-200335
```

Use it alongside the constrained Hermes projects to compare the behavioral
difference in the trace viewer.
