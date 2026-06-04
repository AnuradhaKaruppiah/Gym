# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast
from uuid import uuid4

import ray
import yaml
from fastapi import Body, FastAPI
from minisweagent.config import builtin_config_dir, get_config_path
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import (
    ServerClient,
    get_first_server_config_dict,
)
from responses_api_agents.mini_swe_agent_2.relay_exporter import MiniSWERelayRun, format_output_dir


class MiniSWERelayConfig(BaseModel):
    enabled: bool = False
    output_dir: Optional[str] = None
    strict: bool = False
    mode: Literal["manual", "gateway", "sdk", "trajectory"] = "manual"
    command: str = "nemo-relay"


class MiniSWEObservabilityConfig(BaseModel):
    relay: MiniSWERelayConfig = Field(default_factory=MiniSWERelayConfig)


class MiniSWEAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    env: Literal["sandbox"]
    concurrency: int
    sandbox_provider: Optional[dict[str, Any]] = None
    sandbox_spec: Optional[dict[str, Any]] = None
    sandbox_environment_kwargs: Optional[dict[str, Any]] = None
    run_golden: bool = False
    step_timeout: int = 600
    eval_timeout: int = 1800
    skip_if_exists: bool = False
    step_limit: int = 250
    tool_choice: Optional[str | dict[str, Any]] = None
    sandbox_resource_profiles: Optional[list[dict[str, str]]] = None
    observability: MiniSWEObservabilityConfig = Field(default_factory=MiniSWEObservabilityConfig)


class MiniSWEAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
)
def runner_ray_remote(runner: Callable, params: dict[str, Any]) -> Any:
    return runner(**params)


def _json_dict_from_metadata(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"responses_create_params.metadata.{field_name} must be a JSON object")


def _responses_create_params_to_model_kwargs(
    params: dict[str, Any],
    *,
    default_tool_choice: Any = None,
) -> dict[str, Any]:
    """Convert Gym Responses API rollout params into mini-swe-agent chat-completions kwargs."""
    model_kwargs: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_logprobs", "parallel_tool_calls"):
        value = params.get(key)
        if value is not None:
            model_kwargs[key] = value

    max_output_tokens = params.get("max_output_tokens")
    if max_output_tokens is not None:
        model_kwargs["max_tokens"] = max_output_tokens

    metadata = params.get("metadata") or {}
    extra_body = _json_dict_from_metadata(metadata.get("extra_body"), field_name="extra_body")
    chat_template_kwargs = _json_dict_from_metadata(
        metadata.get("chat_template_kwargs"),
        field_name="chat_template_kwargs",
    )
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:
        model_kwargs["extra_body"] = extra_body

    tool_choice = default_tool_choice if default_tool_choice is not None else params.get("tool_choice")
    if tool_choice == "bash":
        model_kwargs["tool_choice"] = _bash_tool_choice()
    elif tool_choice is not None:
        model_kwargs["tool_choice"] = tool_choice

    return model_kwargs


def _bash_tool_choice() -> dict[str, Any]:
    return {"type": "function", "function": {"name": "bash"}}


def _sandbox_spec_for_instance(
    spec: dict[str, Any] | None,
    *,
    resource_profiles: list[dict[str, str]] | None,
    instance_id: str,
) -> dict[str, Any]:
    instance_spec = dict(spec or {})
    if not resource_profiles:
        return instance_spec

    resources = dict(instance_spec.get("resources") or {})
    digest = hashlib.sha256(instance_id.encode("utf-8")).digest()
    profile = resource_profiles[int.from_bytes(digest[:4], "big") % len(resource_profiles)]
    resources.update(profile)
    instance_spec["resources"] = resources
    return instance_spec


def _swebench_config_path() -> Path:
    for candidate in (
        builtin_config_dir / "extra" / "swebench.yaml",
        builtin_config_dir / "benchmarks" / "swebench.yaml",
    ):
        if candidate.exists():
            return candidate
    return builtin_config_dir / "extra" / "swebench.yaml"


def _swebench_image_name(instance: dict[str, Any], subset: str) -> str:
    image_name = instance.get("image_name")
    if image_name:
        return str(image_name)

    instance_id = instance["instance_id"]
    if subset == "verified":
        docker_compatible_id = instance_id.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()

    docker_compatible_id = instance_id.replace("__", "_s_")
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{docker_compatible_id}:latest".lower()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _relay_observability_plugin_config(
    *,
    output_dir: str,
    trajectory_id: str,
    instance_id: str,
    model_name: str,
    task_index: Any,
    rollout_index: Any,
) -> dict[str, Any]:
    return {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": True,
                        "output_directory": output_dir,
                        "filename": "events.atof.jsonl",
                        "mode": "overwrite",
                    },
                    "atif": {
                        "enabled": True,
                        "agent_name": "mini-swe-agent",
                        "model_name": model_name,
                        "output_directory": output_dir,
                        "filename_template": "trajectory-{session_id}.atif.json",
                        "extra": {
                            "gym_agent": "mini_swe_agent_2",
                            "instance_id": instance_id,
                            "trajectory_id": trajectory_id,
                            "task_index": task_index,
                            "rollout_index": rollout_index,
                        },
                    },
                },
            }
        ],
    }


def _wait_for_relay_gateway(gateway_url: str, process: subprocess.Popen[str], *, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    health_url = f"{gateway_url}/healthz"
    while time.time() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                "nemo-relay gateway exited before becoming healthy "
                f"(returncode={process.returncode}).\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if 200 <= response.status < 300:
                    return
        except (TimeoutError, urllib.error.URLError):
            time.sleep(0.1)
    raise RuntimeError(f"nemo-relay gateway did not become healthy at {health_url}")


def _stop_relay_gateway(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    try:
        process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=1)


def _start_relay_gateway(
    *,
    command: str,
    upstream_base_url: str,
    plugin_config: dict[str, Any],
    api_key: str | None,
) -> tuple[subprocess.Popen[str], str]:
    port = _free_loopback_port()
    gateway_url = f"http://127.0.0.1:{port}"
    argv = [
        *shlex.split(command),
        "--bind",
        f"127.0.0.1:{port}",
        "--openai-base-url",
        upstream_base_url,
        "--plugin-config",
        json.dumps(plugin_config),
    ]
    env = os.environ.copy()
    if api_key and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = api_key
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        _wait_for_relay_gateway(gateway_url, process)
    except Exception:
        _stop_relay_gateway(process)
        raise
    return process, gateway_url


def _relay_artifacts(output_dir: str) -> dict[str, str]:
    root = Path(output_dir)
    artifacts: dict[str, str] = {}
    atof_path = root / "events.atof.jsonl"
    if atof_path.exists():
        artifacts["atof"] = str(atof_path)
    atif_paths = sorted(root.glob("trajectory-*.atif.json"))
    if atif_paths:
        artifacts["atif"] = str(atif_paths[-1])
    return artifacts


def _start_relay_sdk_observer(
    *,
    output_dir: str,
    trajectory_id: str,
    instance_id: str,
    model_name: str,
    task_index: Any,
    rollout_index: Any,
) -> Any:
    from nemo_relay.integrations.mini_swe_agent import MiniSweAgentObservabilityConfig, start_observability

    return start_observability(
        MiniSweAgentObservabilityConfig(
            output_dir=output_dir,
            model_name=model_name,
            trajectory_id=trajectory_id,
            instance_id=instance_id,
            task_index=task_index,
            rollout_index=rollout_index,
            extra={"gym_agent": "mini_swe_agent_2"},
        )
    )


def _export_relay_native_trajectory(
    *,
    output_dir: str,
    trajectory_id: str,
    instance_id: str,
    model_name: str,
    task_index: Any,
    rollout_index: Any,
    trajectory: dict[str, Any],
    task: str,
) -> dict[str, str]:
    from nemo_relay.integrations.mini_swe_agent import MiniSweAgentObservabilityConfig, export_trajectory

    return export_trajectory(
        trajectory,
        MiniSweAgentObservabilityConfig(
            output_dir=output_dir,
            model_name=model_name,
            trajectory_id=trajectory_id,
            instance_id=instance_id,
            task_index=task_index,
            rollout_index=rollout_index,
            extra={
                "gym_agent": "mini_swe_agent_2",
                "binding": "mini-swe-agent-native-trajectory",
                "reconstructed": True,
            },
        ),
        task=task,
    )


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _strip_extra(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        item = item.model_dump()
    if not isinstance(item, dict):
        return {"type": "message", "role": "user", "content": str(item)}
    return {key: value for key, value in item.items() if key != "extra"}


def _split_trajectory_for_responses(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    input_messages: list[dict[str, Any]] = []
    output_items: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    in_initial_prompt = True

    for message in messages:
        role = message.get("role")
        if in_initial_prompt and role in {"system", "user"}:
            input_messages.append(
                {"type": "message", "role": role, "content": _message_content_to_text(message.get("content"))}
            )
            continue

        in_initial_prompt = False
        if message.get("object") == "response":
            response = _strip_extra(message)
            raw_responses.append(response)
            output_items.extend(_strip_extra(item) for item in response.get("output", []))
        elif role == "assistant":
            content = _message_content_to_text(message.get("content"))
            if content:
                output_items.append(
                    {
                        "id": message.get("id") or f"msg_{uuid4()}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                )
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                output_items.append(
                    {
                        "id": tool_call.get("id") or f"fc_{uuid4()}",
                        "type": "function_call",
                        "name": function.get("name") or tool_call.get("name") or "",
                        "call_id": tool_call.get("id") or tool_call.get("call_id") or "",
                        "arguments": function.get("arguments") or tool_call.get("arguments") or "{}",
                    }
                )
        elif role == "tool":
            output_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or message.get("call_id") or "",
                    "output": _message_content_to_text(message.get("content")),
                }
            )
        elif message.get("type") == "function_call_output":
            output_items.append(_strip_extra(message))

    return input_messages, output_items, raw_responses


def _default_response_object() -> dict[str, Any]:
    return {
        "id": f"resp_{str(uuid4())}",
        "created_at": int(time.time()),
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "background": False,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": {
            "effort": None,
            "generate_summary": None,
            "summary": None,
        },
        "service_tier": "default",
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
        "user": None,
        "prompt_cache_key": None,
        "safety_identifier": None,
        "store": True,
    }


def _is_resolved(instance_id: str, eval_report: dict[str, Any]) -> bool:
    try:
        if not eval_report:
            return False
        report = eval_report["eval_report"][instance_id]
        resolved = bool(report["resolved"])
        if not report.get("tests_status"):
            return False

        tests_status = report["tests_status"]
        f2f = tests_status.get("FAIL_TO_PASS", {})
        p2p = tests_status.get("PASS_TO_PASS", {})
        total_reported = (
            len(f2f.get("success", []))
            + len(f2f.get("failure", []))
            + len(p2p.get("success", []))
            + len(p2p.get("failure", []))
        )
        return resolved and total_reported > 0
    except Exception as exc:
        print(f"Error in _is_resolved: {exc}", flush=True)
        return False


def _metadata_dict(verify_response: dict[str, Any]) -> dict[str, Any]:
    metadata = verify_response.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _eval_report_map(verify_response: dict[str, Any]) -> dict[str, Any]:
    report = _metadata_dict(verify_response).get("eval_report") or {}
    return report if isinstance(report, dict) else {}


def _eval_instance_report(verify_response: dict[str, Any]) -> dict[str, Any]:
    report_map = _eval_report_map(verify_response)
    instance_id = verify_response.get("instance_id") or _metadata_dict(verify_response).get("instance_id")
    if instance_id is not None:
        report = report_map.get(str(instance_id))
        if isinstance(report, dict):
            return report

    for report in report_map.values():
        if isinstance(report, dict) and "resolved" in report:
            return report
    return {}


def _test_status_counts(verify_response: dict[str, Any]) -> dict[str, int]:
    report = _eval_instance_report(verify_response)
    tests_status = report.get("tests_status") if isinstance(report, dict) else None
    if not isinstance(tests_status, dict):
        return {}

    counts: dict[str, int] = {}
    for suite_name, suite_report in tests_status.items():
        if not isinstance(suite_report, dict):
            continue
        prefix = str(suite_name).lower()
        counts[f"{prefix}_success"] = len(suite_report.get("success") or [])
        counts[f"{prefix}_failure"] = len(suite_report.get("failure") or [])
    return counts


def _run_eval_v2(
    *,
    instance: dict[str, Any],
    env: Any,
    model_patch: str,
    instance_dir: Path,
    run_id: str,
    is_golden: bool,
) -> dict[str, Any]:
    from swebench.harness.constants import SWEbenchInstance
    from swebench.harness.docker_build import setup_logger
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    swebench_instance = cast(SWEbenchInstance, instance)
    test_spec = make_test_spec(swebench_instance)
    pred = {"instance_id": test_spec.instance_id, "model_patch": model_patch}

    instance_dir.mkdir(parents=True, exist_ok=True)
    log_file = instance_dir / f"run_instance_{run_id}.log"
    report_path = instance_dir / f"report_{run_id}.json"
    patch_file = instance_dir / f"patch_{run_id}.diff"
    patch_file.write_text(model_patch)

    logger = setup_logger(test_spec.instance_id, log_file)
    logger.info(f"DEBUG test_spec {test_spec}")
    logger.info(f"DEBUG eval_script {test_spec.eval_script}")

    if is_golden:
        env.execute(f"cat > patch.diff <<'EOF'\n{model_patch}\n\nEOF")
        env.execute("git status --porcelain")
        env.execute("git apply --check patch.diff")
        env.execute("git apply patch.diff")

    eval_script = test_spec.eval_script.replace("#!/bin/bash", "")
    result = env.execute(eval_script, is_eval=True)
    test_output = result["output"]
    returncode = result["returncode"]
    print(f"[EVAL]{test_spec.instance_id} returncode: {returncode}", flush=True)

    test_output_path = instance_dir / f"test_output_{run_id}.txt"
    test_output_path.write_text(test_output)
    print(f"[EVAL]{test_spec.instance_id} Test output written to {test_output_path}", flush=True)

    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        test_log_path=str(test_output_path),
        include_tests_status=True,
    )
    print(f"[EVAL]{test_spec.instance_id} Result: resolved: {report[test_spec.instance_id]['resolved']}", flush=True)

    report_path.write_text(json.dumps(report, indent=4))
    return {
        "instance_id": test_spec.instance_id,
        "model_patch": model_patch,
        "eval_report": report,
    }


def _run_mini_swe_v2(**params: Any) -> dict[str, Any]:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    instance = params.get("instance_dict")
    if isinstance(instance, str):
        instance = json.loads(instance)
    if not isinstance(instance, dict):
        raise ValueError("mini-swe-agent v2 path requires instance_dict")

    instance = dict(instance)
    instance_id = str(params.get("instance_id") or instance["instance_id"]).lower()
    instance["instance_id"] = instance_id

    output_dir = Path(params["output"])
    instance_dir = output_dir / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(get_config_path(params["config"]).read_text())
    model_config = config.setdefault("model", {})
    model_config["model_class"] = "litellm"
    model_config["model_name"] = params["model"]
    model_config.setdefault("cost_tracking", "ignore_errors")
    model_kwargs = model_config.setdefault("model_kwargs", {})
    model_kwargs["api_key"] = params["api_key"]
    model_kwargs["base_url"] = params["base_url"]
    model_kwargs.pop("api_base", None)
    max_output_tokens = model_kwargs.pop("max_output_tokens", None)
    if max_output_tokens is not None and "max_tokens" not in model_kwargs:
        model_kwargs["max_tokens"] = max_output_tokens

    environment_config = config.setdefault("environment", {})
    environment_config["image"] = _swebench_image_name(instance, params["subset"])
    environment_config["step_timeout"] = params["step_timeout"]
    environment_config["eval_timeout"] = params["eval_timeout"]
    environment_config["instance_id"] = instance_id
    environment_config["environment_class"] = (
        "responses_api_agents.mini_swe_agent_2.sandbox_environment.MiniSWESandboxEnvironment"
    )

    agent_config = config.get("agent", {})
    agent_config["step_limit"] = params["step_limit"]
    agent_config.pop("collapse_limit", None)

    run_id = f"{int(time.time())}_{uuid4()}"
    trajectory_path = instance_dir / f"{instance_id}_{run_id}.traj.json"
    agent_config["output_path"] = trajectory_path
    relay_config = dict(params.get("relay") or {})
    relay_output_template = relay_config.get("output_dir") or "{output}/{instance_id}/relay/{run_id}"
    relay_output_dir = format_output_dir(
        relay_output_template,
        {
            "output": str(output_dir),
            "instance_id": instance_id,
            "run_id": run_id,
            "task_index": params.get("task_index"),
            "rollout_index": params.get("rollout_index"),
            "model": params.get("model"),
        },
    )
    relay_mode = str(relay_config.get("mode") or "manual")
    if relay_mode not in {"manual", "gateway", "sdk", "trajectory"}:
        raise ValueError(f"Unsupported mini-swe-agent relay mode: {relay_mode}")

    relay_enabled = bool(relay_config.get("enabled", False))
    relay_trajectory_id = relay_config.get("trajectory_id") or f"{instance_id}-{run_id}"
    relay_gateway_process: subprocess.Popen[str] | None = None
    relay_gateway_artifacts: dict[str, str] = {}
    relay_sdk_run: Any | None = None
    relay_sdk_artifacts: dict[str, str] = {}
    relay_trajectory_artifacts: dict[str, str] = {}

    if relay_enabled and relay_mode == "gateway" and not params["run_golden"]:
        plugin_config = _relay_observability_plugin_config(
            output_dir=relay_output_dir,
            trajectory_id=relay_trajectory_id,
            instance_id=instance_id,
            model_name=str(params["model"]),
            task_index=params.get("task_index"),
            rollout_index=params.get("rollout_index"),
        )
        relay_gateway_process, relay_gateway_url = _start_relay_gateway(
            command=str(relay_config.get("command") or "nemo-relay"),
            upstream_base_url=str(params["base_url"]),
            plugin_config=plugin_config,
            api_key=params.get("api_key"),
        )
        model_kwargs["base_url"] = f"{relay_gateway_url}/v1"
        extra_headers = dict(model_kwargs.get("extra_headers") or {})
        extra_headers.update(
            {
                "x-nemo-relay-agent-kind": "mini-swe-agent",
                "x-nemo-relay-session-id": str(relay_trajectory_id),
            }
        )
        model_kwargs["extra_headers"] = extra_headers

    if relay_enabled and relay_mode == "sdk" and not params["run_golden"]:
        relay_sdk_run = _start_relay_sdk_observer(
            output_dir=relay_output_dir,
            trajectory_id=relay_trajectory_id,
            instance_id=instance_id,
            model_name=str(params["model"]),
            task_index=params.get("task_index"),
            rollout_index=params.get("rollout_index"),
        )

    relay_run = MiniSWERelayRun(
        enabled=relay_enabled and relay_mode == "manual",
        output_dir=relay_output_dir,
        strict=bool(relay_config.get("strict", False)),
        trajectory_id=relay_trajectory_id,
        instance_id=instance_id,
        model_name=str(params["model"]),
        task_index=params.get("task_index"),
        rollout_index=params.get("rollout_index"),
    )
    env = None
    agent = None
    result_payload: dict[str, Any] | None = None
    try:
        with relay_run:
            print(f"[EVAL]{instance_id} Creating environment...", flush=True)
            relay_run.record_mark("mini_swe_agent_2.environment.create.start", {"instance_id": instance_id})
            env = get_environment(environment_config)
            relay_run.record_mark("mini_swe_agent_2.environment.create.end", {"instance_id": instance_id})
            print(f"[EVAL]{instance_id} Environment created", flush=True)

            model = get_model(config=model_config)
            if relay_sdk_run is not None:
                agent = DefaultAgent(model, env, observer=relay_sdk_run.observer, **agent_config)
            else:
                agent = DefaultAgent(model, env, **agent_config)

            if params["run_golden"]:
                exit_status = "Gold Patch Applied"
                model_patch = instance.get("patch", "")
                data = agent.save(None, {"messages": []})
            else:
                print(f"[EVAL]{instance_id} Running mini-swe-agent v2...", flush=True)
                relay_run.record_mark(
                    "mini_swe_agent_2.agent.run.start",
                    {"instance_id": instance_id, "problem_statement": instance["problem_statement"]},
                )
                info = agent.run(instance["problem_statement"])
                relay_run.record_mark(
                    "mini_swe_agent_2.agent.run.end",
                    {"instance_id": instance_id, "exit_status": info.get("exit_status", "")},
                )
                exit_status = info.get("exit_status", "")
                model_patch = info.get("submission", "")
                data = agent.save(
                    trajectory_path,
                    {"instance_id": instance_id},
                )

                if relay_enabled and relay_mode == "trajectory":
                    try:
                        relay_trajectory_artifacts = _export_relay_native_trajectory(
                            output_dir=relay_output_dir,
                            trajectory_id=relay_trajectory_id,
                            instance_id=instance_id,
                            model_name=str(params["model"]),
                            task_index=params.get("task_index"),
                            rollout_index=params.get("rollout_index"),
                            trajectory=data,
                            task=str(instance["problem_statement"]),
                        )
                    except Exception as exc:
                        if bool(relay_config.get("strict", False)):
                            raise
                        print(
                            f"[MiniSWEAgent][Relay] failed to export native trajectory: {exc}",
                            flush=True,
                        )

            if relay_gateway_process is not None:
                _stop_relay_gateway(relay_gateway_process)
                relay_gateway_process = None
                relay_gateway_artifacts = _relay_artifacts(relay_output_dir)

            relay_run.record_agent_messages(data.get("messages", []))
            print(f"[EVAL]{instance_id} Running eval", flush=True)
            relay_run.record_mark("mini_swe_agent_2.eval.start", {"instance_id": instance_id})
            eval_report = _run_eval_v2(
                instance=instance,
                env=env,
                model_patch=model_patch,
                instance_dir=instance_dir,
                run_id=run_id,
                is_golden=params["run_golden"],
            )
            relay_run.record_mark("mini_swe_agent_2.eval.end", {"instance_id": instance_id})
            print(f"[EVAL]{instance_id} Eval completed", flush=True)

            input_messages, response_output, responses = _split_trajectory_for_responses(data.get("messages", []))

            result_payload = {
                instance_id: {
                    "input_messages": input_messages,
                    "response_output": response_output,
                    "responses": responses,
                    "eval_report": eval_report,
                    "exit_status": exit_status,
                }
            }
        if relay_sdk_run is not None:
            relay_sdk_artifacts = relay_sdk_run.close("run_complete")
            relay_sdk_run = None
        relay_artifacts = relay_run.artifacts()
        relay_artifacts.update(relay_gateway_artifacts)
        relay_artifacts.update(relay_sdk_artifacts)
        relay_artifacts.update(relay_trajectory_artifacts)
        assert result_payload is not None
        if relay_artifacts:
            result_payload[instance_id]["eval_report"]["relay_artifacts"] = relay_artifacts
        return result_payload
    finally:
        if relay_sdk_run is not None:
            relay_sdk_run.close("finally")
        _stop_relay_gateway(relay_gateway_process)
        if env and hasattr(env, "cleanup"):
            env.cleanup()


def run_mini_swe_with_sandbox(**params: Any) -> Any:
    return _run_mini_swe_v2(**params)


class MiniSWEAgent(SimpleResponsesAPIAgent):
    config: MiniSWEAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        metrics, _, _, max_k = compute_pass_majority_metrics(tasks)
        metrics.pop("per_sample_aggregate", None)

        all_rollouts = [rollout for task in tasks for rollout in task]
        rollout_count = len(all_rollouts)
        resolved_task_count = sum(1 for task in tasks if any(float(r.get("reward", 0.0) or 0.0) >= 1.0 for r in task))
        eval_error_count = sum(1 for rollout in all_rollouts if _metadata_dict(rollout).get("error"))
        eval_report_count = sum(1 for rollout in all_rollouts if _eval_report_map(rollout))
        tests_status_count = sum(1 for rollout in all_rollouts if _eval_instance_report(rollout).get("tests_status"))
        patch_applied_count = sum(
            1 for rollout in all_rollouts if _eval_instance_report(rollout).get("patch_successfully_applied")
        )

        metrics.update(
            {
                "task_count": len(tasks),
                "rollout_count": rollout_count,
                "max_rollouts_per_task": max_k,
                "resolved_task_count": resolved_task_count,
                "resolved_task_rate": 100.0 * resolved_task_count / len(tasks) if tasks else 0.0,
                "eval_error_rollout_count": eval_error_count,
                "eval_error_rate": 100.0 * eval_error_count / rollout_count if rollout_count else 0.0,
                "eval_report_rollout_count": eval_report_count,
                "eval_report_rate": 100.0 * eval_report_count / rollout_count if rollout_count else 0.0,
                "tests_status_rollout_count": tests_status_count,
                "tests_status_rate": 100.0 * tests_status_count / rollout_count if rollout_count else 0.0,
                "patch_applied_rollout_count": patch_applied_count,
                "patch_applied_rate": 100.0 * patch_applied_count / rollout_count if rollout_count else 0.0,
                "per_task_metrics": self._compute_per_task_eval_metrics(tasks),
            }
        )

        test_status_totals: dict[str, int] = {}
        for rollout in all_rollouts:
            for key, value in _test_status_counts(rollout).items():
                test_status_totals[key] = test_status_totals.get(key, 0) + value
        metrics.update({f"tests_status/{key}": value for key, value in sorted(test_status_totals.items())})

        return metrics

    def _compute_per_task_eval_metrics(self, tasks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        per_task_metrics: list[dict[str, Any]] = []
        for fallback_idx, rollouts in enumerate(tasks):
            if not rollouts:
                continue

            first = rollouts[0]
            task_index = first.get(TASK_INDEX_KEY_NAME, fallback_idx)
            instance_id = first.get("instance_id") or _metadata_dict(first).get("instance_id")
            resolved_count = sum(1 for rollout in rollouts if float(rollout.get("reward", 0.0) or 0.0) >= 1.0)
            error_count = sum(1 for rollout in rollouts if _metadata_dict(rollout).get("error"))
            eval_report_count = sum(1 for rollout in rollouts if _eval_report_map(rollout))
            tests_status_count = sum(1 for rollout in rollouts if _eval_instance_report(rollout).get("tests_status"))
            patch_applied_count = sum(
                1 for rollout in rollouts if _eval_instance_report(rollout).get("patch_successfully_applied")
            )

            task_metrics: dict[str, Any] = {
                TASK_INDEX_KEY_NAME: task_index,
                "instance_id": instance_id,
                "rollout_count": len(rollouts),
                "resolved": resolved_count > 0,
                "resolved_rollout_count": resolved_count,
                "eval_error_rollout_count": error_count,
                "eval_report_rollout_count": eval_report_count,
                "tests_status_rollout_count": tests_status_count,
                "patch_applied_rollout_count": patch_applied_count,
            }

            test_status_totals: dict[str, int] = {}
            for rollout in rollouts:
                for key, value in _test_status_counts(rollout).items():
                    test_status_totals[key] = test_status_totals.get(key, 0) + value
            task_metrics.update({f"tests_status/{key}": value for key, value in sorted(test_status_totals.items())})
            per_task_metrics.append(task_metrics)

        return per_task_metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key_metrics: dict[str, Any] = {}
        key_metrics.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key_metrics.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        for key in (
            "mean/reward",
            "resolved_task_count",
            "task_count",
            "resolved_task_rate",
            "eval_error_rate",
            "tests_status_rate",
        ):
            if key in agent_metrics:
                key_metrics[key] = agent_metrics[key]
        return key_metrics

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError

    async def run(self, body: MiniSWEAgentRunRequest) -> MiniSWEAgentVerifyResponse:
        async with self.sem:
            model_server_name = self.config.model_server.name
            global_config_dict = ServerClient.load_from_global_config().global_config_dict

            model_server_config = get_first_server_config_dict(
                global_config_dict,
                model_server_name,
            )

            policy_model_name = global_config_dict["policy_model_name"]

            ##### MINI-SWE-AGENT CONFIG #####
            subset = body.subset
            split = body.split
            workers = 1
            run_golden = self.config.run_golden
            base_url = f"http://{model_server_config['host']}:{model_server_config['port']}/v1"
            dummy_key = "dummy_key"
            model_name = f"hosted_vllm/{policy_model_name}"
            step_timeout = self.config.step_timeout
            eval_timeout = self.config.eval_timeout
            step_limit = self.config.step_limit

            instance_id = body.instance_id

            mini_swe_config_path = _swebench_config_path()
            config = yaml.safe_load(get_config_path(mini_swe_config_path).read_text())
            body_dict = body.model_dump()
            responses_create_params_dict = body.responses_create_params.model_dump(exclude_none=True)

            default_model_kwargs = config["model"]["model_kwargs"]
            temperature = (
                body.responses_create_params.temperature
                if body.responses_create_params.temperature is not None
                else default_model_kwargs["temperature"]
            )
            top_p = (
                body.responses_create_params.top_p
                if body.responses_create_params.top_p is not None
                else default_model_kwargs["top_p"]
            )
            model_kwargs = _responses_create_params_to_model_kwargs(
                responses_create_params_dict,
                default_tool_choice=self.config.tool_choice,
            )
            if model_kwargs:
                config.setdefault("model", {}).setdefault("model_kwargs", {}).update(model_kwargs)

            output_file_dir = f"{Path.cwd()}/results/{subset}/{policy_model_name}"
            config_path = mini_swe_config_path
            should_write_config = bool(model_kwargs)
            if self.config.sandbox_provider is None:
                raise ValueError("mini_swe_agent_2 requires sandbox_provider")
            config.setdefault("environment", {}).update(self.config.sandbox_environment_kwargs or {})
            config["environment"]["provider"] = self.config.sandbox_provider
            config["environment"]["spec"] = _sandbox_spec_for_instance(
                self.config.sandbox_spec,
                resource_profiles=self.config.sandbox_resource_profiles,
                instance_id=instance_id,
            )
            should_write_config = True

            if should_write_config:
                config_output_dir = Path(output_file_dir) / "_configs"
                config_output_dir.mkdir(parents=True, exist_ok=True)
                config_path = config_output_dir / f"{instance_id}.sandbox.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))

            if self.config.skip_if_exists:
                if Path(f"{output_file_dir}/{instance_id}/{instance_id}.json").exists():
                    with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "r") as f:
                        print(f"Skipping {instance_id} because it already exists")
                        verify_response = MiniSWEAgentVerifyResponse.model_validate_json(f.read())
                    return verify_response

            #### RUN MINI-SWE-AGENT #####
            try:
                params = dict(
                    subset=subset,
                    split=split,
                    workers=workers,
                    output=output_file_dir,
                    model=model_name,
                    api_key=dummy_key,
                    base_url=base_url,
                    env="sandbox",
                    run_golden=run_golden,
                    instance_id=instance_id,
                    config=config_path,
                    instance_dict=body_dict,
                    responses_create_params=json.dumps(responses_create_params_dict),
                    step_timeout=step_timeout,
                    eval_timeout=eval_timeout,
                    step_limit=step_limit,
                    task_index=body_dict.get(TASK_INDEX_KEY_NAME),
                    rollout_index=body_dict.get(ROLLOUT_INDEX_KEY_NAME),
                    relay=self.config.observability.relay.model_dump(),
                )
                future = runner_ray_remote.remote(run_mini_swe_with_sandbox, params)
                result = await asyncio.to_thread(ray.get, future)
                result = result[instance_id]
                input_messages = result["input_messages"]
                response_output = result["response_output"]
                responses = result["responses"]
                reward = 1.0 if _is_resolved(instance_id, result["eval_report"]) else 0.0

            except Exception as e:
                error_info = {"error": str(e), "traceback": traceback.format_exc()}
                print(f"Error running mini-swe-agent: {e}\n{error_info['traceback']}", flush=True)
                result = {"eval_report": error_info}
                input_messages = []
                response_output = []
                responses = []
                reward = 0.0

            body.responses_create_params.input = input_messages
            response = _default_response_object()
            if responses:
                response.update(dict(responses[-1]))
            response.pop("extra", None)
            response["model"] = policy_model_name
            response["temperature"] = temperature
            response["top_p"] = top_p
            response["output"] = response_output

            verify_response = MiniSWEAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                reward=reward,
                response=response,
                instance_id=instance_id,
                metadata=result.get("eval_report", {}) if result else {},
            )

            output_path = Path(f"{output_file_dir}/{instance_id}")
            output_path.mkdir(parents=True, exist_ok=True)

            with open(f"{output_file_dir}/{instance_id}/{instance_id}.json", "w") as f:
                json.dump(verify_response.model_dump(), f)

            return verify_response


if __name__ == "__main__":
    MiniSWEAgent.run_webserver()
