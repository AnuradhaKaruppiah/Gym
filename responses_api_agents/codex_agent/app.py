# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import logging
import os
import shlex
import shutil
import subprocess
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)
MIN_CODEX_RELAY_VERSION = (0, 129, 0)
MIN_CODEX_HOOK_TRUST_BYPASS_VERSION = (0, 136, 0)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_to_text(part) for part in content)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        if "content" in content:
            return _content_to_text(content.get("content"))
        if "output" in content:
            return _content_to_text(content.get("output"))
    if hasattr(content, "text"):
        return str(getattr(content, "text") or "")
    if hasattr(content, "content"):
        return _content_to_text(getattr(content, "content"))
    return str(content or "")


def _toml_string(value: str) -> str:
    # JSON string syntax is valid TOML basic string syntax for the paths/commands used here.
    return json.dumps(value)


def _parse_codex_version(output: str) -> Optional[tuple[int, int, int]]:
    for token in output.replace("\n", " ").split():
        parts = token.split(".")
        if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
            return tuple(int(part) for part in parts[:3])
    return None


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def _extract_instruction(body_input: Any) -> tuple[str, Optional[str]]:
    """Return (latest user message, optional system message) from Responses input."""
    items = [body_input] if isinstance(body_input, str) else list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            system_message = _content_to_text(content)
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            user_message = _content_to_text(content)
            break

    return user_message, system_message


def _arguments_to_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value)


def _usage_from_event(event: dict[str, Any]) -> tuple[int, int]:
    candidates = [
        event.get("usage"),
        (event.get("response") or {}).get("usage") if isinstance(event.get("response"), dict) else None,
        (event.get("message") or {}).get("usage") if isinstance(event.get("message"), dict) else None,
        (event.get("item") or {}).get("usage") if isinstance(event.get("item"), dict) else None,
    ]
    input_tokens = 0
    output_tokens = 0
    for usage in candidates:
        if not isinstance(usage, dict):
            continue
        input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return input_tokens, output_tokens


def _append_message(output_items: list[Any], text: str) -> None:
    output_items.append(
        NeMoGymResponseOutputMessage(
            id=f"msg-{len(output_items)}",
            content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
            role="assistant",
            status="completed",
            type="message",
        )
    )


def _append_tool_call(output_items: list[Any], call_id: str, name: str, arguments: Any) -> None:
    output_items.append(
        NeMoGymResponseFunctionToolCall(
            arguments=_arguments_to_json(arguments),
            call_id=call_id,
            name=name,
            type="function_call",
            id=call_id,
            status="completed",
        )
    )


def _append_tool_output(output_items: list[Any], call_id: str, output: Any) -> None:
    output_items.append(
        NeMoGymFunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output=_content_to_text(output),
            status="completed",
        )
    )


def _parse_codex_item(output_items: list[Any], item: dict[str, Any]) -> None:
    item_type = item.get("type") or item.get("kind")
    if item_type == "agent_message":
        text = _content_to_text(item.get("text") or item.get("content"))
        if text:
            _append_message(output_items, text)
    elif item_type == "message" and item.get("role") == "assistant":
        text = _content_to_text(item.get("content") or item.get("text"))
        if text:
            _append_message(output_items, text)
    elif item_type in {"function_call", "tool_call"}:
        call_id = item.get("call_id") or item.get("id") or f"call-{uuid4().hex[:8]}"
        _append_tool_call(output_items, call_id, item.get("name") or "", item.get("arguments") or item.get("input"))
    elif item_type in {"function_call_output", "tool_result"}:
        call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id") or f"call-{uuid4().hex[:8]}"
        _append_tool_output(output_items, call_id, item.get("output") or item.get("content"))


def parse_codex_jsonl(stdout: str) -> tuple[list[Any], dict[str, int]]:
    """Convert `codex exec --json` stdout into Gym response output items."""
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        usage_input, usage_output = _usage_from_event(event)
        input_tokens += usage_input
        output_tokens += usage_output

        etype = str(event.get("type") or event.get("event") or event.get("kind") or "")
        item = event.get("item")
        if isinstance(item, dict):
            _parse_codex_item(output_items, item)
            continue

        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            text = _content_to_text(message.get("content") or message.get("text"))
            if text:
                _append_message(output_items, text)
            continue

        if etype in {"assistant_message", "agent_message", "final_message", "message"}:
            text = _content_to_text(event.get("content") or event.get("text") or event.get("message"))
            if text:
                _append_message(output_items, text)
        elif etype in {"tool_call", "function_call"}:
            call_id = event.get("call_id") or event.get("id") or f"call-{uuid4().hex[:8]}"
            _append_tool_call(
                output_items, call_id, event.get("name") or "", event.get("arguments") or event.get("input")
            )
        elif etype in {"tool_result", "function_call_output", "observation"}:
            call_id = event.get("call_id") or event.get("tool_call_id") or event.get("id") or f"call-{uuid4().hex[:8]}"
            _append_tool_output(
                output_items, call_id, event.get("output") or event.get("content") or event.get("result")
            )
        elif etype.endswith(".begin") and "exec" in etype:
            call_id = event.get("call_id") or event.get("id") or f"exec-{uuid4().hex[:8]}"
            _append_tool_call(output_items, call_id, "exec_command", {"cmd": event.get("cmd") or event.get("command")})
        elif etype.endswith(".end") and "exec" in etype:
            call_id = event.get("call_id") or event.get("id") or f"exec-{uuid4().hex[:8]}"
            _append_tool_output(
                output_items, call_id, event.get("output") or event.get("stdout") or event.get("stderr")
            )

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


class CodexOpenInferenceConfig(BaseModel):
    enabled: bool = False
    endpoint: Optional[str] = None
    transport: str = "http_binary"
    service_name: str = "nemo-relay-codex"
    service_namespace: Optional[str] = "gym"
    service_version: Optional[str] = None
    instrumentation_scope: str = "nemo-relay-openinference"
    timeout_millis: int = 3000
    project_name: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)
    resource_attributes: dict[str, str] = Field(default_factory=dict)


class CodexNemoRelayConfig(BaseModel):
    enabled: bool = False
    command: str = "nemo-relay"
    bypass_hook_trust: bool = False
    interactive_prompt_delay: float = 5.0
    interactive_startup_timeout: float = 60.0
    interactive_shutdown_timeout: float = 10.0
    output_dir: Optional[str] = None
    atof_filename: str = "codex.atof.jsonl"
    atif_filename_template: str = "codex-{session_id}.atif.json"
    agent_name: str = "codex"
    agent_version: str = "0.1.0"
    mode: str = "overwrite"
    openinference: CodexOpenInferenceConfig = Field(default_factory=CodexOpenInferenceConfig)


class CodexAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: Optional[ResourcesServerRef] = None
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "codex"
    model: Optional[str] = None
    profile: Optional[str] = None
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    work_dir: Optional[str] = None
    workspace_root: str = "outputs/codex_agent/workspaces"
    container_formatter: Optional[str] = None
    apptainer_command: str = "apptainer"
    setup_timeout: int = 900
    timeout: int = 900
    sandbox: str = "workspace-write"
    approval_policy: str = "never"
    ephemeral: bool = True
    ignore_user_config: bool = False
    ignore_rules: bool = False
    skip_git_repo_check: bool = True
    system_prompt: Optional[str] = None
    config_overrides: list[str] = []
    extra_args: list[str] = []
    nemo_relay: CodexNemoRelayConfig = Field(default_factory=CodexNemoRelayConfig)
    verify_swebench: bool = False
    swebench_setup_dir: Optional[str] = None
    swebench_results_root: str = "outputs/codex_agent/swebench-verifier"
    swebench_verifier_timeout: int = 1200
    swebench_model_name: str = "codex_agent"

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class CodexAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class CodexAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class CodexAgent(SimpleResponsesAPIAgent):
    config: CodexAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("Codex command %r is not on PATH yet", self.config.command)
        relay_command = shlex.split(self.config.nemo_relay.command)[0] if self.config.nemo_relay.command else ""
        if self.config.nemo_relay.enabled and not (relay_command and (shutil.which(relay_command) or Path(relay_command).exists())):
            LOG.warning("NeMo Relay command %r is not available yet", self.config.nemo_relay.command)
        if self.config.nemo_relay.enabled:
            self._warn_if_codex_relay_unsupported()

    def _warn_if_codex_relay_unsupported(self) -> None:
        try:
            proc = subprocess.run(
                [*self.config.command_parts, "--version"],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except Exception as exc:
            LOG.warning("Could not check Codex version for NeMo Relay hook support: %s", exc)
            return
        version = _parse_codex_version(f"{proc.stdout} {proc.stderr}")
        if version and version < MIN_CODEX_RELAY_VERSION:
            LOG.warning(
                "Codex %s is older than the NeMo Relay hook-capture minimum %s; "
                "Relay may produce empty ATOF/ATIF artifacts.",
                _format_version(version),
                _format_version(MIN_CODEX_RELAY_VERSION),
            )
        if (
            version
            and self.config.nemo_relay.bypass_hook_trust
            and version < MIN_CODEX_HOOK_TRUST_BYPASS_VERSION
        ):
            LOG.warning(
                "Codex %s is older than the hook-trust bypass minimum %s; "
                "disable nemo_relay.bypass_hook_trust or use a newer Codex CLI.",
                _format_version(version),
                _format_version(MIN_CODEX_HOOK_TRUST_BYPASS_VERSION),
            )

    def _resolve_model_base_url(self) -> Optional[str]:
        if self.config.model_server:
            cfg = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            return f"{self.server_client._build_server_base_url(cfg)}/v1"
        return self.config.openai_base_url

    @staticmethod
    def _candidate_instance_ids(instance_id: str) -> list[str]:
        candidates = [instance_id]
        for replacement in ("_1776_", "_s_"):
            replaced = instance_id.replace("__", replacement)
            candidates.extend([replaced, replaced.lower()])
        return list(dict.fromkeys(candidates))

    def _resolve_container_path(self, instance_id: str) -> Path:
        if not self.config.container_formatter:
            raise ValueError("container_formatter is required for SWE-bench workspace materialization")

        tried: list[Path] = []
        for candidate in self._candidate_instance_ids(instance_id):
            path = Path(self.config.container_formatter.format(instance_id=candidate)).expanduser()
            tried.append(path)
            if path.exists():
                return path

        raise FileNotFoundError(
            f"No SWE-bench container found for {instance_id}. Tried: {', '.join(str(p) for p in tried)}"
        )

    async def _run_shell(self, command: str, timeout: int) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"Timed out after {timeout}s: {command}") from None

        if proc.returncode:
            raise RuntimeError(
                f"Command failed with code {proc.returncode}: {command}\n{stderr.decode(errors='replace')[:2000]}"
            )
        return stdout.decode(errors="replace")

    async def _materialize_swebench_workspace(self, metadata: dict[str, Any]) -> Optional[str]:
        if self.config.work_dir or not self.config.container_formatter:
            return self.config.work_dir

        instance_id = metadata.get("instance_id")
        if not instance_id:
            return None

        container = self._resolve_container_path(instance_id)
        base_commit = metadata.get("base_commit")
        workspace_root = Path(self.config.workspace_root).expanduser()
        work_dir = workspace_root / f"{instance_id}_{uuid4().hex[:8]}" / "testbed"
        work_dir.mkdir(parents=True, exist_ok=True)

        copy_cmd = (
            f"{shlex.quote(self.config.apptainer_command)} exec {shlex.quote(str(container))} "
            "bash -lc 'cd /testbed && tar cf - .' "
            f"| tar -C {shlex.quote(str(work_dir))} -xf -"
        )
        await self._run_shell(copy_cmd, timeout=self.config.setup_timeout)

        if base_commit:
            await self._run_shell(
                f"git -C {shlex.quote(str(work_dir))} checkout -f {shlex.quote(str(base_commit))}",
                timeout=120,
            )
            await self._run_shell(f"git -C {shlex.quote(str(work_dir))} clean -fd", timeout=120)

        return str(work_dir)

    async def _collect_patch(self, work_dir: Optional[str]) -> str:
        if not work_dir:
            return ""
        return await self._run_shell(f"git -C {shlex.quote(work_dir)} diff --binary", timeout=120)

    def _resolve_swebench_setup_dir(self) -> Path:
        if self.config.swebench_setup_dir:
            setup_dir = Path(self.config.swebench_setup_dir).expanduser()
            if not setup_dir.is_absolute():
                setup_dir = Path.cwd() / setup_dir
            return setup_dir

        return Path(__file__).resolve().parents[1] / "swe_agents" / "swe_swebench_setup"

    @staticmethod
    def _load_instance_dict(metadata: dict[str, Any]) -> dict[str, Any]:
        raw = metadata.get("instance_dict")
        if raw:
            instance = json.loads(raw) if isinstance(raw, str) else dict(raw)
        else:
            instance = {
                key: metadata[key]
                for key in ("repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement")
                if key in metadata
            }
        if "repo" in instance and "repo_name" not in instance:
            instance["repo_name"] = instance["repo"]
        return instance

    async def _verify_swebench_patch(self, metadata: dict[str, Any], patch: str) -> dict[str, Any]:
        instance_id = str(metadata.get("instance_id") or "")
        if not instance_id:
            return {"swebench_resolved": False, "swebench_error": "missing instance_id"}
        if not patch.strip():
            return {"swebench_resolved": False, "swebench_error": "empty patch"}

        setup_dir = self._resolve_swebench_setup_dir()
        swebench_dir = setup_dir / "SWE-bench"
        python_bin = swebench_dir / "venv" / "bin" / "python"
        if not python_bin.exists():
            raise FileNotFoundError(f"SWE-bench setup is missing: {python_bin}")

        container = self._resolve_container_path(instance_id)
        result_root = Path(self.config.swebench_results_root).expanduser()
        if not result_root.is_absolute():
            result_root = Path.cwd() / result_root
        result_dir = result_root / f"{instance_id}_{uuid4().hex[:8]}"
        result_dir.mkdir(parents=True, exist_ok=True)

        instance = self._load_instance_dict(metadata)
        instance.setdefault("instance_id", instance_id)
        dataset_path = result_dir / "data.jsonl"
        prediction_path = result_dir / "output_for_eval.jsonl"
        patch_path = result_dir / "patch.diff"
        dataset_path.write_text(json.dumps(instance) + "\n")
        prediction_path.write_text(
            json.dumps(
                {
                    "model_name_or_path": self.config.swebench_model_name,
                    "instance_id": instance_id,
                    "model_patch": patch if patch.endswith("\n") else f"{patch}\n",
                }
            )
            + "\n"
        )
        patch_path.write_text(patch)

        run_id = f"codex_{uuid4().hex[:8]}"
        split = str(metadata.get("split") or "test")
        setup_dir_q = shlex.quote(str(setup_dir))
        command = (
            f"{shlex.quote(self.config.apptainer_command)} exec --writable-tmpfs --cleanenv --pid "
            "--no-mount home,tmp,bind-paths "
            f"--mount type=bind,src={shlex.quote(str(result_dir))},dst=/trajectories_mount "
            f"--mount type=bind,src={shlex.quote(str(result_dir))},dst=/root/dataset "
            f"--mount type=bind,src={setup_dir_q},dst=/swebench_setup "
            f"--mount type=bind,src={setup_dir_q},dst={setup_dir_q} "
            f"{shlex.quote(str(container))} bash -lc "
            + shlex.quote(
                "cd /swebench_setup/SWE-bench && "
                f"export UV_INSTALL_DIR={shlex.quote(str(setup_dir / 'uv'))} && "
                f"export UV_PYTHON_INSTALL_DIR={shlex.quote(str(setup_dir / 'python'))} && "
                f"export PATH={shlex.quote(str(setup_dir / 'uv' / 'bin'))}:$PATH && "
                f"env -u VIRTUAL_ENV {shlex.quote(str(python_bin))} "
                "-m swebench.harness.run_local_evaluation "
                "--predictions_path /trajectories_mount/output_for_eval.jsonl "
                f"--instance_ids {shlex.quote(instance_id)} "
                f"--timeout {self.config.swebench_verifier_timeout} "
                "--dataset_name /root/dataset/data.jsonl "
                f"--split {shlex.quote(split)} "
                f"--run_id {shlex.quote(run_id)}"
            )
        )
        await self._run_shell(command, timeout=self.config.swebench_verifier_timeout + 180)

        source_summary = swebench_dir / f"{self.config.swebench_model_name}.{run_id}.json"
        source_instance_dir = (
            swebench_dir / "logs" / "run_evaluation" / run_id / self.config.swebench_model_name / instance_id
        )
        source_report = source_instance_dir / "report.json"

        summary_path = result_dir / source_summary.name
        report_dir = result_dir / "logs" / instance_id
        report_path = report_dir / "report.json"
        if source_summary.exists():
            shutil.copy2(source_summary, summary_path)
        if source_instance_dir.exists():
            shutil.copytree(source_instance_dir, report_dir, dirs_exist_ok=True)

        report = json.loads(source_report.read_text() if source_report.exists() else report_path.read_text())
        resolved = bool(report.get(instance_id, {}).get("resolved", False))
        return {
            "swebench_resolved": resolved,
            "swebench_report_path": str(report_path),
            "swebench_summary_path": str(summary_path),
            "swebench_output_dir": str(result_dir),
        }

    def _resolve_nemo_relay_output_dir(self, work_dir: Optional[str]) -> Optional[Path]:
        if not self.config.nemo_relay.enabled:
            return None

        if work_dir:
            run_name = Path(work_dir).parent.name
        else:
            run_name = f"codex_{uuid4().hex[:8]}"

        if self.config.nemo_relay.output_dir:
            output_root = Path(self.config.nemo_relay.output_dir).expanduser()
            if not output_root.is_absolute():
                output_root = Path.cwd() / output_root
            relay_dir = output_root / run_name
        elif work_dir:
            relay_dir = Path(work_dir).parent / "nemo-relay"
        else:
            relay_dir = Path(self.config.workspace_root).expanduser() / "nemo-relay" / run_name
            if not relay_dir.is_absolute():
                relay_dir = Path.cwd() / relay_dir

        relay_dir.mkdir(parents=True, exist_ok=True)
        return relay_dir

    def _build_nemo_relay_plugin_config(
        self,
        *,
        relay_dir: Path,
        model_name: str,
        metadata: dict[str, Any],
    ) -> str:
        config = self.config.nemo_relay
        headers = dict(config.openinference.headers)
        if config.openinference.project_name:
            headers["x-project-name"] = config.openinference.project_name

        observability: dict[str, Any] = {
            "atof": {
                "enabled": True,
                "output_directory": str(relay_dir),
                "filename": config.atof_filename,
                "mode": config.mode,
            },
            "atif": {
                "enabled": True,
                "agent_name": config.agent_name,
                "agent_version": config.agent_version,
                "model_name": model_name,
                "output_directory": str(relay_dir),
                "filename_template": config.atif_filename_template,
                "extra": {
                    "gym_agent": "codex_agent",
                    "instance_id": str(metadata.get("instance_id") or ""),
                    "dataset_name": str(metadata.get("dataset_name") or ""),
                },
            },
        }

        if config.openinference.enabled:
            observability["openinference"] = {
                "enabled": True,
                "transport": config.openinference.transport,
                "endpoint": config.openinference.endpoint,
                "headers": headers,
                "resource_attributes": config.openinference.resource_attributes,
                "service_name": config.openinference.service_name,
                "service_namespace": config.openinference.service_namespace,
                "service_version": config.openinference.service_version,
                "instrumentation_scope": config.openinference.instrumentation_scope,
                "timeout_millis": config.openinference.timeout_millis,
            }

        return json.dumps(
            {
                "version": 1,
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": observability,
                    }
                ],
            }
        )

    def _write_nemo_relay_run_config(self, relay_dir: Path) -> Path:
        config_path = relay_dir / "nemo-relay.config.toml"
        config_path.write_text(
            "[agents.codex]\n"
            f"command = {_toml_string(self.config.command)}\n",
            encoding="utf-8",
        )
        return config_path

    def _build_nemo_relay_command(
        self,
        *,
        codex_cmd: list[str],
        relay_dir: Path,
        model_name: str,
        metadata: dict[str, Any],
        base_url: Optional[str],
    ) -> list[str]:
        relay_config_path = self._write_nemo_relay_run_config(relay_dir)
        codex_args = codex_cmd[len(self.config.command_parts) :]
        cmd = [
            *shlex.split(self.config.nemo_relay.command),
            "run",
            "--agent",
            "codex",
            "--config",
            str(relay_config_path),
        ]
        if base_url:
            cmd.extend(["--openai-base-url", base_url])
        cmd.extend(
            [
                "--plugin-config",
                self._build_nemo_relay_plugin_config(
                    relay_dir=relay_dir,
                    model_name=model_name,
                    metadata=metadata,
                ),
                "--",
                *codex_args,
            ]
        )
        return cmd

    def _collect_nemo_relay_metadata(self, relay_dir: Path) -> dict[str, str]:
        config = self.config.nemo_relay
        atif_pattern = config.atif_filename_template.replace("{session_id}", "*")
        atif_paths = sorted(str(path) for path in relay_dir.glob(atif_pattern))
        atof_path = relay_dir / config.atof_filename
        metadata = {
            "nemo_relay_output_dir": str(relay_dir),
            "nemo_relay_atof_path": str(atof_path) if atof_path.exists() else "",
            "nemo_relay_atif_path": atif_paths[0] if atif_paths else "",
            "nemo_relay_atif_paths": json.dumps(atif_paths),
        }
        pty_log_path = relay_dir / "codex.pty.log"
        if pty_log_path.exists():
            metadata["nemo_relay_pty_log_path"] = str(pty_log_path)
        if atof_path.exists() and atof_path.stat().st_size == 0:
            metadata["nemo_relay_warning"] = (
                "ATOF was created but empty. For Codex, confirm codex-cli >= "
                f"{_format_version(MIN_CODEX_RELAY_VERSION)} and that Codex hooks are active."
            )
        if config.openinference.enabled:
            metadata.update(
                {
                    "nemo_relay_openinference_enabled": "true",
                    "nemo_relay_openinference_endpoint": config.openinference.endpoint or "",
                    "nemo_relay_openinference_project": config.openinference.project_name or "",
                }
            )
        return metadata

    def _atof_has_turn_end(self, relay_dir: Path) -> bool:
        atof_path = relay_dir / self.config.nemo_relay.atof_filename
        if not atof_path.exists() or atof_path.stat().st_size == 0:
            return False
        try:
            for raw_line in reversed(atof_path.read_text(encoding="utf-8").splitlines()):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("name") == "codex-turn" and event.get("scope_category") == "end":
                    return True
        except OSError:
            return False
        return False

    def _build_codex_command(self, prompt: str, work_dir: Optional[str]) -> list[str]:
        cmd = [
            *self.config.command_parts,
            "exec",
            "--json",
            "--sandbox",
            self.config.sandbox,
        ]
        if self.config.approval_policy:
            cmd.extend(["--config", f'approval_policy="{self.config.approval_policy}"'])
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        if self.config.profile:
            cmd.extend(["--profile", self.config.profile])
        if work_dir:
            cmd.extend(["--cd", work_dir])
        if self.config.ephemeral:
            cmd.append("--ephemeral")
        if self.config.ignore_user_config:
            cmd.append("--ignore-user-config")
        if self.config.ignore_rules:
            cmd.append("--ignore-rules")
        if self.config.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")
        if self.config.nemo_relay.enabled and self.config.nemo_relay.bypass_hook_trust:
            cmd.append("--dangerously-bypass-hook-trust")
        for override in self.config.config_overrides:
            cmd.extend(["--config", override])
        cmd.extend(self.config.extra_args)
        cmd.extend(["--", prompt])
        return cmd

    def _build_codex_interactive_command(self, work_dir: Optional[str], prompt: Optional[str] = None) -> list[str]:
        cmd = [
            *self.config.command_parts,
            "--sandbox",
            self.config.sandbox,
        ]
        if self.config.approval_policy:
            cmd.extend(["--ask-for-approval", self.config.approval_policy])
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        if self.config.profile:
            cmd.extend(["--profile", self.config.profile])
        if work_dir:
            cmd.extend(["--cd", work_dir])
        if self.config.nemo_relay.bypass_hook_trust:
            cmd.append("--dangerously-bypass-hook-trust")
        for override in self.config.config_overrides:
            cmd.extend(["--config", override])
        cmd.extend(self.config.extra_args)
        if "--no-alt-screen" not in cmd:
            cmd.append("--no-alt-screen")
        if prompt:
            cmd.extend(["--", prompt])
        return cmd

    async def _run_codex_exec_command(
        self,
        *,
        cmd: list[str],
        env: dict[str, str],
    ) -> tuple[str, str, int | None]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            LOG.warning("codex timed out after %ds", self.config.timeout)
            return "", "timeout", None

        if proc.returncode not in (0, None):
            LOG.warning("codex exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:2000])

        return stdout.decode(errors="replace"), stderr.decode(errors="replace"), proc.returncode

    async def _run_codex_relay_pty(
        self,
        *,
        cmd: list[str],
        prompt: Optional[str],
        env: dict[str, str],
        relay_dir: Path,
    ) -> tuple[str, str, int | None]:
        master_fd, slave_fd = os.openpty()
        os.set_blocking(master_fd, False)
        chunks: list[bytes] = []
        pty_log_path = relay_dir / "codex.pty.log"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
        )
        os.close(slave_fd)

        async def drain_pty() -> None:
            while True:
                try:
                    chunk = os.read(master_fd, 4096)
                except BlockingIOError:
                    if proc.returncode is not None:
                        break
                    await asyncio.sleep(0.1)
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
                with pty_log_path.open("ab") as pty_log:
                    pty_log.write(chunk)

        reader_task = asyncio.create_task(drain_pty())
        status = "relay_turn_completed"
        deadline = asyncio.get_running_loop().time() + self.config.timeout
        try:
            startup_deadline = (
                asyncio.get_running_loop().time()
                + self.config.nemo_relay.interactive_startup_timeout
            )
            while asyncio.get_running_loop().time() < startup_deadline:
                terminal_text = b"".join(chunks).decode(errors="replace")
                if "Do you trust" in terminal_text or "Press enter to continue" in terminal_text:
                    os.write(master_fd, b"\r")
                    await asyncio.sleep(0.5)
                    break
                if self._atof_has_turn_end(relay_dir) or proc.returncode is not None:
                    break
                if "›" in terminal_text and prompt is None:
                    break
                await asyncio.sleep(0.2)

            if prompt:
                await asyncio.sleep(self.config.nemo_relay.interactive_prompt_delay)
                if proc.returncode is not None:
                    status = "process_exited"
                else:
                    while asyncio.get_running_loop().time() < startup_deadline:
                        terminal_text = b"".join(chunks).decode(errors="replace")
                        if "Continue anyway?" in terminal_text:
                            os.write(master_fd, b"y\r")
                        if "›" in terminal_text:
                            break
                        if proc.returncode is not None:
                            status = "process_exited"
                            break
                        await asyncio.sleep(0.2)
                    try:
                        if proc.returncode is None:
                            os.write(master_fd, b"\x1b[200~" + prompt.encode("utf-8") + b"\x1b[201~\r")
                            await asyncio.sleep(0.2)
                            os.write(master_fd, b"\n")
                    except OSError:
                        status = "prompt_write_failed"
            while True:
                if self._atof_has_turn_end(relay_dir):
                    break
                terminal_text = b"".join(chunks).decode(errors="replace")
                if "Quota exceeded" in terminal_text:
                    status = "quota_exceeded"
                    break
                if proc.returncode is not None:
                    status = "process_exited"
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    status = "timeout"
                    LOG.warning("codex relay turn timed out after %ds", self.config.timeout)
                    break
                await asyncio.sleep(0.5)
        finally:
            if proc.returncode is None:
                try:
                    os.write(master_fd, b"\x03")
                except OSError:
                    pass
                try:
                    await asyncio.wait_for(
                        proc.wait(),
                        timeout=self.config.nemo_relay.interactive_shutdown_timeout,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    status = f"{status}; forced_shutdown"
            try:
                os.close(master_fd)
            except OSError:
                pass
            await asyncio.gather(reader_task, return_exceptions=True)

        return b"".join(chunks).decode(errors="replace"), status, proc.returncode

    async def _run_codex(
        self,
        instruction: str,
        system_prompt: Optional[str],
        work_dir: Optional[str],
        relay_dir: Optional[Path],
        metadata: dict[str, Any],
    ) -> tuple[str, str, str, int | None]:
        prompt = instruction
        if system_prompt:
            prompt = f"{system_prompt}\n\n{instruction}"

        env = {**os.environ}
        model_name = self.config.model or "codex-default"
        base_url = self._resolve_model_base_url()
        if self.config.openai_api_key:
            env["OPENAI_API_KEY"] = self.config.openai_api_key
        if relay_dir:
            if not env.get("TERM") or env.get("TERM") == "dumb":
                env["TERM"] = "xterm-256color"
            cmd = self._build_codex_interactive_command(work_dir, prompt=prompt)
            cmd = self._build_nemo_relay_command(
                codex_cmd=cmd,
                relay_dir=relay_dir,
                model_name=model_name,
                metadata=metadata,
                base_url=base_url,
            )
        else:
            cmd = self._build_codex_command(prompt, work_dir)
            if base_url:
                env["OPENAI_BASE_URL"] = base_url
        if relay_dir:
            stdout, stderr, returncode = await self._run_codex_relay_pty(
                cmd=cmd,
                prompt=None,
                env=env,
                relay_dir=relay_dir,
            )
        elif base_url:
            stdout, stderr, returncode = await self._run_codex_exec_command(
                cmd=cmd,
                env=env,
            )
        else:
            stdout, stderr, returncode = await self._run_codex_exec_command(
                cmd=cmd,
                env=env,
            )

        return (
            stdout,
            model_name,
            stderr,
            returncode,
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        metadata = getattr(body, "metadata", None) or {}

        work_dir = await self._materialize_swebench_workspace(metadata)
        relay_dir = self._resolve_nemo_relay_output_dir(work_dir)
        stdout, model_name, stderr, returncode = await self._run_codex(
            user_message,
            system_prompt,
            work_dir,
            relay_dir,
            dict(metadata),
        )
        output_items, usage = parse_codex_jsonl(stdout)
        patch = await self._collect_patch(work_dir)
        relay_metadata: dict[str, str] = {}

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("codex produced no assistant message; padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(type="output_text", text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        if relay_dir:
            relay_metadata = self._collect_nemo_relay_metadata(relay_dir)

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
            metadata={
                "work_dir": work_dir or "",
                "patch": patch,
                "patch_exists": str(bool(patch.strip())).lower(),
                "instance_id": str(metadata.get("instance_id") or ""),
                "codex_returncode": "" if returncode is None else str(returncode),
                "codex_stderr": stderr[-4000:],
                "codex_stdout_tail": stdout[-4000:],
            }
            | relay_metadata,
        )

    async def run(self, request: Request, body: CodexAgentRunRequest) -> CodexAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            if self.config.resources_server:
                seed_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/seed_session",
                    json=body.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(seed_resp)
                cookies = seed_resp.cookies

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)

            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            if not self.config.resources_server:
                reward = 0.0
                verify_fields: dict[str, Any] = {}
                metadata = body.responses_create_params.metadata or {}
                response_metadata = gym_resp.metadata or {}
                if self.config.verify_swebench:
                    try:
                        verify_fields = await self._verify_swebench_patch(
                            dict(metadata),
                            str(response_metadata.get("patch") or ""),
                        )
                        reward = 1.0 if verify_fields.get("swebench_resolved") else 0.0
                    except Exception as exc:
                        LOG.exception("SWE-bench verification failed")
                        verify_fields = {
                            "swebench_resolved": False,
                            "swebench_error": str(exc),
                        }

                return CodexAgentVerifyResponse(
                    responses_create_params=body.responses_create_params,
                    response=gym_resp,
                    reward=reward,
                    turns_used=turns,
                    finished_naturally=naturally,
                    **verify_fields,
                )

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            return CodexAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    CodexAgent.run_webserver()
