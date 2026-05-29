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
import sys
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


def _iter_codex_json_events(stdout: str):
    for line_no, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            yield line_no, {"type": "unparsed", "raw": raw_line}
            continue
        if isinstance(event, dict):
            yield line_no, event
        else:
            yield line_no, {"type": "non_object", "value": event}


class CodexNemoRelayConfig(BaseModel):
    enabled: bool = False
    python_path: Optional[str] = None
    output_dir: Optional[str] = None
    atof_filename: str = "codex.atof.jsonl"
    atif_filename_template: str = "codex-{session_id}.atif.json"
    agent_name: str = "codex"
    agent_version: str = "0.1.0"
    mode: str = "overwrite"
    include_raw_events: bool = True


def _json_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    if value is None:
        return {}
    return value


def _model_output_type(item: Any) -> str:
    return str(getattr(item, "type", "") or "")


def _message_text(item: Any) -> str:
    return _content_to_text(getattr(item, "content", None))


class _CodexNemoRelayCapture:
    def __init__(
        self,
        *,
        relay_dir: Path,
        config: CodexNemoRelayConfig,
        model_name: str,
        input_text: str,
        metadata: dict[str, Any],
    ) -> None:
        if config.python_path:
            python_path = str(Path(config.python_path).expanduser().resolve())
            if python_path not in sys.path:
                sys.path.insert(0, python_path)

        try:
            from nemo_relay import (  # noqa: PLC0415
                AtifExporter,
                AtofExporter,
                AtofExporterConfig,
                AtofExporterMode,
                LLMRequest,
                ScopeType,
                llm,
                scope,
                subscribers,
                tools,
            )
        except ImportError:
            from nemo_flow import (  # type: ignore[no-redef]  # noqa: PLC0415
                AtifExporter,
                AtofExporter,
                AtofExporterConfig,
                AtofExporterMode,
                LLMRequest,
                ScopeType,
                llm,
                scope,
                subscribers,
                tools,
            )

        self._llm = llm
        self._scope = scope
        self._subscribers = subscribers
        self._tools = tools
        self._LLMRequest = LLMRequest
        self.relay_dir = relay_dir
        self.config = config
        self.model_name = model_name
        self.input_text = input_text
        self.session_id = f"codex-{uuid4().hex[:8]}"
        self.atof_path = relay_dir / config.atof_filename
        self.atif_path = relay_dir / config.atif_filename_template.format(session_id=self.session_id)
        self._tool_handles: dict[str, Any] = {}

        atof_config = AtofExporterConfig()
        atof_config.output_directory = str(relay_dir)
        atof_config.filename = config.atof_filename
        atof_config.mode = AtofExporterMode.Overwrite if config.mode == "overwrite" else AtofExporterMode.Append
        self._atof_exporter = AtofExporter(atof_config)
        self._atof_subscriber = f"codex_atof_{uuid4().hex}"
        self._atof_exporter.register(self._atof_subscriber)

        self._atif_exporter = AtifExporter(
            self.session_id,
            config.agent_name,
            config.agent_version,
            model_name=model_name,
            extra={
                "gym_agent": "codex_agent",
                "instance_id": str(metadata.get("instance_id") or ""),
            },
        )
        self._atif_subscriber = f"codex_atif_{uuid4().hex}"
        self._atif_exporter.register(self._atif_subscriber)

        self._agent_handle = scope.push(
            config.agent_name,
            ScopeType.Agent,
            input={
                "prompt": input_text,
                "metadata": {
                    "instance_id": str(metadata.get("instance_id") or ""),
                    "dataset_name": str(metadata.get("dataset_name") or ""),
                },
            },
        )

    def record_raw_events(self, stdout: str) -> None:
        if not self.config.include_raw_events:
            return
        for line_no, event in _iter_codex_json_events(stdout):
            self._scope.event(
                "llm.chunk",
                handle=self._agent_handle,
                data={
                    "provider": "codex",
                    "line": line_no,
                    "event": event,
                },
                metadata={"hook_event_name": "codex.json_event"},
            )

    def _llm_call_end(self, messages: list[dict[str, Any]], response: dict[str, Any], index: int) -> None:
        request = self._LLMRequest(
            {},
            {
                "messages": messages,
                "model": self.model_name,
                "codex_projection_index": index,
            },
        )
        handle = self._llm.call(
            "codex_assistant_message",
            request,
            handle=self._agent_handle,
            model_name=self.model_name,
            metadata={"projection": True},
        )
        self._llm.call_end(handle, response)

    def project_output_items(self, output_items: list[Any]) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": self.input_text}]
        projection_index = 0
        for item in output_items:
            item_type = _model_output_type(item)
            if item_type == "message":
                text = _message_text(item)
                response = {
                    "role": "assistant",
                    "content": text,
                    "codex_output_type": item_type,
                }
                self._llm_call_end(list(messages), response, projection_index)
                messages.append({"role": "assistant", "content": text})
                projection_index += 1
            elif item_type == "function_call":
                call_id = str(getattr(item, "call_id", None) or getattr(item, "id", None) or f"call-{uuid4().hex[:8]}")
                name = str(getattr(item, "name", "") or "")
                arguments = _json_arguments(getattr(item, "arguments", None))
                response = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
                self._llm_call_end(list(messages), response, projection_index)
                messages.append({"role": "assistant", "content": "", "tool_calls": response["tool_calls"]})
                self._tool_handles[call_id] = self._tools.call(
                    name,
                    arguments,
                    handle=self._agent_handle,
                    tool_call_id=call_id,
                )
                projection_index += 1
            elif item_type == "function_call_output":
                call_id = str(getattr(item, "call_id", None) or f"call-{uuid4().hex[:8]}")
                output = _content_to_text(getattr(item, "output", None))
                handle = self._tool_handles.pop(call_id, None)
                if handle is None:
                    handle = self._tools.call(
                        "unknown",
                        {},
                        handle=self._agent_handle,
                        tool_call_id=call_id,
                    )
                self._tools.call_end(handle, output)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

    def close(
        self,
        *,
        stdout: str,
        output_items: list[Any],
        output: dict[str, Any],
    ) -> dict[str, str]:
        self.record_raw_events(stdout)
        self.project_output_items(output_items)
        for handle in list(self._tool_handles.values()):
            self._tools.call_end(handle, {"status": "unclosed"})
        self._tool_handles.clear()

        self._scope.pop(self._agent_handle, output=output)
        self.atif_path.write_text(self._atif_exporter.export_json())
        self._atif_exporter.deregister(self._atif_subscriber)
        self._atof_exporter.deregister(self._atof_subscriber)
        self._atof_exporter.force_flush()
        self._atof_exporter.shutdown()
        self._subscribers.deregister(self._atof_subscriber)
        self._subscribers.deregister(self._atif_subscriber)

        return {
            "nemo_relay_output_dir": str(self.relay_dir),
            "nemo_relay_atof_path": str(self.atof_path) if self.atof_path.exists() else "",
            "nemo_relay_atif_path": str(self.atif_path) if self.atif_path.exists() else "",
            "nemo_relay_atif_paths": json.dumps([str(self.atif_path)] if self.atif_path.exists() else []),
        }


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
        for override in self.config.config_overrides:
            cmd.extend(["--config", override])
        cmd.extend(self.config.extra_args)
        cmd.extend(["--", prompt])
        return cmd

    async def _run_codex(
        self, instruction: str, system_prompt: Optional[str], work_dir: Optional[str]
    ) -> tuple[str, str, str, int | None]:
        prompt = instruction
        if system_prompt:
            prompt = f"{system_prompt}\n\n{instruction}"

        cmd = self._build_codex_command(prompt, work_dir)

        env = {**os.environ}
        base_url = self._resolve_model_base_url()
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if self.config.openai_api_key:
            env["OPENAI_API_KEY"] = self.config.openai_api_key

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
            return "", self.config.model or "codex-default", "timeout", None

        if proc.returncode not in (0, None):
            LOG.warning("codex exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:2000])

        return (
            stdout.decode(errors="replace"),
            self.config.model or "codex-default",
            stderr.decode(errors="replace"),
            proc.returncode,
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
        stdout, model_name, stderr, returncode = await self._run_codex(user_message, system_prompt, work_dir)
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
            relay_capture = _CodexNemoRelayCapture(
                relay_dir=relay_dir,
                config=self.config.nemo_relay,
                model_name=model_name,
                input_text=user_message,
                metadata=dict(metadata),
            )
            relay_metadata = relay_capture.close(
                stdout=stdout,
                output_items=output_items,
                output={
                    "patch_exists": bool(patch.strip()),
                    "assistant_messages": sum(1 for item in output_items if getattr(item, "type", None) == "message"),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "codex_returncode": returncode,
                },
            )

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
