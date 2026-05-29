# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from responses_api_agents.claude_code_agent.setup_claude_code import ensure_claude_code
from responses_api_agents.opencode_agent.app import OpenCodeAgent as _SWEBenchHelpers


LOG = logging.getLogger(__name__)


def _extract_text(content: list[Any]) -> str:
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _extract_thinking(content: list[Any]) -> str:
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("thinking", "reasoning"):
            parts.append(b.get("thinking") or b.get("text") or "")
    return "\n".join(p for p in parts if p)


def parse_stream_json(stdout: str) -> tuple[list[Any], dict]:
    """Convert claude -p --output-format=stream-json stdout into (output_items, usage)."""
    raw_events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    output_items: list[Any] = []
    pending_calls: dict[str, dict] = {}
    buffered_think: str | None = None
    total_input = 0
    total_output = 0

    for event in raw_events:
        etype = event.get("type")

        if etype == "result":
            usage = event.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)

        elif etype == "assistant":
            message = event.get("message", {})
            content = message.get("content") or []
            usage = message.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)

            if not isinstance(content, list):
                content = []

            think = _extract_thinking(content)
            if think:
                buffered_think = (buffered_think + "\n" + think) if buffered_think else think

            text = _extract_text(content)
            if text:
                if buffered_think:
                    text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                    buffered_think = None
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                input_data = block.get("input") or {}
                arguments = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
                pending_calls[call_id] = {"name": block.get("name", ""), "call_id": call_id, "arguments": arguments}

        elif etype == "user":
            message = event.get("message", {})
            content = message.get("content") or []
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id", "")
                call_info = pending_calls.pop(tool_id, None)
                if call_info:
                    output_items.append(
                        NeMoGymResponseFunctionToolCall(
                            arguments=call_info["arguments"],
                            call_id=tool_id,
                            name=call_info["name"],
                            type="function_call",
                            id=tool_id,
                            status="completed",
                        )
                    )
                result_content = block.get("content") or ""
                if isinstance(result_content, list):
                    result_text = _extract_text(result_content)
                else:
                    result_text = str(result_content)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_id,
                        output=result_text,
                        status="completed",
                    )
                )

    return output_items, {"input_tokens": total_input, "output_tokens": total_output}


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
    items = list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            system_message = content or ""
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            user_message = content or ""
            break

    return user_message, system_message


class ClaudeCodeNemoFlowConfig(BaseModel):
    enabled: bool = False
    command: str = "nemo-relay"
    agent: str = "claude-code"
    output_dir: Optional[str] = None
    atof_filename: str = "claude-code.atof.jsonl"
    atif_filename_template: str = "claude-code-{session_id}.atif.json"
    agent_name: str = "claude-code"
    mode: str = "overwrite"

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class ClaudeCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: Optional[ResourcesServerRef] = None
    # When model_server is set, ANTHROPIC_BASE_URL is resolved from the Gym model
    # server's URL (requires the server to expose POST /v1/messages. None is pushed yet).
    # When None, anthropic_base_url is used directly.
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    command: str = "claude"
    model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""  # pragma: allowlist secret
    anthropic_base_url: Optional[str] = None
    max_turns: int = 30
    timeout: int = 300
    system_prompt: Optional[str] = None
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None
    claude_code_version: Optional[str] = None
    thinking: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    bare: bool = True
    work_dir: Optional[str] = None
    workspace_root: str = "outputs/claude_code_agent/workspaces"
    container_formatter: Optional[str] = None
    apptainer_command: str = "apptainer"
    setup_timeout: int = 900
    nemo_flow: ClaudeCodeNemoFlowConfig = Field(default_factory=ClaudeCodeNemoFlowConfig)
    verify_swebench: bool = False
    swebench_setup_dir: Optional[str] = None
    swebench_results_root: str = "outputs/claude_code_agent/swebench-verifier"
    swebench_verifier_timeout: int = 1200
    swebench_model_name: str = "claude_code_agent"

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class ClaudeCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class ClaudeCodeAgent(_SWEBenchHelpers, SimpleResponsesAPIAgent):
    config: ClaudeCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_claude_code(self.config.claude_code_version)
        try:
            command = self.config.command_parts[0] if self.config.command_parts else "claude"
            ver = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
            LOG.warning("claude-code version: %s", ver or "(unknown)")
        except Exception as exc:
            LOG.warning("could not determine claude-code version: %s", exc)

    def _resolve_base_url(self) -> str:
        if self.config.model_server:
            cfg = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            return self.server_client._build_server_base_url(cfg)
        return self.config.anthropic_base_url or ""

    def _resolve_nemo_flow_output_dir(self, work_dir: Optional[str]) -> Optional[Path]:
        if not self.config.nemo_flow.enabled:
            return None

        if work_dir:
            run_name = Path(work_dir).parent.name
        else:
            run_name = f"claude_code_{uuid4().hex[:8]}"

        if self.config.nemo_flow.output_dir:
            output_root = Path(self.config.nemo_flow.output_dir).expanduser()
            if not output_root.is_absolute():
                output_root = Path.cwd() / output_root
            output_dir = output_root / run_name
        elif work_dir:
            output_dir = Path(work_dir).parent / "nemo-relay"
        else:
            output_dir = Path(self.config.workspace_root).expanduser() / "nemo-relay" / run_name
            if not output_dir.is_absolute():
                output_dir = Path.cwd() / output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _nemo_flow_plugin_config(self, output_dir: Path) -> str:
        relay = self.config.nemo_flow
        return json.dumps(
            {
                "version": 1,
                "components": [
                    {
                        "kind": "observability",
                        "enabled": True,
                        "config": {
                            "version": 1,
                            "atof": {
                                "enabled": True,
                                "output_directory": str(output_dir),
                                "filename": relay.atof_filename,
                                "mode": relay.mode,
                            },
                            "atif": {
                                "enabled": True,
                                "agent_name": relay.agent_name,
                                "output_directory": str(output_dir),
                                "filename_template": relay.atif_filename_template,
                            },
                            "opentelemetry": {"enabled": False},
                            "openinference": {"enabled": False},
                        },
                    }
                ],
            }
        )

    def _nemo_flow_metadata(self, output_dir: Optional[Path]) -> dict[str, str]:
        if not output_dir:
            return {}

        relay = self.config.nemo_flow
        atif_glob = relay.atif_filename_template.replace("{session_id}", "*")
        atif_paths = sorted(str(path) for path in output_dir.glob(atif_glob))
        atof_path = output_dir / relay.atof_filename
        return {
            "nemo_relay_output_dir": str(output_dir),
            "nemo_flow_output_dir": str(output_dir),
            "nemo_relay_atof_path": str(atof_path) if atof_path.exists() else "",
            "nemo_flow_atof_path": str(atof_path) if atof_path.exists() else "",
            "nemo_relay_atif_paths": json.dumps(atif_paths),
            "nemo_flow_atif_paths": json.dumps(atif_paths),
        }

    def _wrap_with_nemo_flow(self, cmd: list[str], output_dir: Optional[Path], base_url: str) -> list[str]:
        if not output_dir:
            return cmd

        relay = self.config.nemo_flow
        command_parts = self.config.command_parts
        agent_args = cmd[len(command_parts) :] if command_parts and cmd[: len(command_parts)] == command_parts else cmd
        wrapped = [
            *relay.command_parts,
            "run",
            "--agent",
            relay.agent,
            "--plugin-config",
            self._nemo_flow_plugin_config(output_dir),
        ]
        if base_url:
            wrapped.extend(["--anthropic-base-url", base_url])
        wrapped.extend(["--", *agent_args])
        return wrapped

    async def _run_claude_code(
        self,
        instruction: str,
        system_prompt: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> tuple[str, str, dict[str, str]]:
        """Run claude -p --output-format=stream-json and return stdout plus artifact metadata."""
        base_url = self._resolve_base_url()
        # Keep full model name for local/custom endpoints; strip provider prefix for real Anthropic API.
        model = self.config.model if base_url else self.config.model.split("/")[-1]
        api_key = self.config.anthropic_api_key

        claude_config_dir = Path.home() / ".claude_code_agent" / uuid4().hex
        claude_config_dir.mkdir(parents=True)
        try:
            (claude_config_dir / "settings.json").write_text(
                json.dumps(
                    {
                        "env": {
                            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                            "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                        }
                    }
                )
            )

            relay_dir = self._resolve_nemo_flow_output_dir(work_dir)
            env = {
                **os.environ,
                "ANTHROPIC_API_KEY": api_key,  # pragma: allowlist secret
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "CLAUDE_CODE_SUBAGENT_MODEL": model,
                "IS_SANDBOX": "1",
                "CLAUDE_CONFIG_DIR": str(claude_config_dir),
            }
            if base_url and not relay_dir:
                env["ANTHROPIC_BASE_URL"] = base_url
                env["ANTHROPIC_AUTH_TOKEN"] = api_key or "local"

            cmd = [
                *self.config.command_parts,
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--max-turns",
                str(self.config.max_turns),
                "--model",
                model,
            ]
            if self.config.bare and not relay_dir:
                cmd.append("--bare")
            elif self.config.bare and relay_dir:
                LOG.warning("Claude Code --bare disabled while nemo_flow is enabled so Relay hooks can load")
            if system_prompt:
                cmd += ["--append-system-prompt", system_prompt]
            if self.config.allowed_tools:
                cmd += ["--allowedTools", self.config.allowed_tools]
            if self.config.disallowed_tools:
                cmd += ["--disallowedTools", self.config.disallowed_tools]
            if self.config.thinking:
                cmd += ["--thinking", self.config.thinking]
            if self.config.max_thinking_tokens is not None:
                cmd += ["--max-thinking-tokens", str(self.config.max_thinking_tokens)]
            cmd += ["--", instruction]
            cmd = self._wrap_with_nemo_flow(cmd, relay_dir, base_url)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                LOG.warning("claude-code timed out after %ds", self.config.timeout)
                return "", model, self._nemo_flow_metadata(relay_dir)

            if proc.returncode not in (0, None):
                LOG.warning("claude-code exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            LOG.debug("claude-code stdout (%d chars): %s", len(stdout), stdout[:2000].decode(errors="replace"))
            return stdout.decode(errors="replace"), model, self._nemo_flow_metadata(relay_dir)
        finally:
            shutil.rmtree(claude_config_dir, ignore_errors=True)

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
        stdout, model_name, artifact_metadata = await self._run_claude_code(
            user_message,
            system_prompt=system_prompt,
            work_dir=work_dir,
        )
        output_items, usage = parse_stream_json(stdout)
        patch = await self._collect_patch(work_dir)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("claude-code produced no assistant message; padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

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
            }
            | artifact_metadata,
        )

    async def run(self, request: Request, body: ClaudeCodeAgentRunRequest) -> ClaudeCodeAgentVerifyResponse:
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

                return ClaudeCodeAgentVerifyResponse(
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

            return ClaudeCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    ClaudeCodeAgent.run_webserver()
