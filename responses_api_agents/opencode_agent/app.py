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
import copy
import json
import logging
import os
import shlex
import shutil
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

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
        return "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in content
        )
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


def parse_opencode_jsonl(stdout: str) -> tuple[list[Any], dict[str, int]]:
    """Convert `opencode run --format=json` stdout into Gym response output items."""
    output_items: list[Any] = []
    reasoning_buffer: list[str] = []
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

        etype = event.get("type")
        part = event.get("part") or {}

        if etype == "text" or part.get("type") == "text":
            text = part.get("text") or event.get("text") or ""
            if not text:
                continue
            if reasoning_buffer:
                text = f"<think>\n{'\n\n'.join(reasoning_buffer)}\n</think>\n\n{text}"
                reasoning_buffer.clear()
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        elif etype == "reasoning" or part.get("type") == "reasoning":
            text = part.get("text") or event.get("text") or ""
            if text:
                reasoning_buffer.append(text)

        elif etype == "tool_use" or part.get("type") == "tool":
            state = part.get("state") or {}
            call_id = part.get("callID") or part.get("id") or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=part.get("tool") or part.get("name") or "",
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
            if state.get("output") is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(state.get("output")),
                        status="completed",
                    )
                )

        elif etype == "step_finish":
            finish = part or {}
            tokens = finish.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            output_tokens += int(tokens.get("output") or 0)

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


class OpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: Optional[ResourcesServerRef] = None
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "opencode"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    work_dir: Optional[str] = None
    workspace_root: str = "outputs/opencode_agent/workspaces"
    container_formatter: Optional[str] = None
    apptainer_command: str = "apptainer"
    setup_timeout: int = 900
    timeout: int = 900
    thinking: bool = True
    system_prompt: Optional[str] = None
    extra_args: list[str] = []
    opencode_config: dict[str, Any] = Field(default_factory=dict)
    verify_swebench: bool = False
    swebench_setup_dir: Optional[str] = None
    swebench_results_root: str = "outputs/opencode_agent/swebench-verifier"
    swebench_verifier_timeout: int = 1200
    swebench_model_name: str = "opencode_agent"

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class OpenCodeAgent(SimpleResponsesAPIAgent):
    config: OpenCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("OpenCode command %r is not on PATH yet", self.config.command)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenCodeAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

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

        run_id = f"opencode_{uuid4().hex[:8]}"
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

    def _write_opencode_config(self, work_dir: Optional[str]) -> Optional[str]:
        if not self.config.opencode_config:
            return None

        config_home = Path(work_dir).parent / ".opencode-config" if work_dir else Path(self.config.workspace_root)
        config_dir = config_home / "opencode"
        config_dir.mkdir(parents=True, exist_ok=True)
        config = self._deep_merge({}, copy.deepcopy(self.config.opencode_config))
        (config_dir / "opencode.json").write_text(json.dumps(config, indent=2))
        return str(config_home)

    async def _run_opencode(
        self, instruction: str, system_prompt: Optional[str], work_dir: Optional[str]
    ) -> tuple[str, str]:
        prompt = instruction
        if system_prompt:
            prompt = f"{system_prompt}\n\n{instruction}"

        cmd = [
            *self.config.command_parts,
            "run",
            f"--model={self.config.model}",
            "--format=json",
        ]
        if self.config.thinking:
            cmd.append("--thinking")
        if work_dir:
            cmd.extend(["--dir", work_dir])
        cmd.extend(self.config.extra_args)
        cmd.extend(["--", prompt])

        env = {**os.environ}
        base_url = self._resolve_model_base_url()
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if self.config.openai_api_key:
            env["OPENAI_API_KEY"] = self.config.openai_api_key
        config_home = self._write_opencode_config(work_dir)
        if config_home:
            env["XDG_CONFIG_HOME"] = config_home
        env["OPENCODE_FAKE_VCS"] = "git"

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
            LOG.warning("opencode timed out after %ds", self.config.timeout)
            return "", self.config.model

        if proc.returncode not in (0, None):
            LOG.warning("opencode exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

        return stdout.decode(errors="replace"), self.config.model

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
        stdout, model_name = await self._run_opencode(user_message, system_prompt, work_dir)
        output_items, usage = parse_opencode_jsonl(stdout)
        patch = await self._collect_patch(work_dir)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("opencode produced no assistant message; padding empty output")
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
            },
        )

    async def run(self, request: Request, body: OpenCodeAgentRunRequest) -> OpenCodeAgentVerifyResponse:
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

                return OpenCodeAgentVerifyResponse(
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

            return OpenCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    OpenCodeAgent.run_webserver()
