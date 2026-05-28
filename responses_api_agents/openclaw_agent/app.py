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
from typing import Any, ClassVar, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.opencode_agent.app import OpenCodeAgent as _SWEBenchHelpers
from responses_api_agents.opencode_agent.app import _extract_instruction


LOG = logging.getLogger(__name__)

_NEMO_RELAY_PLUGIN_MANIFEST_ID = "nemo-relay"
_NEMO_RELAY_OPENCLAW_NPM_VERSION = "0.3.0"
_NEMO_RELAY_ATIF_DIR_NAME = "nemo-relay-atif"


def _decode_last_json_dict_suffix(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and not text[start + consumed :].strip():
            return obj
    return None


def _text_from_openclaw_payloads(envelope: dict[str, Any]) -> str:
    payloads = envelope.get("payloads")
    if not isinstance(payloads, list):
        return ""

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in payloads:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if item.get("isReasoning") is True:
            reasoning_parts.append(text.strip())
        else:
            text_parts.append(text.strip())

    assistant_text = "\n\n".join(text_parts)
    if not assistant_text:
        meta = envelope.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("finalAssistantVisibleText"), str):
            assistant_text = meta["finalAssistantVisibleText"].strip()
    if reasoning_parts and assistant_text:
        return f"<think>\n{'\n\n'.join(reasoning_parts)}\n</think>\n\n{assistant_text}"
    return assistant_text


def parse_openclaw_output(stdout: str) -> tuple[list[Any], dict[str, int], dict[str, Any] | None]:
    """Convert `openclaw agent --local --json` stdout into Gym response output items."""
    envelope = _decode_last_json_dict_suffix(stdout)
    if not envelope:
        return [], {"input_tokens": 0, "output_tokens": 0}, None

    text = _text_from_openclaw_payloads(envelope)
    output_items: list[Any] = []
    if text:
        output_items.append(
            NeMoGymResponseOutputMessage(
                id="msg-0",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    agent_meta = (envelope.get("meta") or {}).get("agentMeta") if isinstance(envelope.get("meta"), dict) else {}
    usage = agent_meta.get("usage") if isinstance(agent_meta, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    cache_read = int(usage.get("cacheRead") or 0)
    input_tokens = int(usage.get("input") or 0) + cache_read
    output_tokens = int(usage.get("output") or 0)
    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}, envelope


class OpenClawNemoFlowConfig(BaseModel):
    enabled: bool = False
    plugin_manifest_id: str = _NEMO_RELAY_PLUGIN_MANIFEST_ID
    plugin_package: str = f"npm:nemo-relay-openclaw@{_NEMO_RELAY_OPENCLAW_NPM_VERSION}"
    plugin_local_path: Optional[str] = None
    output_dir: Optional[str] = None


class OpenClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: Optional[ResourcesServerRef] = None
    concurrency: int = 4
    command: str = "openclaw"
    model: str = "nvidia/nvidia/qwen/qwen-235b"
    node_bin_dir: Optional[str] = None
    nvidia_api_key: str = ""  # pragma: allowlist secret
    nvidia_base_url: Optional[str] = None
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    work_dir: Optional[str] = None
    workspace_root: str = "outputs/openclaw_agent/workspaces"
    container_formatter: Optional[str] = None
    apptainer_command: str = "apptainer"
    setup_timeout: int = 900
    timeout: int = 900
    openclaw_agent_id: str = "main"
    thinking: str = "off"
    system_prompt: Optional[str] = None
    extra_args: list[str] = []
    openclaw_config: dict[str, Any] = Field(default_factory=dict)
    nemo_flow: OpenClawNemoFlowConfig = Field(default_factory=OpenClawNemoFlowConfig)
    verify_swebench: bool = False
    swebench_setup_dir: Optional[str] = None
    swebench_results_root: str = "outputs/openclaw_agent/swebench-verifier"
    swebench_verifier_timeout: int = 1200
    swebench_model_name: str = "openclaw_agent"

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenClawAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenClawAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class OpenClawAgent(_SWEBenchHelpers, SimpleResponsesAPIAgent):
    """OpenClaw as a Gym responses_api_agent.

    The SWE-bench materialization and verifier helpers are intentionally reused
    from the OpenCode POC while we validate the second harness shape.
    """

    config: OpenClawAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _SETUP_BASELINE: ClassVar[dict[str, Any]] = {
        "agents": {"defaults": {"workspace": "."}},
        "gateway": {"mode": "local"},
    }
    _HEADLESS_TOOL_DENY: ClassVar[tuple[str, ...]] = ("message",)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("OpenClaw command %r is not on PATH yet", self.config.command)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenClawAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    @classmethod
    def _merge_headless_tool_denies(cls, cfg: dict[str, Any]) -> None:
        tools = cfg.setdefault("tools", {})
        deny = tools.get("deny")
        if not isinstance(deny, list):
            deny = []
        merged = list(dict.fromkeys([item for item in deny if isinstance(item, str)] + list(cls._HEADLESS_TOOL_DENY)))
        tools["deny"] = merged

    def _normalize_nvidia_models_provider(self, cfg: dict[str, Any]) -> None:
        if not self.config.model.startswith("nvidia/"):
            return
        models = cfg.setdefault("models", {})
        providers = models.setdefault("providers", {})
        nvidia = providers.setdefault("nvidia", {})
        if not isinstance(nvidia.get("models"), list):
            nvidia["models"] = []
        base_url = nvidia.get("baseUrl")
        if not isinstance(base_url, str) or not base_url.strip() or base_url.strip().startswith("{env:"):
            nvidia["baseUrl"] = self.config.nvidia_base_url or os.environ.get("NVIDIA_BASE_URL") or (
                "https://integrate.api.nvidia.com/v1"
            )
        if not nvidia["models"]:
            model_id = self.config.model.split("/", 1)[1]
            nvidia["models"] = [{"id": model_id, "name": self.config.model}]

    def _merge_nemo_flow_plugin(self, cfg: dict[str, Any], output_dir: Path) -> None:
        if not self.config.nemo_flow.enabled:
            return

        pid = self.config.nemo_flow.plugin_manifest_id
        plugins = cfg.setdefault("plugins", {})
        plugins.setdefault("bundledDiscovery", "compat")
        if self.config.nemo_flow.plugin_local_path:
            load = plugins.setdefault("load", {})
            paths = load.get("paths")
            if not isinstance(paths, list):
                paths = []
            plugin_path = str(Path(self.config.nemo_flow.plugin_local_path).expanduser().resolve())
            if plugin_path not in paths:
                paths.append(plugin_path)
            load["paths"] = paths
        allow = plugins.get("allow")
        if isinstance(allow, list):
            if pid not in allow:
                allow.append(pid)
        else:
            plugins["allow"] = [pid]

        entries = plugins.setdefault("entries", {})
        existing = entries.get(pid) if isinstance(entries.get(pid), dict) else {}
        entry = {
            "enabled": True,
            "hooks": {"allowConversationAccess": True},
            "config": {
                "enabled": True,
                "backend": "hooks",
                "plugins": {
                    "version": 1,
                    "components": [
                        {
                            "kind": "observability",
                            "enabled": True,
                            "config": {
                                "version": 1,
                                "atif": {
                                    "enabled": True,
                                    "agent_name": "openclaw",
                                    "output_directory": str(output_dir / _NEMO_RELAY_ATIF_DIR_NAME),
                                },
                                "opentelemetry": {"enabled": False},
                                "openinference": {"enabled": False},
                            },
                        }
                    ],
                },
                "capture": {
                    "includePrompts": True,
                    "includeResponses": True,
                    "stripToolArgs": True,
                    "stripToolResults": True,
                },
            },
        }
        entries[pid] = self._deep_merge(entry, copy.deepcopy(existing))

    def _build_openclaw_config(self, output_dir: Path) -> dict[str, Any]:
        cfg = copy.deepcopy(self._SETUP_BASELINE)
        self._deep_merge(cfg, copy.deepcopy(self.config.openclaw_config))
        self._normalize_nvidia_models_provider(cfg)
        self._merge_headless_tool_denies(cfg)
        self._merge_nemo_flow_plugin(cfg, output_dir)
        return cfg

    def _artifact_root(self, work_dir: Optional[str]) -> Path:
        if work_dir:
            root = Path(work_dir).parent / "openclaw-artifacts"
        else:
            root = Path(self.config.workspace_root).expanduser() / f"openclaw_{uuid4().hex[:8]}"
            if not root.is_absolute():
                root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _home_root(self, work_dir: Optional[str]) -> Path:
        if work_dir:
            root = Path(work_dir).parent / ".openclaw-home"
        else:
            root = Path(self.config.workspace_root).expanduser() / f"openclaw_home_{uuid4().hex[:8]}"
            if not root.is_absolute():
                root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _env(self, home: Path) -> dict[str, str]:
        env = {**os.environ, "HOME": str(home)}
        node_bin_dir = self.config.node_bin_dir
        if not node_bin_dir:
            nvm_versions = Path(os.environ.get("NVM_DIR", Path.home() / ".nvm")) / "versions" / "node"
            v22_bins = sorted(nvm_versions.glob("v22*/bin"), reverse=True)
            node_bin_dir = str(v22_bins[0]) if v22_bins else None
        if node_bin_dir:
            env["PATH"] = f"{node_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        if self.config.nvidia_api_key:
            env["NVIDIA_API_KEY"] = self.config.nvidia_api_key
        if self.config.nvidia_base_url:
            env["NVIDIA_BASE_URL"] = self.config.nvidia_base_url
        if self.config.openai_api_key:
            env["OPENAI_API_KEY"] = self.config.openai_api_key
        if self.config.openai_base_url:
            env["OPENAI_BASE_URL"] = self.config.openai_base_url
        return env

    async def _run_exec(
        self,
        args: list[str],
        *,
        cwd: Optional[str],
        env: dict[str, str],
        timeout: int,
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"Timed out after {timeout}s: {shlex.join(args)}") from None
        return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def _run_openclaw(
        self,
        instruction: str,
        system_prompt: Optional[str],
        work_dir: Optional[str],
    ) -> tuple[str, str, dict[str, str]]:
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        artifact_root = self._artifact_root(work_dir)
        home = self._home_root(work_dir)
        env = self._env(home)

        upload_path = artifact_root / "openclaw.upload.json"
        upload_path.write_text(json.dumps(self._build_openclaw_config(artifact_root), indent=2) + "\n")
        (artifact_root / "instruction.txt").write_text(instruction)

        openclaw_home = home / ".openclaw"
        openclaw_home.mkdir(parents=True, exist_ok=True)

        setup_code, setup_out, setup_err = await self._run_exec(
            [*self.config.command_parts, "setup", "--workspace", "."],
            cwd=work_dir,
            env=env,
            timeout=self.config.setup_timeout,
        )
        (artifact_root / "openclaw.setup.stdout.txt").write_text(setup_out)
        (artifact_root / "openclaw.setup.stderr.txt").write_text(setup_err)
        if setup_code:
            raise RuntimeError(f"openclaw setup failed with code {setup_code}: {setup_err[:2000]}")

        shutil.copy2(upload_path, openclaw_home / "openclaw.json")

        cmd = [
            *self.config.command_parts,
            "agent",
            "--local",
            "--json",
            "--agent",
            self.config.openclaw_agent_id,
            "--thinking",
            self.config.thinking,
            "--model",
            self.config.model,
            "--message",
            prompt,
            *self.config.extra_args,
        ]
        code, stdout, stderr = await self._run_exec(cmd, cwd=work_dir, env=env, timeout=self.config.timeout)
        (artifact_root / "openclaw.txt").write_text(stdout)
        (artifact_root / "openclaw.stderr.txt").write_text(stderr)
        if code:
            LOG.warning("openclaw exited %d: %s", code, stderr[:500])

        metadata = {
            "openclaw_output_dir": str(artifact_root),
            "openclaw_log_path": str(artifact_root / "openclaw.txt"),
            "openclaw_stderr_path": str(artifact_root / "openclaw.stderr.txt"),
            "openclaw_setup_stdout_path": str(artifact_root / "openclaw.setup.stdout.txt"),
            "openclaw_setup_stderr_path": str(artifact_root / "openclaw.setup.stderr.txt"),
        }
        if self.config.nemo_flow.enabled:
            nemo_flow_output_dir = artifact_root / _NEMO_RELAY_ATIF_DIR_NAME
            metadata["nemo_relay_output_dir"] = str(nemo_flow_output_dir)
            metadata["nemo_flow_output_dir"] = str(nemo_flow_output_dir)
            atif_paths = sorted(str(path) for path in nemo_flow_output_dir.glob("*.json"))
            if atif_paths:
                metadata["nemo_relay_atif_path"] = atif_paths[0]
                metadata["nemo_relay_atif_paths"] = json.dumps(atif_paths)
                metadata["nemo_flow_atif_path"] = atif_paths[0]
                metadata["nemo_flow_atif_paths"] = json.dumps(atif_paths)

        envelope = _decode_last_json_dict_suffix(stdout)
        agent_meta = (envelope.get("meta") or {}).get("agentMeta") if envelope else {}
        session_file = agent_meta.get("sessionFile") if isinstance(agent_meta, dict) else None
        if isinstance(session_file, str) and session_file:
            source = Path(session_file)
            if source.exists():
                session_copy = artifact_root / "openclaw.session.jsonl"
                shutil.copy2(source, session_copy)
                metadata["openclaw_session_jsonl_path"] = str(session_copy)

        return stdout, self.config.model, metadata

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
        stdout, model_name, artifact_metadata = await self._run_openclaw(user_message, system_prompt, work_dir)
        output_items, usage, _ = parse_openclaw_output(stdout)
        patch = await self._collect_patch(work_dir)

        if not output_items:
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

    async def run(self, request: Request, body: OpenClawAgentRunRequest) -> OpenClawAgentVerifyResponse:
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

                return OpenClawAgentVerifyResponse(
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

            return OpenClawAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    OpenClawAgent.run_webserver()
