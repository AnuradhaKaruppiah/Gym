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
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.codex_agent.app import (
    CodexAgent,
    CodexAgentConfig,
    CodexNemoRelayConfig,
    _CodexNemoRelayCapture,
    _extract_instruction,
    parse_codex_jsonl,
)


def _config(**kwargs) -> CodexAgentConfig:
    return CodexAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="codex_agent",
        **kwargs,
    )


def _make_agent(**kwargs) -> CodexAgent:
    with patch("responses_api_agents.codex_agent.app.CodexAgent.model_post_init"):
        agent = CodexAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _event(type_: str, **kwargs) -> str:
    return json.dumps({"type": type_, **kwargs})


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "codex"
        assert cfg.command_parts == ["codex"]
        assert cfg.model is None
        assert cfg.sandbox == "workspace-write"
        assert cfg.approval_policy == "never"
        assert cfg.ephemeral is True
        assert cfg.nemo_relay.enabled is False
        assert cfg.nemo_relay.atof_filename == "codex.atof.jsonl"
        assert cfg.nemo_relay.atif_filename_template == "codex-{session_id}.atif.json"
        assert cfg.nemo_relay.include_raw_events is True
        assert cfg.nemo_relay.openinference.enabled is False
        assert cfg.nemo_relay.openinference.endpoint is None
        assert cfg.nemo_relay.openinference.transport == "http_binary"
        assert cfg.nemo_relay.openinference.project_name is None
        assert cfg.verify_swebench is False
        assert cfg.swebench_results_root == "outputs/codex_agent/swebench-verifier"
        assert cfg.swebench_model_name == "codex_agent"

    def test_command_prefix_splits(self) -> None:
        cfg = _config(command="npx -y @openai/codex")
        assert cfg.command_parts == ["npx", "-y", "@openai/codex"]

    def test_candidate_instance_ids(self) -> None:
        assert CodexAgent._candidate_instance_ids("django__django-13741") == [
            "django__django-13741",
            "django_1776_django-13741",
            "django_s_django-13741",
        ]

    def test_load_instance_dict_from_metadata(self) -> None:
        metadata = {"instance_dict": json.dumps({"repo": "django/django", "instance_id": "django__django-13741"})}

        instance = CodexAgent._load_instance_dict(metadata)

        assert instance["repo"] == "django/django"
        assert instance["repo_name"] == "django/django"
        assert instance["instance_id"] == "django__django-13741"

    def test_semaphore_initialized(self) -> None:
        agent = _make_agent(concurrency=3)
        assert agent.sem._value == 3

    def test_build_codex_command(self) -> None:
        agent = _make_agent(profile="frontier", config_overrides=['model_reasoning_effort="high"'])

        cmd = agent._build_codex_command("fix it", "/tmp/work")

        assert cmd[:2] == ["codex", "exec"]
        assert "--json" in cmd
        assert "--ask-for-approval" not in cmd
        assert ["--sandbox", "workspace-write"] == cmd[cmd.index("--sandbox") : cmd.index("--sandbox") + 2]
        assert ["--config", 'approval_policy="never"'] == cmd[cmd.index("--config") : cmd.index("--config") + 2]
        assert ["--profile", "frontier"] == cmd[cmd.index("--profile") : cmd.index("--profile") + 2]
        assert ["--cd", "/tmp/work"] == cmd[cmd.index("--cd") : cmd.index("--cd") + 2]
        assert cmd[-2:] == ["--", "fix it"]


class TestExtractInstruction:
    def test_user_only(self) -> None:
        user, system = _extract_instruction([NeMoGymEasyInputMessage(role="user", content="fix this")])
        assert user == "fix this"
        assert system is None

    def test_system_plus_user(self) -> None:
        user, system = _extract_instruction(
            [
                NeMoGymEasyInputMessage(role="system", content="be concise"),
                NeMoGymEasyInputMessage(role="user", content="fix this"),
            ]
        )
        assert user == "fix this"
        assert system == "be concise"


class TestParseCodexJsonl:
    def test_empty(self) -> None:
        items, usage = parse_codex_jsonl("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_item_completed_message(self) -> None:
        line = _event(
            "item.completed",
            item={"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        )
        items, _ = parse_codex_jsonl(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "done"

    def test_item_completed_agent_message(self) -> None:
        line = _event("item.completed", item={"type": "agent_message", "text": "done"})
        items, _ = parse_codex_jsonl(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "done"

    def test_function_call_and_output_items(self) -> None:
        lines = "\n".join(
            [
                _event(
                    "item.completed",
                    item={
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "exec_command",
                        "arguments": {"cmd": "ls"},
                    },
                ),
                _event(
                    "item.completed",
                    item={"type": "function_call_output", "call_id": "call-1", "output": "file.txt\n"},
                ),
            ]
        )
        items, _ = parse_codex_jsonl(lines)
        assert len(items) == 2
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "exec_command"
        assert items[0].arguments == '{"cmd": "ls"}'
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].output == "file.txt\n"

    def test_usage_from_response_event(self) -> None:
        line = _event("response.completed", response={"usage": {"input_tokens": 10, "output_tokens": 4}})
        _, usage = parse_codex_jsonl(line)
        assert usage == {"input_tokens": 10, "output_tokens": 4}

    def test_malformed_lines_skipped(self) -> None:
        good = _event("final_message", text="ok")
        items, _ = parse_codex_jsonl(f"not-json\n{good}")
        assert len(items) == 1


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "codex_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        inner = data["codex_agent"]["responses_api_agents"]["codex_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["command"] == "codex"
        assert inner["model"] is None
        assert inner["concurrency"] == 8
        assert inner["container_formatter"] is None
        assert inner["sandbox"] == "workspace-write"
        assert inner["approval_policy"] == "never"
        assert inner["nemo_relay"]["enabled"] is False
        assert inner["nemo_relay"]["atof_filename"] == "codex.atof.jsonl"
        assert inner["nemo_relay"]["atif_filename_template"] == "codex-{session_id}.atif.json"
        assert inner["nemo_relay"]["include_raw_events"] is True
        assert inner["nemo_relay"]["openinference"]["enabled"] is False
        assert inner["nemo_relay"]["openinference"]["endpoint"] is None
        assert inner["nemo_relay"]["openinference"]["transport"] == "http_binary"
        assert inner["nemo_relay"]["openinference"]["service_name"] == "nemo-relay-codex"
        assert inner["nemo_relay"]["openinference"]["project_name"] is None
        assert inner["nemo_relay"]["openinference"]["headers"] == {}
        assert inner["verify_swebench"] is False
        assert inner["swebench_setup_dir"] is None
        assert inner["swebench_results_root"] == "outputs/codex_agent/swebench-verifier"
        assert inner["swebench_model_name"] == "codex_agent"
        assert inner["datasets"][0]["jsonl_fpath"] == "responses_api_agents/codex_agent/data/django_13741_smoke.jsonl"

    def test_django_smoke_jsonl_parses(self) -> None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "django_13741_smoke.jsonl"
        lines = data_path.read_text().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["instance_id"] == "django__django-13741"
        assert row["agent_ref"] == {"type": "responses_api_agents", "name": "codex_agent"}
        assert row["responses_create_params"]["model"] == "codex-default"
        metadata = row["responses_create_params"]["metadata"]
        assert metadata["harbor_task_path"].endswith("django__django-13741")
        assert metadata["instance_id"] == row["instance_id"]


class TestOpenInferenceConfig:
    def test_disabled_openinference_skips_registration(self) -> None:
        capture = object.__new__(_CodexNemoRelayCapture)
        capture.config = CodexNemoRelayConfig()
        capture._openinference = None
        capture._openinference_subscriber = None

        capture._register_openinference(MagicMock(), MagicMock())

        assert capture._openinference is None
        assert capture._openinference_subscriber is None

    def test_register_openinference_project_header(self) -> None:
        class FakeOpenInferenceConfig:
            def __init__(self) -> None:
                self.headers = {}
                self.resource_attributes = {}

            def set_header(self, key: str, value: str) -> None:
                self.headers[key] = value

        class FakeOpenInferenceSubscriber:
            def __init__(self, config: FakeOpenInferenceConfig) -> None:
                self.config = config
                self.registered = []

            def register(self, name: str) -> None:
                self.registered.append(name)

        capture = object.__new__(_CodexNemoRelayCapture)
        capture.config = CodexNemoRelayConfig(
            openinference={
                "enabled": True,
                "endpoint": "http://127.0.0.1:6006/v1/traces",
                "project_name": "oi-gym-codex-relay",
                "headers": {"authorization": "Bearer test"},
                "resource_attributes": {"deployment.environment": "test"},
            }
        )
        capture._openinference = None
        capture._openinference_subscriber = None

        capture._register_openinference(FakeOpenInferenceConfig, FakeOpenInferenceSubscriber)

        assert isinstance(capture._openinference, FakeOpenInferenceSubscriber)
        assert capture._openinference_subscriber.startswith("codex_openinference_")
        config = capture._openinference.config
        assert config.endpoint == "http://127.0.0.1:6006/v1/traces"
        assert config.transport == "http_binary"
        assert config.service_name == "nemo-relay-codex"
        assert config.service_namespace == "gym"
        assert config.headers["authorization"] == "Bearer test"
        assert config.headers["x-project-name"] == "oi-gym-codex-relay"
        assert config.resource_attributes == {"deployment.environment": "test"}
        assert capture._openinference.registered == [capture._openinference_subscriber]
