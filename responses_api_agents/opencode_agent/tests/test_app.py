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
from responses_api_agents.opencode_agent.app import (
    OpenCodeAgent,
    OpenCodeAgentConfig,
    _extract_instruction,
    parse_opencode_jsonl,
)


def _config(**kwargs) -> OpenCodeAgentConfig:
    return OpenCodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="opencode_agent",
        **kwargs,
    )


def _make_agent(**kwargs) -> OpenCodeAgent:
    with patch("responses_api_agents.opencode_agent.app.OpenCodeAgent.model_post_init"):
        agent = OpenCodeAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _event(type_: str, **kwargs) -> str:
    return json.dumps({"type": type_, **kwargs})


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "opencode"
        assert cfg.timeout == 900
        assert cfg.thinking is True

    def test_semaphore_initialized(self) -> None:
        agent = _make_agent(concurrency=3)
        assert agent.sem._value == 3


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


class TestParseOpenCodeJsonl:
    def test_empty(self) -> None:
        items, usage = parse_opencode_jsonl("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_text_event(self) -> None:
        line = _event("text", part={"type": "text", "text": "done"})
        items, _ = parse_opencode_jsonl(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "done"

    def test_reasoning_prepended_to_next_text(self) -> None:
        lines = "\n".join(
            [
                _event("reasoning", part={"type": "reasoning", "text": "think"}),
                _event("text", part={"type": "text", "text": "answer"}),
            ]
        )
        items, _ = parse_opencode_jsonl(lines)
        assert len(items) == 1
        assert "<think>\nthink\n</think>" in items[0].content[0].text
        assert "answer" in items[0].content[0].text

    def test_tool_call_and_output(self) -> None:
        line = _event(
            "tool_use",
            part={
                "type": "tool",
                "tool": "bash",
                "callID": "call-1",
                "state": {"input": {"command": "ls"}, "output": "file.txt\n"},
            },
        )
        items, _ = parse_opencode_jsonl(line)
        assert len(items) == 2
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "bash"
        assert items[0].arguments == '{"command": "ls"}'
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].output == "file.txt\n"

    def test_step_finish_usage(self) -> None:
        line = _event(
            "step_finish",
            part={"tokens": {"input": 10, "output": 4, "cache": {"read": 2}}},
        )
        _, usage = parse_opencode_jsonl(line)
        assert usage == {"input_tokens": 12, "output_tokens": 4}

    def test_malformed_lines_skipped(self) -> None:
        good = _event("text", part={"type": "text", "text": "ok"})
        items, _ = parse_opencode_jsonl(f"not-json\n{good}")
        assert len(items) == 1


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "opencode_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        inner = data["opencode_agent"]["responses_api_agents"]["opencode_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["command"] == "opencode"
        assert inner["concurrency"] == 8
