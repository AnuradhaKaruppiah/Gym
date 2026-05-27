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
        assert cfg.command_parts == ["opencode"]
        assert cfg.timeout == 900
        assert cfg.thinking is True
        assert cfg.opencode_config == {}
        assert cfg.nemo_relay.enabled is False
        assert cfg.nemo_relay.plugin_package == "nemo-flow-opencode"
        assert cfg.nemo_relay.server_module_path is None
        assert cfg.nemo_relay.wrapper_filename == "nemo-relay-opencode-plugin.mjs"
        assert cfg.verify_swebench is False
        assert cfg.swebench_setup_dir is None
        assert cfg.swebench_results_root == "outputs/opencode_agent/swebench-verifier"
        assert cfg.swebench_model_name == "opencode_agent"

    def test_command_prefix_splits(self) -> None:
        cfg = _config(command="npx -y opencode-ai")
        assert cfg.command_parts == ["npx", "-y", "opencode-ai"]

    def test_candidate_instance_ids(self) -> None:
        assert OpenCodeAgent._candidate_instance_ids("django__django-13741") == [
            "django__django-13741",
            "django_1776_django-13741",
            "django_s_django-13741",
        ]

    def test_load_instance_dict_from_metadata(self) -> None:
        metadata = {"instance_dict": json.dumps({"repo": "django/django", "instance_id": "django__django-13741"})}

        instance = OpenCodeAgent._load_instance_dict(metadata)

        assert instance["repo"] == "django/django"
        assert instance["repo_name"] == "django/django"
        assert instance["instance_id"] == "django__django-13741"

    def test_load_instance_dict_from_flat_metadata(self) -> None:
        metadata = {
            "repo": "django/django",
            "instance_id": "django__django-13741",
            "base_commit": "abc123",
            "ignored": "x",
        }

        instance = OpenCodeAgent._load_instance_dict(metadata)

        assert instance == {
            "repo": "django/django",
            "repo_name": "django/django",
            "instance_id": "django__django-13741",
            "base_commit": "abc123",
        }

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
        assert inner["model"] == "nvidia/opus-frontier"
        assert inner["concurrency"] == 8
        assert inner["container_formatter"] is None
        assert inner["nemo_relay"]["enabled"] is False
        assert inner["nemo_relay"]["plugin_package"] == "nemo-flow-opencode"
        assert inner["nemo_relay"]["server_module_path"] is None
        assert inner["nemo_relay"]["wrapper_filename"] == "nemo-relay-opencode-plugin.mjs"
        assert inner["nemo_relay"]["atof_filename"] == "opencode.atof.jsonl"
        assert inner["verify_swebench"] is False
        assert inner["swebench_setup_dir"] is None
        assert inner["swebench_results_root"] == "outputs/opencode_agent/swebench-verifier"
        assert inner["swebench_model_name"] == "opencode_agent"
        assert inner["opencode_config"]["provider"]["nvidia"]["options"]["baseURL"] == "{env:NVIDIA_BASE_URL}"
        assert inner["opencode_config"]["provider"]["nvidia"]["models"]["opus-frontier"]["id"] == (
            "aws/anthropic/claude-opus-4-5"
        )
        assert inner["datasets"][0]["jsonl_fpath"] == (
            "responses_api_agents/opencode_agent/data/django_13741_smoke.jsonl"
        )

    def test_django_smoke_jsonl_parses(self) -> None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "django_13741_smoke.jsonl"
        lines = data_path.read_text().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["instance_id"] == "django__django-13741"
        assert row["agent_ref"] == {"type": "responses_api_agents", "name": "opencode_agent"}
        metadata = row["responses_create_params"]["metadata"]
        assert metadata["harbor_task_path"].endswith("django__django-13741")
        assert metadata["instance_id"] == row["instance_id"]

    def test_write_opencode_config_isolated(self, tmp_path: Path) -> None:
        agent = _make_agent(
            workspace_root=str(tmp_path / "workspaces"),
            opencode_config={"provider": {"nvidia": {"models": {"opus-frontier": {"id": "model-id"}}}}},
        )
        work_dir = tmp_path / "workspaces" / "task" / "testbed"
        work_dir.mkdir(parents=True)

        config_home = agent._write_opencode_config(str(work_dir))

        assert config_home == str(work_dir.parent / ".opencode-config")
        written = json.loads((Path(config_home) / "opencode" / "opencode.json").read_text())
        assert written["provider"]["nvidia"]["models"]["opus-frontier"]["id"] == "model-id"

    def test_write_opencode_config_with_nemo_relay_plugin(self, tmp_path: Path) -> None:
        server_module = tmp_path / "server.js"
        server_module.write_text("export default async function server() { return {}; }\n")
        agent = _make_agent(
            workspace_root=str(tmp_path / "workspaces"),
            opencode_config={"provider": {"nvidia": {"models": {"opus-frontier": {"id": "model-id"}}}}},
            nemo_relay={
                "enabled": True,
                "server_module_path": str(server_module),
                "output_dir": str(tmp_path / "relay-root"),
            },
        )
        work_dir = tmp_path / "workspaces" / "task" / "testbed"
        work_dir.mkdir(parents=True)
        relay_dir = agent._resolve_nemo_relay_output_dir(str(work_dir))

        config_home = agent._write_opencode_config(str(work_dir), relay_dir)

        written = json.loads((Path(config_home) / "opencode" / "opencode.json").read_text())
        plugin_uri = written["plugin"][0]
        assert plugin_uri.startswith("file://")
        wrapper_path = Path(plugin_uri.removeprefix("file://"))
        assert wrapper_path.name == "nemo-relay-opencode-plugin.mjs"
        wrapper = wrapper_path.read_text()
        assert server_module.as_uri() in wrapper
        assert '"logPath":' in wrapper
        assert str(relay_dir / "opencode-plugin.log") in wrapper
        plugin_options = json.loads(wrapper.split("const options = ", 1)[1].split(";\n\n", 1)[0])
        assert plugin_options["enabled"] is True
        assert plugin_options["logPath"] == str(relay_dir / "opencode-plugin.log")
        observability = plugin_options["plugins"]["components"][0]["config"]
        assert observability["atof"]["output_directory"] == str(relay_dir)
        assert observability["atof"]["filename"] == "opencode.atof.jsonl"
        assert observability["atif"]["filename_template"] == "opencode-{session_id}.atif.json"
        assert written["provider"]["nvidia"]["models"]["opus-frontier"]["id"] == "model-id"

    def test_nemo_relay_metadata_discovers_artifacts(self, tmp_path: Path) -> None:
        agent = _make_agent(nemo_relay={"enabled": True})
        relay_dir = tmp_path / "nemo-relay"
        relay_dir.mkdir()
        (relay_dir / "opencode.atof.jsonl").write_text("{}\n")
        (relay_dir / "opencode-plugin.log").write_text("{}\n")
        (relay_dir / "opencode-session-1.atif.json").write_text("{}")

        metadata = agent._nemo_relay_metadata(relay_dir)

        assert metadata["nemo_relay_output_dir"] == str(relay_dir)
        assert metadata["nemo_relay_atof_path"] == str(relay_dir / "opencode.atof.jsonl")
        assert json.loads(metadata["nemo_relay_atif_paths"]) == [str(relay_dir / "opencode-session-1.atif.json")]
        assert metadata["nemo_relay_log_path"] == str(relay_dir / "opencode-plugin.log")
