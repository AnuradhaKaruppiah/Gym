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

from nemo_gym.openai_utils import NeMoGymResponseOutputMessage
from nemo_gym.server_utils import ServerClient
from responses_api_agents.openclaw_agent.app import (
    OpenClawAgent,
    OpenClawAgentConfig,
    _decode_last_json_dict_suffix,
    parse_openclaw_output,
)


def _config(**kwargs) -> OpenClawAgentConfig:
    return OpenClawAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="openclaw_agent",
        **kwargs,
    )


def _make_agent(**kwargs) -> OpenClawAgent:
    with patch("responses_api_agents.openclaw_agent.app.OpenClawAgent.model_post_init"):
        agent = OpenClawAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


class TestOpenClawHelpers:
    def test_decode_last_json_dict_suffix(self) -> None:
        assert _decode_last_json_dict_suffix('log\n{"ok": true}') == {"ok": True}

    def test_parse_openclaw_output_text_and_usage(self) -> None:
        stdout = json.dumps(
            {
                "payloads": [{"text": "done"}],
                "meta": {"agentMeta": {"usage": {"input": 10, "output": 4, "cacheRead": 2}}},
            }
        )

        items, usage, envelope = parse_openclaw_output(stdout)

        assert envelope is not None
        assert usage == {"input_tokens": 12, "output_tokens": 4}
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "done"

    def test_parse_openclaw_output_reasoning(self) -> None:
        stdout = json.dumps(
            {
                "payloads": [
                    {"text": "think", "isReasoning": True},
                    {"text": "answer"},
                ],
                "meta": {},
            }
        )

        items, _, _ = parse_openclaw_output(stdout)

        assert "<think>\nthink\n</think>" in items[0].content[0].text
        assert "answer" in items[0].content[0].text


class TestOpenClawConfig:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.command_parts == ["openclaw"]
        assert cfg.model == "nvidia/nvidia/qwen/qwen-235b"
        assert cfg.node_bin_dir is None
        assert cfg.openclaw_agent_id == "main"
        assert cfg.thinking == "off"
        assert cfg.nemo_flow.enabled is False
        assert cfg.verify_swebench is False
        assert cfg.swebench_model_name == "openclaw_agent"

    def test_build_config_adds_nvidia_and_headless_denies(self, tmp_path: Path) -> None:
        agent = _make_agent(nvidia_base_url="https://example.test/v1")

        cfg = agent._build_openclaw_config(tmp_path)

        assert cfg["gateway"]["mode"] == "local"
        assert cfg["tools"]["deny"] == ["message"]
        nvidia = cfg["models"]["providers"]["nvidia"]
        assert nvidia["baseUrl"] == "https://example.test/v1"
        assert nvidia["models"] == [{"id": "nvidia/qwen/qwen-235b", "name": "nvidia/nvidia/qwen/qwen-235b"}]

    def test_build_config_adds_nemo_flow_plugin(self, tmp_path: Path) -> None:
        agent = _make_agent(nemo_flow={"enabled": True})

        cfg = agent._build_openclaw_config(tmp_path)

        entry = cfg["plugins"]["entries"]["nemo-relay"]
        component = entry["config"]["plugins"]["components"][0]
        assert component["kind"] == "observability"
        assert component["config"]["atof"]["enabled"] is True
        assert component["config"]["atof"]["output_directory"] == str(tmp_path / "nemo-relay-atof")
        assert component["config"]["atif"]["agent_name"] == "openclaw"
        assert component["config"]["atif"]["output_directory"] == str(tmp_path / "nemo-relay-atif")
        assert entry["config"]["capture"]["stripToolArgs"] is False
        assert entry["config"]["capture"]["stripToolResults"] is False

    def test_build_config_adds_nemo_flow_local_plugin_path(self, tmp_path: Path) -> None:
        plugin_root = tmp_path / "nemo-relay-openclaw"
        plugin_root.mkdir()
        agent = _make_agent(nemo_flow={"enabled": True, "plugin_local_path": str(plugin_root)})

        cfg = agent._build_openclaw_config(tmp_path)

        assert cfg["plugins"]["load"]["paths"] == [str(plugin_root.resolve())]

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "openclaw_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        inner = data["openclaw_agent"]["responses_api_agents"]["openclaw_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["command"] == "openclaw"
        assert inner["model"] == "nvidia/nvidia/qwen/qwen-235b"
        assert inner["node_bin_dir"] is None
        assert inner["nemo_flow"]["enabled"] is False
        assert inner["nemo_flow"]["plugin_manifest_id"] == "nemo-relay"
        assert inner["nemo_flow"]["plugin_package"] == "npm:nemo-relay-openclaw@0.3.0"
        assert inner["nemo_flow"]["plugin_local_path"] is None
        assert inner["datasets"][0]["jsonl_fpath"] == (
            "responses_api_agents/openclaw_agent/data/django_13741_smoke.jsonl"
        )

    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")
