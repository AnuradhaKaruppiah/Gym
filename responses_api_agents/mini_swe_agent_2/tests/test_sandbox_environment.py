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

from types import SimpleNamespace

from responses_api_agents.mini_swe_agent_2 import sandbox_environment as sandbox_environment_module
from responses_api_agents.mini_swe_agent_2.sandbox_environment import MiniSWESandboxEnvironment, Submitted


def test_check_finished_raises_submitted_for_submit_sentinel() -> None:
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    try:
        env._check_finished(
            {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
                "returncode": 0,
                "exception_info": "",
            }
        )
    except Submitted as error:
        assert error.messages == (
            {
                "role": "exit",
                "content": "patch contents\n",
                "extra": {"exit_status": "Submitted", "submission": "patch contents\n"},
            },
        )
    else:
        raise AssertionError("Expected Submitted")


def test_check_finished_ignores_nonzero_submit_sentinel() -> None:
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    env._check_finished(
        {
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
            "returncode": 1,
            "exception_info": "",
        }
    )


def test_execute_records_relay_tool_span(monkeypatch) -> None:
    events = []

    class FakeRelayRun:
        def start_sandbox_exec(self, **kwargs):
            events.append(("start", kwargs))
            return "relay-handle"

        def finish_sandbox_exec(self, handle, **kwargs):
            events.append(("finish", handle, kwargs))

    class FakeSandbox:
        def exec(self, handle, command, *, cwd, timeout_s, user):
            events.append(("exec", handle, command, cwd, timeout_s, user))
            return SimpleNamespace(stdout="out", stderr="err", return_code=0)

    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)
    env.config = SimpleNamespace(
        activate_conda=False,
        conda_env=None,
        step_timeout=10,
        eval_timeout=20,
        cwd="/workspace",
        user="root",
    )
    env._sandbox = FakeSandbox()
    env._handle = "sandbox-handle"
    monkeypatch.setattr(sandbox_environment_module, "current_relay_run", lambda: FakeRelayRun())

    result = env.execute({"command": "echo hi"}, cwd="/testbed", is_eval=True)

    assert result == {"output": "out\nerr", "returncode": 0, "exception_info": ""}
    assert events[0] == (
        "start",
        {
            "command": "echo hi",
            "cwd": "/testbed",
            "is_eval": True,
            "timeout_s": 20,
            "user": "root",
        },
    )
    assert events[1] == ("exec", "sandbox-handle", "echo hi", "/", 20, "root")
    assert events[2] == (
        "finish",
        "relay-handle",
        {"response": {"output": "out\nerr", "returncode": 0, "exception_info": ""}},
    )
