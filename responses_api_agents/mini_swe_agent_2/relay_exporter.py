# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional NeMo Relay exporter for sandbox-backed mini-swe-agent runs."""

from __future__ import annotations

import contextvars
import json
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4


_CURRENT_RUN: contextvars.ContextVar["MiniSWERelayRun | None"] = contextvars.ContextVar(
    "mini_swe_agent_2_relay_run",
    default=None,
)


def current_relay_run() -> "MiniSWERelayRun | None":
    """Return the active relay run for this worker context, if one is enabled."""

    run = _CURRENT_RUN.get()
    return run if run and run.active else None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def format_output_dir(template: str, values: dict[str, Any]) -> str:
    """Format an output-dir template while tolerating optional Gym placeholders."""

    safe_values = {key: "" if value is None else str(value) for key, value in values.items()}
    return template.format_map(_MissingKeyDict(safe_values))


class _MissingKeyDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class MiniSWERelayRun:
    """Owns Relay subscribers and root scope for a single mini-swe-agent run."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str,
        strict: bool,
        trajectory_id: str,
        instance_id: str,
        model_name: str,
        task_index: int | str | None = None,
        rollout_index: int | str | None = None,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.strict = strict
        self.trajectory_id = trajectory_id
        self.instance_id = instance_id
        self.model_name = model_name
        self.task_index = task_index
        self.rollout_index = rollout_index

        self.active = False
        self.error: str | None = None
        self.atof_path = self.output_dir / "events.atof.jsonl"
        self.atif_path = self.output_dir / "trajectory.atif.json"

        self._nemo_relay: Any | None = None
        self._atif_exporter: Any | None = None
        self._atof_exporter: Any | None = None
        self._root_handle: Any | None = None
        self._token: contextvars.Token | None = None
        self._atif_subscriber = f"mini_swe_agent_2_atif_{uuid4().hex}"
        self._atof_subscriber = f"mini_swe_agent_2_atof_{uuid4().hex}"

    def __enter__(self) -> "MiniSWERelayRun":
        if not self.enabled:
            return self

        try:
            import nemo_relay

            self._nemo_relay = nemo_relay
            self.output_dir.mkdir(parents=True, exist_ok=True)

            atof_config = nemo_relay.AtofExporterConfig()
            atof_config.output_directory = str(self.output_dir)
            atof_config.filename = self.atof_path.name
            atof_config.mode = nemo_relay.AtofExporterMode.Overwrite
            self._atof_exporter = nemo_relay.AtofExporter(atof_config)
            self._atof_exporter.register(self._atof_subscriber)

            self._atif_exporter = nemo_relay.AtifExporter(
                self.trajectory_id,
                "mini_swe_agent_2",
                "0.1.0",
                model_name=self.model_name,
                tool_definitions=[{"name": "sandbox.exec"}],
                extra={
                    "gym_agent": "mini_swe_agent_2",
                    "instance_id": self.instance_id,
                    "task_index": self.task_index,
                    "rollout_index": self.rollout_index,
                },
            )
            self._atif_exporter.register(self._atif_subscriber)

            self._root_handle = nemo_relay.scope.push(
                "mini_swe_agent_2",
                nemo_relay.ScopeType.Agent,
                input={
                    "instance_id": self.instance_id,
                    "model_name": self.model_name,
                    "task_index": self.task_index,
                    "rollout_index": self.rollout_index,
                },
                metadata={"source": "nemo_gym"},
            )
            self._token = _CURRENT_RUN.set(self)
            self.active = True
        except Exception as exc:  # pragma: no cover - exercised through strict/fail-open behavior.
            self._shutdown_exporters()
            self._handle_error("failed to initialize NeMo Relay exporter", exc)

        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, _tb: Any) -> None:
        if not self.enabled:
            return

        if self._token is not None:
            _CURRENT_RUN.reset(self._token)
            self._token = None

        try:
            if self.active and self._nemo_relay is not None and self._root_handle is not None:
                output: dict[str, Any] = {
                    "instance_id": self.instance_id,
                    "status": "error" if exc else "completed",
                }
                if exc is not None:
                    output["error"] = str(exc)
                self._nemo_relay.scope.pop(self._root_handle, output=output)
                self._root_handle = None
                self._nemo_relay.subscribers.flush()

            if self._atif_exporter is not None:
                self.atif_path.write_text(self._atif_exporter.export_json(), encoding="utf-8")
        except Exception as finalize_exc:  # pragma: no cover - defensive around native exporter behavior.
            self._handle_error("failed to finalize NeMo Relay exporter", finalize_exc)
        finally:
            self._shutdown_exporters()
            self.active = False

    def record_mark(self, name: str, data: dict[str, Any] | None = None) -> None:
        if not self.active or self._nemo_relay is None:
            return
        try:
            self._nemo_relay.scope.event(name, handle=self._root_handle, data=_json_safe(data or {}))
        except Exception as exc:
            self._handle_error(f"failed to record Relay mark {name}", exc)

    def record_agent_messages(self, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        self.record_mark("mini_swe_agent_2.messages", {"messages": _json_safe(messages)})

    def start_sandbox_exec(
        self,
        *,
        command: str,
        cwd: str,
        is_eval: bool,
        timeout_s: int,
        user: str | int | None,
    ) -> Any | None:
        if not self.active or self._nemo_relay is None:
            return None
        try:
            return self._nemo_relay.tools.call(
                "sandbox.exec",
                {
                    "command": command,
                    "cwd": cwd,
                    "is_eval": is_eval,
                    "timeout_s": timeout_s,
                    "user": user,
                },
                handle=self._root_handle,
                metadata={"is_eval": is_eval},
            )
        except Exception as exc:
            self._handle_error("failed to start Relay sandbox.exec tool span", exc)
            return None

    def finish_sandbox_exec(
        self,
        handle: Any | None,
        *,
        response: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if handle is None or not self.active or self._nemo_relay is None:
            return
        result: dict[str, Any] = _json_safe(response or {})
        metadata: dict[str, Any] = {"status": "error" if error else "completed"}
        if error is not None:
            metadata["exception_type"] = type(error).__name__
            result = {
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        try:
            self._nemo_relay.tools.call_end(handle, result, metadata=metadata)
        except Exception as exc:
            self._handle_error("failed to finish Relay sandbox.exec tool span", exc)

    def artifacts(self) -> dict[str, str]:
        if not self.enabled:
            return {}
        paths: dict[str, str] = {}
        if self.atof_path.exists():
            paths["atof"] = str(self.atof_path)
        if self.atif_path.exists():
            paths["atif"] = str(self.atif_path)
        return paths

    def _shutdown_exporters(self) -> None:
        if self._atif_exporter is not None:
            try:
                self._atif_exporter.deregister(self._atif_subscriber)
            except Exception as exc:  # pragma: no cover
                self._handle_error("failed to deregister ATIF exporter", exc)
            self._atif_exporter = None
        if self._atof_exporter is not None:
            try:
                self._atof_exporter.deregister(self._atof_subscriber)
                self._atof_exporter.force_flush()
                self._atof_exporter.shutdown()
            except Exception as exc:  # pragma: no cover
                self._handle_error("failed to shutdown ATOF exporter", exc)
            self._atof_exporter = None
        if self._nemo_relay is not None:
            try:
                self._nemo_relay.subscribers.deregister(self._atif_subscriber)
                self._nemo_relay.subscribers.deregister(self._atof_subscriber)
            except Exception as exc:  # pragma: no cover
                self._handle_error("failed to deregister Relay subscribers", exc)

    def _handle_error(self, message: str, exc: Exception) -> None:
        self.error = f"{message}: {exc}"
        if self.strict:
            raise RuntimeError(self.error) from exc
        print(f"[MiniSWEAgent][Relay] {self.error}", flush=True)
