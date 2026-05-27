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

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INSTANCE_ID = "django__django-13741"


def _read_jsonl_one(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.loads(stream.readline())


def _latest_file(root: Path, pattern: str) -> Path:
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} under {root}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def _input_text(rollout: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in rollout.get("responses_create_params", {}).get("input", []):
        content = item.get("content") if isinstance(item, dict) else item
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(block["text"] for block in content if isinstance(block, dict) and block.get("text"))
    return "\n\n".join(chunks)


def _message_text(item: dict[str, Any]) -> str:
    return "\n\n".join(
        block.get("text") or "" for block in item.get("content") or [] if isinstance(block, dict)
    )


def _step_from_output(index: int, item: dict[str, Any], timestamp: str) -> dict[str, Any]:
    item_type = item.get("type")
    step_id = index + 2
    if item_type == "message":
        return {
            "step_id": step_id,
            "timestamp": timestamp,
            "source": "agent",
            "message": [{"role": item.get("role") or "assistant", "content": _message_text(item)}],
            "extra": {
                "source": "gym.response.output",
                "response_item_type": item_type,
                "response_item_id": item.get("id"),
                "status": item.get("status"),
            },
        }
    if item_type == "function_call":
        return {
            "step_id": step_id,
            "timestamp": timestamp,
            "source": "agent",
            "message": f"tool call: {item.get('name') or ''}",
            "extra": {
                "source": "gym.response.output",
                "response_item_type": item_type,
                "call_id": item.get("call_id"),
                "tool_name": item.get("name"),
                "arguments": item.get("arguments") or "",
            },
        }
    if item_type == "function_call_output":
        return {
            "step_id": step_id,
            "timestamp": timestamp,
            "source": "system",
            "message": "",
            "observation": {
                "results": [
                    {
                        "source_call_id": item.get("call_id"),
                        "content": item.get("output") or "",
                    }
                ]
            },
            "extra": {
                "source": "gym.response.output",
                "response_item_type": item_type,
                "call_id": item.get("call_id"),
                "status": item.get("status"),
            },
        }
    return {
        "step_id": step_id,
        "timestamp": timestamp,
        "source": "system",
        "message": json.dumps(item, sort_keys=True),
        "extra": {"source": "gym.response.output", "response_item_type": item_type},
    }


def _response_metadata(rollout: dict[str, Any]) -> dict[str, Any]:
    metadata = rollout.get("response", {}).get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _rollout_patch_path(rollout: dict[str, Any]) -> Path | None:
    output_dir = rollout.get("swebench_output_dir")
    if not output_dir:
        return None
    return Path(output_dir) / "patch.diff"


def _patch_text(rollout: dict[str, Any], patch_path: Path | None) -> str | None:
    if patch_path and patch_path.exists():
        return patch_path.read_text()
    metadata_patch = _response_metadata(rollout).get("patch")
    if isinstance(metadata_patch, str) and metadata_patch:
        return metadata_patch
    return None


def _default_tmp_root() -> Path:
    env_value = os.environ.get("GYM_OUTPUT_DIR")
    if env_value:
        return Path(env_value)
    return Path(__file__).resolve().parents[6] / ".tmp/nemo-gym"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def regenerate(
    tmp_root: Path,
    output_dir: Path,
    no_relay_rollout_path: Path | None = None,
    relay_rollout_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    no_relay_rollout_path = no_relay_rollout_path or tmp_root / "hermes_django_13741_rollout.jsonl"
    relay_rollout_path = relay_rollout_path or tmp_root / "hermes_django_13741_relay_rollout.jsonl"

    no_relay = _read_jsonl_one(no_relay_rollout_path)
    relay = _read_jsonl_one(relay_rollout_path)
    relay_metadata = _response_metadata(relay)
    relay_atof_path = Path(
        relay_metadata.get("nemo_relay_atof_path") or _latest_file(tmp_root / "hermes-nemo-relay", "**/hermes.atof.jsonl")
    )
    relay_atif_path = Path(
        relay_metadata.get("nemo_relay_atif_path") or _latest_file(tmp_root / "hermes-nemo-relay", "**/*.atif.json")
    )
    no_relay_patch_path = _rollout_patch_path(no_relay)
    relay_patch_path = _rollout_patch_path(relay)
    relay_atif = json.loads(relay_atif_path.read_text())
    timestamp = datetime.now(timezone.utc).isoformat()

    steps: list[dict[str, Any]] = [
        {
            "step_id": 1,
            "timestamp": timestamp,
            "source": "user",
            "message": [{"role": "user", "content": _input_text(no_relay)}],
            "extra": {"source": "gym.responses_create_params.input", "instance_id": INSTANCE_ID},
        }
    ]
    steps.extend(
        _step_from_output(index, item, timestamp)
        for index, item in enumerate(no_relay.get("response", {}).get("output", []))
    )

    no_relay_atif = {
        "schema_version": "ATIF-v1.6",
        "session_id": f"gym-no-relay-hermes-{uuid.uuid4()}",
        "agent": {
            "name": "hermes",
            "model_name": no_relay.get("response", {}).get("model") or "policy_model",
            "version": "gym-rollout-reconstruction",
        },
        "steps": steps,
        "final_metrics": {
            "total_steps": len(steps),
            "reward": no_relay.get("reward"),
            "turns_used": no_relay.get("turns_used"),
            "finished_naturally": no_relay.get("finished_naturally"),
            "swebench_resolved": no_relay.get("swebench_resolved"),
            "output_items": len(no_relay.get("response", {}).get("output", [])),
        },
        "extra": {
            "source": "post-hoc Gym rollout reconstruction",
            "source_rollout_path": str(no_relay_rollout_path),
            "instance_id": INSTANCE_ID,
            "agent_ref": no_relay.get("agent_ref"),
            "swebench_output_dir": no_relay.get("swebench_output_dir"),
            "patch": _patch_text(no_relay, no_relay_patch_path),
            "limitations": [
                "Gym rollout output stores Responses API output items, not Relay event capture.",
                "Timestamps in this reconstructed ATIF were generated during post-processing.",
            ],
        },
    }

    (output_dir / "without-relay.gym-reconstructed.atif.json").write_text(
        json.dumps(no_relay_atif, indent=2) + "\n"
    )
    shutil.copyfile(relay_atif_path, output_dir / "with-relay.nemo-relay.atif.json")
    shutil.copyfile(relay_atof_path, output_dir / "with-relay.nemo-relay.atof.jsonl")

    with relay_atof_path.open() as stream:
        atof_events = [json.loads(line) for line in stream if line.strip()]

    relay_sources = Counter(step.get("source", "unknown") for step in relay_atif.get("steps", []))
    atof_categories = Counter(event.get("category") or event.get("kind") or event.get("name") or "unknown" for event in atof_events)
    no_relay_output_types = Counter(item.get("type", "unknown") for item in no_relay.get("response", {}).get("output", []))
    relay_output_types = Counter(item.get("type", "unknown") for item in relay.get("response", {}).get("output", []))

    summary = {
        "task": INSTANCE_ID,
        "agent": "hermes",
        "without_relay": {
            "source_rollout": str(no_relay_rollout_path),
            "atif_path": str(output_dir / "without-relay.gym-reconstructed.atif.json"),
            "step_count": len(no_relay_atif["steps"]),
            "response_output_type_counts": dict(sorted(no_relay_output_types.items())),
            "reward": no_relay.get("reward"),
            "turns_used": no_relay.get("turns_used"),
            "finished_naturally": no_relay.get("finished_naturally"),
            "swebench_resolved": no_relay.get("swebench_resolved"),
            "patch_path": str(no_relay_patch_path) if no_relay_patch_path else None,
            "note": "Post-hoc ATIF reconstructed from Gym Responses output items; baseline path uses Gym output without Relay event capture.",
        },
        "with_relay": {
            "source_rollout": str(relay_rollout_path),
            "atif_path": str(output_dir / "with-relay.nemo-relay.atif.json"),
            "atof_path": str(output_dir / "with-relay.nemo-relay.atof.jsonl"),
            "step_count": len(relay_atif.get("steps", [])),
            "step_sources": dict(sorted(relay_sources.items())),
            "atof_event_count": len(atof_events),
            "atof_event_categories": dict(sorted(atof_categories.items())),
            "response_output_type_counts": dict(sorted(relay_output_types.items())),
            "reward": relay.get("reward"),
            "turns_used": relay.get("turns_used"),
            "finished_naturally": relay.get("finished_naturally"),
            "swebench_resolved": relay.get("swebench_resolved"),
            "patch_path": str(relay_patch_path) if relay_patch_path else None,
            "note": "Adapter-level NeMoRelay capture around Hermes callbacks emitted ATOF plus normalized ATIF.",
        },
        "comparison_note": "Hermes is useful for comparison because the baseline Gym response already has tool call/output items, and Relay adds ATOF plus a normalized ATIF trajectory. The ATIF is currently noisier than OpenClaw because the adapter projects Hermes callbacks and assistant messages rather than consuming a native harness session log.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Hermes SWE-task comparison artifacts from .tmp rollouts.")
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=None,
        help="Directory containing Gym rollout artifacts. Defaults to GYM_OUTPUT_DIR or <legacy repo root>/.tmp/nemo-gym.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory where this comparison bundle's JSON/JSONL artifacts should be written.",
    )
    parser.add_argument(
        "--no-relay-rollout",
        type=Path,
        default=None,
        help="Baseline Gym rollout JSONL. Defaults to <tmp-root>/hermes_django_13741_rollout.jsonl.",
    )
    parser.add_argument(
        "--relay-rollout",
        type=Path,
        default=None,
        help="Relay-enabled Gym rollout JSONL. Defaults to <tmp-root>/hermes_django_13741_relay_rollout.jsonl.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Legacy shortcut: sets --tmp-root to <repo-root>/.tmp/nemo-gym when --tmp-root is omitted.",
    )
    args = parser.parse_args()
    tmp_root = args.tmp_root
    output_dir = args.output_dir
    if args.repo_root is not None:
        if tmp_root is None:
            tmp_root = args.repo_root / ".tmp/nemo-gym"
        if output_dir == _default_output_dir():
            output_dir = args.repo_root / "external/nemo-gym/docs/environment-tutorials/agent-harness-relay/hermes/artifacts"
    if tmp_root is None:
        tmp_root = _default_tmp_root()

    print(
        json.dumps(
            regenerate(
                tmp_root=tmp_root.resolve(),
                output_dir=output_dir.resolve(),
                no_relay_rollout_path=args.no_relay_rollout.resolve() if args.no_relay_rollout else None,
                relay_rollout_path=args.relay_rollout.resolve() if args.relay_rollout else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
