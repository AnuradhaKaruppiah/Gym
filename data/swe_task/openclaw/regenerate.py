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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


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


def _session_summary(path: Path) -> dict[str, Any]:
    events = _read_jsonl(path)
    return {
        "total_events": len(events),
        "event_types": dict(sorted(Counter(event.get("type", "unknown") for event in events).items())),
    }


def _build_no_relay_atif(
    rollout: dict[str, Any],
    rollout_path: Path,
    session_path: Path,
    patch_path: Path,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    output = rollout.get("response", {}).get("output", [])
    response_id = rollout.get("response", {}).get("id")

    steps: list[dict[str, Any]] = [
        {
            "step_id": 1,
            "timestamp": timestamp,
            "source": "user",
            "message": [{"role": "user", "content": _input_text(rollout)}],
            "extra": {"source": "gym.responses_create_params.input", "instance_id": INSTANCE_ID},
        }
    ]
    for item in output:
        steps.append(
            {
                "step_id": len(steps) + 1,
                "timestamp": timestamp,
                "source": "agent",
                "message": [{"role": item.get("role") or "assistant", "content": _message_text(item)}],
                "extra": {
                    "source": "gym.response.output",
                    "response_id": response_id,
                    "status": item.get("status"),
                },
            }
        )

    steps.append(
        {
            "step_id": len(steps) + 1,
            "timestamp": timestamp,
            "source": "system",
            "message": [{"role": "system", "content": "SWE-bench verifier completed for Gym rollout."}],
            "extra": {
                "source": "gym.swebench_verifier",
                "reward": rollout.get("reward"),
                "swebench_resolved": rollout.get("swebench_resolved"),
                "swebench_report_path": rollout.get("swebench_report_path"),
                "swebench_summary_path": rollout.get("swebench_summary_path"),
            },
        }
    )

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": f"gym-no-relay-openclaw-{uuid.uuid4()}",
        "agent": {
            "name": "openclaw",
            "model_name": rollout.get("response", {}).get("model") or "unknown",
            "version": "gym-rollout-reconstruction",
        },
        "steps": steps,
        "final_metrics": {
            "total_steps": len(steps),
            "reward": rollout.get("reward"),
            "turns_used": rollout.get("turns_used"),
            "finished_naturally": rollout.get("finished_naturally"),
            "swebench_resolved": rollout.get("swebench_resolved"),
            "output_items": len(output),
        },
        "extra": {
            "source": "post-hoc Gym rollout reconstruction",
            "source_rollout_path": str(rollout_path),
            "instance_id": INSTANCE_ID,
            "agent_ref": rollout.get("agent_ref"),
            "native_openclaw_session_jsonl_path": str(session_path),
            "swebench_output_dir": rollout.get("swebench_output_dir"),
            "patch": patch_path.read_text() if patch_path.exists() else None,
            "limitations": [
                "Gym rollout output stores the final Responses API result, not full normalized harness events.",
                "OpenClaw native session JSONL has richer event boundaries but is not ATIF-normalized without Relay.",
                "Timestamps in this reconstructed ATIF were generated during post-processing.",
            ],
        },
    }


def _default_tmp_root() -> Path:
    env_value = os.environ.get("GYM_OUTPUT_DIR")
    if env_value:
        return Path(env_value)
    return Path(__file__).resolve().parents[5] / ".tmp/nemo-gym"


def regenerate(
    tmp_root: Path,
    output_dir: Path,
    no_relay_rollout_path: Path | None = None,
    relay_rollout_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    no_relay_rollout_path = no_relay_rollout_path or tmp_root / "openclaw_django_13741_rollout.jsonl"
    relay_rollout_path = relay_rollout_path or tmp_root / "openclaw_django_13741_relay_rollout.jsonl"
    no_relay_session_path = _latest_file(
        tmp_root / "openclaw-workspaces",
        "django__django-13741_*/openclaw-artifacts/openclaw.session.jsonl",
    )
    relay_session_path = _latest_file(
        tmp_root / "openclaw-relay-workspaces",
        "django__django-13741_*/openclaw-artifacts/openclaw.session.jsonl",
    )
    relay_atif_path = _latest_file(
        tmp_root / "openclaw-relay-workspaces",
        "django__django-13741_*/openclaw-artifacts/nemo-flow-atif/*.json",
    )

    no_relay = _read_jsonl_one(no_relay_rollout_path)
    relay = _read_jsonl_one(relay_rollout_path)
    no_relay_patch_path = Path(no_relay["swebench_output_dir"]) / "patch.diff"
    relay_patch_path = Path(relay["swebench_output_dir"]) / "patch.diff"
    relay_atif = json.loads(relay_atif_path.read_text())

    no_relay_atif = _build_no_relay_atif(
        rollout=no_relay,
        rollout_path=no_relay_rollout_path,
        session_path=no_relay_session_path,
        patch_path=no_relay_patch_path,
    )
    (output_dir / "without-relay.gym-reconstructed.atif.json").write_text(
        json.dumps(no_relay_atif, indent=2) + "\n"
    )
    shutil.copyfile(relay_atif_path, output_dir / "with-relay.nemo-relay.atif.json")
    shutil.copyfile(no_relay_session_path, output_dir / "without-relay.openclaw.session.jsonl")
    shutil.copyfile(relay_session_path, output_dir / "with-relay.openclaw.session.jsonl")

    relay_sources = Counter(step.get("source", "unknown") for step in relay_atif.get("steps", []))
    no_relay_summary = _session_summary(no_relay_session_path)
    relay_summary = _session_summary(relay_session_path)

    summary = {
        "task": INSTANCE_ID,
        "agent": "openclaw",
        "without_relay": {
            "source_rollout": str(no_relay_rollout_path),
            "atif_path": str(output_dir / "without-relay.gym-reconstructed.atif.json"),
            "native_session_jsonl_path": str(output_dir / "without-relay.openclaw.session.jsonl"),
            "native_session_summary": no_relay_summary,
            "step_count": len(no_relay_atif["steps"]),
            "reward": no_relay.get("reward"),
            "turns_used": no_relay.get("turns_used"),
            "finished_naturally": no_relay.get("finished_naturally"),
            "swebench_resolved": no_relay.get("swebench_resolved"),
            "patch_path": str(no_relay_patch_path),
            "note": "Post-hoc ATIF reconstructed from Gym rollout output. OpenClaw native session JSONL is also included, but it is not normalized ATIF.",
        },
        "with_relay": {
            "source_rollout": str(relay_rollout_path),
            "atif_path": str(output_dir / "with-relay.nemo-relay.atif.json"),
            "native_session_jsonl_path": str(output_dir / "with-relay.openclaw.session.jsonl"),
            "native_session_summary": relay_summary,
            "step_count": len(relay_atif.get("steps", [])),
            "step_sources": dict(sorted(relay_sources.items())),
            "reward": relay.get("reward"),
            "turns_used": relay.get("turns_used"),
            "finished_naturally": relay.get("finished_naturally"),
            "swebench_resolved": relay.get("swebench_resolved"),
            "patch_path": str(relay_patch_path),
            "note": "NeMoRelay/NeMoFlow plugin emitted normalized ATIF from OpenClaw events. OpenClaw session JSONL is included as the raw harness log.",
        },
        "comparison_note": "OpenClaw appears more hook-friendly than OpenCode for this POC: both baseline and Relay runs have a compact native session JSONL with message/tool events, and Relay converts the enabled run into a 20-step ATIF trajectory.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate OpenClaw SWE-task comparison artifacts from .tmp rollouts.")
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=None,
        help="Directory containing Gym rollout artifacts. Defaults to GYM_OUTPUT_DIR or <legacy repo root>/.tmp/nemo-gym.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where this comparison bundle should be written.",
    )
    parser.add_argument(
        "--no-relay-rollout",
        type=Path,
        default=None,
        help="Baseline Gym rollout JSONL. Defaults to <tmp-root>/openclaw_django_13741_rollout.jsonl.",
    )
    parser.add_argument(
        "--relay-rollout",
        type=Path,
        default=None,
        help="Relay-enabled Gym rollout JSONL. Defaults to <tmp-root>/openclaw_django_13741_relay_rollout.jsonl.",
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
        if output_dir == Path(__file__).resolve().parent:
            output_dir = args.repo_root / "external/nemo-gym/data/swe_task/openclaw"
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
