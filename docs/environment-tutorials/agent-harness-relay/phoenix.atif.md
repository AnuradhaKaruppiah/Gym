# Viewing ATIF In Phoenix

Phoenix can load an ATIF trajectory as spans so the run can be inspected as a trace. Keep this setup in a separate virtual environment so Gym, agent harnesses, and Phoenix dependencies do not interfere with each other.

## Install Phoenix

From any workspace:

```bash
python3 -m venv .venv-phoenix
.venv-phoenix/bin/python -m pip install --upgrade pip
.venv-phoenix/bin/python -m pip install arize-phoenix
```

If you need a local Phoenix checkout instead of the published package:

```bash
export PHOENIX_SOURCE_DIR=/path/to/phoenix
.venv-phoenix/bin/python -m pip install -e "${PHOENIX_SOURCE_DIR}"
```

## Start Phoenix

Terminal 1:

```bash
export PHOENIX_PORT=6006
.venv-phoenix/bin/python -m phoenix.server.main serve \
  --host 127.0.0.1 \
  --port "${PHOENIX_PORT}"
```

Open `http://127.0.0.1:6006` in a browser.

## Upload Any ATIF JSON

Terminal 2:

```bash
export PHOENIX_URL=http://127.0.0.1:6006
export ATIF_JSON=/path/to/trajectory.atif.json
export PHOENIX_PROJECT="atif-$(date +%Y%m%d-%H%M%S)"

.venv-phoenix/bin/python - <<'PY'
import json
import os
from pathlib import Path

from phoenix.client import Client
from phoenix.client.helpers.atif import upload_atif_trajectories_as_spans

atif_path = Path(os.environ["ATIF_JSON"])
project_name = os.environ["PHOENIX_PROJECT"]
client = Client(base_url=os.environ.get("PHOENIX_URL", "http://127.0.0.1:6006"))

trajectory = json.loads(atif_path.read_text())
result = upload_atif_trajectories_as_spans(
    client,
    [trajectory],
    project_name=project_name,
)

print(f"uploaded {atif_path}")
print(f"project: {project_name}")
print(result)
PY
```

Use a fresh `PHOENIX_PROJECT` value for repeated uploads. Phoenix derives deterministic span IDs from ATIF identity fields, so reusing the same project for the same ATIF can hit duplicate span errors.

## Example ATIF Paths

From this directory:

```bash
export ATIF_JSON=hermes/artifacts/with-relay.nemo-relay.atif.json
export ATIF_JSON=openclaw/artifacts/with-relay.nemo-relay.atif.json
export ATIF_JSON=opencode/artifacts/with-relay.nemo-relay.atif.json
```

The ATIF file must pass Phoenix's ATIF validator. If validation fails, inspect the error first; common issues include missing `message` fields on v1.7 steps or older artifacts generated before the latest ATIF exporter fixes.
