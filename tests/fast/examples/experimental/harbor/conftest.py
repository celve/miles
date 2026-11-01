"""Put the example on sys.path so its module imports by bare name, the way a
rollout loads it (``--custom-agent-function-path``)."""

import sys

from . import EXAMPLE_DIR

# the agent function shares the agent-server example's client, as every launcher's PYTHONPATH allows
for example_dir in (EXAMPLE_DIR.parents[1] / "swe-agent-harbor-docker", EXAMPLE_DIR):
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
