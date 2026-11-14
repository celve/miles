"""Put the robotics example's modules on sys.path: a rollout loads them by path, so they import each other by bare name."""

import sys

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
