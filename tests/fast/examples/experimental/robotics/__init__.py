"""Tests for the robotics (LIBERO) example, which lives outside this tree; conftest puts it on sys.path."""

from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[5] / "examples" / "experimental" / "robotics"
