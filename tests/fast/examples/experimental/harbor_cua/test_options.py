"""The run-wide options: one set of defaults, reaching both the parser and the rollout."""

import argparse
from argparse import Namespace
from dataclasses import fields

import pytest

from examples.experimental.harbor_cua.options import (
    WRITE_OFF_ABORT_GROUP,
    WRITE_OFF_MODES,
    HarborCuaOptions,
    add_arguments,
)


def parsed(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(argv)


def test_every_field_has_a_flag_and_the_same_default():
    defaults = HarborCuaOptions()
    namespace = parsed([])

    for field in fields(HarborCuaOptions):
        assert getattr(namespace, f"harbor_cua_{field.name}") == getattr(defaults, field.name)


def test_flags_round_trip_into_the_options():
    namespace = parsed(["--harbor-cua-max-turns", "3", "--harbor-cua-max-pixels", "1000"])

    options = HarborCuaOptions.from_args(namespace)

    assert options.max_turns == 3
    assert options.resize_settings == {"max_pixels": 1000}


def test_options_fall_back_to_their_defaults_on_a_bare_namespace():
    options = HarborCuaOptions.from_args(Namespace())

    assert options.max_turns == HarborCuaOptions().max_turns
    assert options.write_off_mode == WRITE_OFF_ABORT_GROUP
    assert options.resize_settings is None


def test_an_impossible_option_is_refused_at_construction():
    with pytest.raises(ValueError, match="write-off-mode"):
        HarborCuaOptions(write_off_mode="shrug")
    with pytest.raises(ValueError, match="max-turns"):
        HarborCuaOptions(max_turns=0)
    with pytest.raises(ValueError, match="max-sandboxes"):
        HarborCuaOptions(max_sandboxes=0)


def test_the_sample_budget_falls_back_to_the_rollout_context_length():
    options = HarborCuaOptions()

    assert options.sample_token_budget(Namespace(rollout_max_context_len=4096)) == 4096
    assert options.sample_token_budget(Namespace()) is None
    assert HarborCuaOptions(max_sample_tokens=99).sample_token_budget(Namespace(rollout_max_context_len=4096)) == 99


def test_the_write_off_modes_are_the_two_documented_ones():
    assert WRITE_OFF_MODES == ("abort_group", "mask_sample")
