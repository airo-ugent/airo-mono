"""Tests for the Zenoh session configuration used by the multiprocess module."""

import json

import pytest

pytest.importorskip("zenoh", reason="eclipse-zenoh not installed")

from airo_camera_toolkit.cameras.multiprocess.base_publisher import (  # noqa: E402
    ZENOH_ROUTER_ENV_VAR,
    _make_zenoh_config,
)

_ROUTER = "tcp/192.168.0.10:7447"


def test_default_config_is_confined_to_localhost():
    """Without a router, sessions must not scout or listen beyond the local host.

    Zenoh's default config scouts over multicast on every interface, which would
    let two machines with the same namespace consume each other's frames.
    """
    conf = _make_zenoh_config()
    assert json.loads(conf.get_json("scouting/multicast/interface")) == "127.0.0.1"
    assert json.loads(conf.get_json("listen/endpoints")) == ["tcp/127.0.0.1:0"]
    assert json.loads(conf.get_json("transport/shared_memory/enabled")) is True


def test_router_endpoint_replaces_multicast_scouting():
    conf = _make_zenoh_config(router_endpoint=_ROUTER)
    assert json.loads(conf.get_json("scouting/multicast/enabled")) is False
    assert json.loads(conf.get_json("connect/endpoints")) == [_ROUTER]


def test_router_endpoint_can_be_set_through_the_environment(monkeypatch):
    monkeypatch.setenv(ZENOH_ROUTER_ENV_VAR, _ROUTER)
    conf = _make_zenoh_config()
    assert json.loads(conf.get_json("connect/endpoints")) == [_ROUTER]


def test_empty_router_environment_variable_is_ignored(monkeypatch):
    monkeypatch.setenv(ZENOH_ROUTER_ENV_VAR, "")
    conf = _make_zenoh_config()
    assert json.loads(conf.get_json("listen/endpoints")) == ["tcp/127.0.0.1:0"]


def test_shared_memory_can_be_disabled():
    conf = _make_zenoh_config(shm=False)
    assert json.loads(conf.get_json("transport/shared_memory/enabled")) is False
