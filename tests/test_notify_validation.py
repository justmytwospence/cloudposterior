"""Test the polymorphic notify kwarg validator."""

import pytest

from cloudposterior.api import _parse_notify


def test_notify_true_returns_auto():
    assert _parse_notify(True) == (None, None)


def test_notify_string_is_topic():
    assert _parse_notify("my-channel") == ("my-channel", None)


def test_notify_dict_with_topic_only():
    assert _parse_notify({"topic": "abc"}) == ("abc", None)


def test_notify_dict_with_topic_and_server():
    assert _parse_notify({"topic": "abc", "server": "https://ntfy.example"}) == (
        "abc",
        "https://ntfy.example",
    )


def test_notify_dict_with_unknown_keys_raises():
    with pytest.raises(ValueError, match="unexpected keys.*channel"):
        _parse_notify({"channel": "abc"})


def test_notify_dict_with_partial_unknown_keys_raises():
    with pytest.raises(ValueError, match="unexpected keys"):
        _parse_notify({"topic": "abc", "url": "x"})


@pytest.mark.parametrize("bad", [42, 3.14, object(), ["a"], ("a",)])
def test_notify_rejects_unknown_types(bad):
    with pytest.raises(TypeError, match="notify must be"):
        _parse_notify(bad)
