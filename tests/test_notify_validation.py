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


# -- topic hardening ---------------------------------------------------------

def test_auto_topic_carries_real_entropy():
    """ntfy topics are world-readable and world-writable, and the wordhash
    alone is ~22 bits behind a guessable model-name prefix."""
    import re

    import pymc as pm

    from cloudposterior.notify import NtfyNotifier

    with pm.Model(name="eight_schools") as model:
        pm.Normal("mu", 0, 1)

    topic = NtfyNotifier(model=model).topic
    assert re.search(r"-[0-9a-f]{16}$", topic), topic
    assert len(topic) <= 64


def test_auto_topics_are_unique_per_notifier():
    import pymc as pm

    from cloudposterior.notify import NtfyNotifier

    with pm.Model(name="m") as model:
        pm.Normal("mu", 0, 1)

    assert NtfyNotifier(model=model).topic != NtfyNotifier(model=model).topic


@pytest.mark.parametrize(
    "bad",
    ["has/slash", "..", "has?query", "has#frag", "x" * 65, "sp ace"],
)
def test_invalid_topics_are_rejected(bad):
    """The topic is interpolated into a URL path (a '/' or '?' retargets the
    POST) and rendered into a link."""
    from cloudposterior.notify import NtfyNotifier

    with pytest.raises(ValueError, match="invalid ntfy topic"):
        NtfyNotifier(topic=bad)


def test_empty_topic_falls_back_to_auto_generation():
    """An empty topic means "auto", same as None -- not an error."""
    from cloudposterior.notify import NtfyNotifier

    assert NtfyNotifier(topic="").topic


def test_long_model_names_are_truncated_to_fit():
    import pymc as pm

    from cloudposterior.notify import NtfyNotifier

    with pm.Model(name="a_very_long_model_name_that_goes_on_and_on_forever") as model:
        pm.Normal("mu", 0, 1)

    assert len(NtfyNotifier(model=model).topic) <= 64
