"""Test ntfy notification formatting."""

from unittest.mock import patch, MagicMock

import numpy as np
import pymc as pm

from cloudposterior.notify import NtfyNotifier, _model_topic_name
from cloudposterior.progress import ChainProgress, JobPhase, PhaseUpdate, SamplingProgress


def test_notifier_generates_topic_from_model_name():
    """NtfyNotifier should derive topic from model name."""
    with pm.Model(name="eight_schools") as model:
        pm.Normal("mu", 0, 1)

    notifier = NtfyNotifier(model=model)
    assert notifier.topic.startswith("pd-eight-schools-")


def test_notifier_generates_topic_from_rv_names():
    """NtfyNotifier should derive topic from RV names when model has no name."""
    with pm.Model() as model:
        pm.Normal("mu", 0, 1)
        pm.HalfCauchy("tau", 5)

    notifier = NtfyNotifier(model=model)
    assert "mu" in notifier.topic
    assert "tau" in notifier.topic
    assert notifier.topic.startswith("pd-")


def test_notifier_custom_topic():
    """NtfyNotifier should accept a custom topic."""
    notifier = NtfyNotifier(topic="my-custom-topic")
    assert notifier.topic == "my-custom-topic"
    assert notifier.url == "https://ntfy.sh/my-custom-topic"


def test_notifier_custom_server():
    """NtfyNotifier should accept a custom server."""
    notifier = NtfyNotifier(topic="my-topic", server="https://ntfy.example.com")
    assert notifier.url == "https://ntfy.example.com/my-topic"
    assert notifier.server == "https://ntfy.example.com"


def test_notifier_server_from_env(monkeypatch):
    """NtfyNotifier should read server from env var."""
    monkeypatch.setenv("CLOUDPOSTERIOR_NTFY_SERVER", "https://ntfy.internal.io")
    notifier = NtfyNotifier(topic="test")
    assert notifier.server == "https://ntfy.internal.io"
    assert notifier.url == "https://ntfy.internal.io/test"


def test_notifier_env_topic(monkeypatch):
    """NtfyNotifier should read topic from env var."""
    monkeypatch.setenv("CLOUDPOSTERIOR_NTFY_TOPIC", "env-topic")
    notifier = NtfyNotifier()
    assert notifier.topic == "env-topic"


@patch("cloudposterior.notify.requests.post")
def test_notifier_sends_phase(mock_post):
    """Phase updates should trigger an HTTP POST to ntfy."""
    notifier = NtfyNotifier(topic="test-topic")
    notifier.show_phase(PhaseUpdate(
        phase=JobPhase.SERIALIZING,
        status="done",
        message="model + data packaged",
        elapsed=0.5,
    ))

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "test-topic" in call_kwargs[0][0]
    assert call_kwargs[1]["headers"]["X-Markdown"] == "yes"


@patch("cloudposterior.notify.requests.post")
def test_notifier_sends_sampling_progress(mock_post):
    """Sampling progress should include chain table in body."""
    notifier = NtfyNotifier(topic="test-topic")
    notifier.show_sampling(SamplingProgress(
        chains={
            0: ChainProgress(draw=500, total=1000, phase="sampling", draws_per_sec=100, divergences=2, step_size=0.5, tree_size=15),
            1: ChainProgress(draw=400, total=1000, phase="tuning", draws_per_sec=90, divergences=0, step_size=0.8, tree_size=7),
        },
        total_divergences=2,
        elapsed=5.0,
    ))

    mock_post.assert_called_once()
    body = mock_post.call_args[1]["data"].decode()
    assert "Chain" in body
    assert "500/1000" in body
    assert "400/1000" in body


@patch("cloudposterior.notify.requests.post")
def test_notifier_best_effort(mock_post):
    """HTTP failures should be silently swallowed."""
    mock_post.side_effect = ConnectionError("network down")
    notifier = NtfyNotifier(topic="test-topic")
    # Should not raise
    notifier.show_phase(PhaseUpdate(
        phase=JobPhase.SERIALIZING,
        status="done",
        message="test",
        elapsed=0.0,
    ))
