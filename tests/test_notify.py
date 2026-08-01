"""Test ntfy notification formatting."""

from unittest.mock import patch

import pymc as pm

from cloudposterior.notify import NtfyNotifier
from cloudposterior.progress import ChainProgress, JobPhase, PhaseUpdate, SamplingProgress


def test_notifier_generates_topic_from_model_name():
    """NtfyNotifier should derive topic from model name."""
    with pm.Model(name="eight_schools") as model:
        pm.Normal("mu", 0, 1)

    notifier = NtfyNotifier(model=model)
    assert notifier.topic.startswith("eight-schools-")
    # model-name + wordhash + random suffix
    assert len(notifier.topic.split("-")) == 5


def test_notifier_generates_topic_from_rv_names():
    """Unnamed models use RV names in the topic."""
    with pm.Model() as model:
        pm.Normal("mu", 0, 1)
        pm.HalfCauchy("tau", 5)

    notifier = NtfyNotifier(model=model)
    assert notifier.topic.startswith("mu-tau-")
    assert len(notifier.topic) <= 64  # ntfy's limit


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
def test_notifier_sends_on_sampling_start(mock_post):
    """Sampling start should trigger a notification."""
    notifier = NtfyNotifier(topic="test-topic")
    notifier.show_phase(PhaseUpdate(
        phase=JobPhase.SAMPLING,
        status="in_progress",
        message="MCMC sampling started",
        elapsed=0.0,
    ))
    notifier.stop()  # sends run on a worker thread; flush before asserting

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "test-topic" in call_kwargs[0][0]
    assert call_kwargs[1]["headers"]["X-Markdown"] == "yes"


@patch("cloudposterior.notify.requests.post")
def test_notifier_sends_on_sampling_complete(mock_post):
    """Sampling completion should include progress summary in body."""
    notifier = NtfyNotifier(topic="test-topic")
    # Feed sampling progress (not sent)
    notifier.show_sampling(SamplingProgress(
        chains={
            0: ChainProgress(draw=1000, total=1000, phase="sampling", draws_per_sec=100, divergences=2, step_size=0.5, tree_size=15),
            1: ChainProgress(draw=1000, total=1000, phase="sampling", draws_per_sec=90, divergences=0, step_size=0.8, tree_size=7),
        },
        total_divergences=2,
        elapsed=10.0,
    ))
    mock_post.assert_not_called()

    # Sampling complete triggers send with accumulated progress
    notifier.show_phase(PhaseUpdate(
        phase=JobPhase.SAMPLING,
        status="done",
        message="sampling complete",
        elapsed=10.0,
    ))
    notifier.stop()

    mock_post.assert_called_once()
    body = mock_post.call_args[1]["data"].decode()
    assert "Chain" in body
    assert "1000/1000" in body


@patch("cloudposterior.notify.requests.post")
def test_notifier_best_effort(mock_post):
    """HTTP failures should be silently swallowed."""
    mock_post.side_effect = ConnectionError("network down")
    notifier = NtfyNotifier(topic="test-topic")
    # Should not raise
    notifier.show_phase(PhaseUpdate(
        phase=JobPhase.SAMPLING,
        status="in_progress",
        message="test",
        elapsed=0.0,
    ))
    notifier.stop()


@patch("cloudposterior.notify.requests.post")
def test_notifier_completion_styling(mock_post):
    """The sampling-done send carries the completion tag/priority/title
    (regression: _is_complete() required a phase that never arrived in time,
    so the checkmark styling was dead code)."""
    notifier = NtfyNotifier(topic="test-topic")
    notifier.show_phase(PhaseUpdate(
        phase=JobPhase.SAMPLING, status="done",
        message="sampling complete", elapsed=10.0,
    ))
    notifier.stop()

    mock_post.assert_called_once()
    headers = mock_post.call_args[1]["headers"]
    assert headers["X-Tags"] == "white_check_mark"
    assert headers["X-Priority"] == "3"
    assert headers["X-Title"].endswith("[complete]")


@patch("cloudposterior.notify.requests.post")
def test_notifier_sends_failure_styling_on_error(mock_post):
    """An error phase triggers a send with the failure styling."""
    notifier = NtfyNotifier(topic="test-topic")
    notifier.show_phase(PhaseUpdate(
        phase=JobPhase.SAMPLING, status="error",
        message="kaboom", elapsed=3.0,
    ))
    notifier.stop()

    mock_post.assert_called_once()
    headers = mock_post.call_args[1]["headers"]
    assert headers["X-Tags"] == "rotating_light"
    assert headers["X-Priority"] == "4"
    assert headers["X-Title"].endswith("[failed]")
