"""Free process-level watchdog regressions. No Modal calls or cloud jobs."""

import faulthandler
import signal
import time
from urllib.error import HTTPError

import pytest
from _modal_stop_worker import _send, assert_stop_outcome, supervise

PROJECT = "extraction-123456abcdef"
VOLUME = f"cp-{PROJECT}"


def fake_child(conn, stack_path, project):
    with open(stack_path, "w") as stack:
        faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)
        _send(conn, "app.run entry", control_dict="cp-dash-owned", app_id="ap-owned")
        _send(conn, "URLs ready", dashboard_url="https://example.test/?token=secret",
              progress_url="https://progress.test/?other=1", stop_url="https://stop.test/",
              token="secret")
        _send(conn, "waiting for first worker event")
        while True:
            time.sleep(0.01)


def never_start(conn, stack_path, project):
    with open(stack_path, "w") as stack:
        faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)
        _send(conn, "app.run entry")
        while True:
            time.sleep(0.01)


def result_error(conn, stack_path, project):
    _send(conn, "result transfer", control_dict="cp-dash-owned", app_id="ap-owned")
    _send(conn, "error", error_type="RuntimeError")


def hung_teardown(conn, stack_path, project):
    _send(conn, "teardown", control_dict="cp-dash-owned", app_id="ap-owned")
    while True:
        time.sleep(0.01)


def partial_child(conn, stack_path, project):
    _send(conn, "URLs ready", control_dict="cp-dash-owned", app_id="ap-owned",
          dashboard_url="https://example.test/", progress_url="https://progress.test/",
          stop_url="https://stop.test/", token="secret")
    # Leave time for the parent to observe live progress and stop.
    time.sleep(0.6)
    _send(conn, "result", status="stopped", draws=21, cache_hit=False)


def exact_cleanup(conn, identities):
    conn.send({"targets": identities, "app_inactive": True,
               "dict_absent": True, "volume_absent": True})
    conn.close()


def hung_cleanup(conn, identities):
    while True:
        time.sleep(0.01)


def no_progress(url, *, method="GET", origin=None, timeout=1):
    return {"sampling": {"total_draws": 0}}, {
        "Access-Control-Allow-Origin": origin, "Vary": "Origin"}


def live_progress(url, *, method="GET", origin=None, timeout=1):
    headers = {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
    if "invalid" in url:
        raise HTTPError(url, 403, "forbidden", headers, None)
    if method == "POST":
        return {"stopped": True}, headers
    return {"sampling": {"total_draws": 11}}, headers


def fail_stop(url, *, method="GET", origin=None, timeout=1):
    if method == "POST":
        raise HTTPError(url, 500, "failure", {}, None)
    return live_progress(url, method=method, origin=origin, timeout=timeout)


def run(child=fake_child, *, http=no_progress, cleanup=exact_cleanup):
    outcome = supervise(run_seconds=1.2, cleanup_seconds=1.5, child_target=child,
                        cleanup_target=cleanup, http=http, project=PROJECT,
                        stop_grace=0.05)
    assert outcome.elapsed < 3.0
    assert outcome.reaped
    return outcome


def test_startup_never_finishes_and_only_exact_volume_targeted():
    outcome = run(never_start)
    assert outcome.error and outcome.stage == "app.run entry"
    assert outcome.cleanup["targets"] == {"volume": VOLUME}
    assert "Current thread" in outcome.stack  # stack frames, not locals


def test_no_progress_and_exact_owned_resource_cleanup():
    outcome = run()
    assert outcome.error and outcome.stage == "waiting for first worker event"
    assert outcome.cleanup["targets"] == {
        "volume": VOLUME, "control_dict": "cp-dash-owned", "app_id": "ap-owned"}
    assert "Current thread" in outcome.stack


def test_failed_stop_is_not_retried():
    calls = []

    def failed(url, **kwargs):
        if kwargs.get("method") == "POST":
            calls.append(url)
        return fail_stop(url, **kwargs)

    outcome = run(http=failed)
    assert outcome.error == "stop request failed"
    assert len(calls) == 1
    assert outcome.cleanup["targets"]["app_id"] == "ap-owned"


def test_blocked_stop_http_is_bounded_without_retry():
    calls = []

    def blocked(url, **kwargs):
        if kwargs.get("method") == "POST":
            calls.append(url)
            time.sleep(5)
        return live_progress(url, **kwargs)

    outcome = run(http=blocked)
    assert outcome.error == "stop request failed"
    assert len(calls) == 1
    assert outcome.reaped


def test_future_result_error_does_not_hang_or_repeat_sampling():
    outcome = run(result_error)
    assert "RuntimeError" in outcome.error
    assert outcome.stage == "result transfer"
    assert outcome.cleanup["targets"]["volume"] == VOLUME


def test_hung_graceful_shutdown_is_killed_and_reaped():
    outcome = run(hung_teardown)
    assert outcome.error and outcome.stage == "teardown"
    assert outcome.reaped


def test_hung_cleanup_returns_explicit_unverified_leak():
    outcome = run(result_error, cleanup=hung_cleanup)
    assert outcome.cleanup == {"error": "cleanup deadline exceeded or process exited"}
    with pytest.raises(AssertionError, match="remote leak/unverified"):
        assert_stop_outcome(outcome)


def test_successful_partial_result_with_real_auth_and_stop_checks():
    outcome = run(partial_child, http=live_progress)
    assert outcome.error is None
    assert outcome.stage == "result"
    assert outcome.result["draws"] == 21
    assert_stop_outcome(outcome)


def test_partial_result_must_not_be_cached():
    outcome = run(cached_child)
    assert outcome.error == "partial result/status/cache assertion failed"


def cached_child(conn, stack_path, project):
    _send(conn, "result", status="stopped", draws=21, cache_hit=True)
