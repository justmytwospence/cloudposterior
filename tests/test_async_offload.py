"""_run_blocking off-loads blocking Modal calls when an event loop is active."""

import asyncio
import threading

from cloudposterior.backends.modal_backend import _run_blocking


def test_run_blocking_runs_directly_without_loop():
    main = threading.current_thread().ident
    seen = {}

    def fn():
        seen["tid"] = threading.current_thread().ident
        return 42

    assert _run_blocking(fn) == 42
    assert seen["tid"] == main  # no running loop -> runs inline


def test_run_blocking_offloads_under_running_loop():
    main = threading.current_thread().ident
    seen = {}

    def fn(a, b):
        seen["tid"] = threading.current_thread().ident
        return a + b

    async def go():
        return _run_blocking(fn, 2, 3)

    assert asyncio.run(go()) == 5
    assert seen["tid"] != main  # active loop -> ran in a worker thread
