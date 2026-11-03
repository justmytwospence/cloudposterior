"""Test-only process watchdog for the paid manual-stop smoke.

The parent owns all deadlines. The sampler runs on the child's main thread so a
stalled Modal client call cannot keep pytest's interpreter alive after a failure.
"""

from __future__ import annotations

import faulthandler
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

RUN_SECONDS = 240
CLEANUP_SECONDS = 60


def _send(conn, stage, **fields):
    conn.send({"stage": stage, **fields})


def _sample_child(conn, stack_path, project):
    # SIGUSR1 writes stack frames, not locals (which might contain credentials).
    with open(stack_path, "w", encoding="utf-8") as stack_file:
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file=stack_file, all_threads=True)
        try:
            import numpy as np
            import pymc as pm

            import cloudposterior as cp
            from cloudposterior import api
            from cloudposterior.backends import modal_backend
            from cloudposterior.config import RemoteConfig

            RemoteConfig.from_instance = classmethod(
                lambda cls, *args, **kwargs: cls(cpu=2, memory=4096, timeout=180,
                                                 idle_timeout=30)
            )
            original_provision = modal_backend.ModalBackend.provision
            original_running = modal_backend.ModalEnvironment._ensure_running
            original_upload = modal_backend.ModalEnvironment._upload_if_needed
            original_stream = modal_backend.PersistentModalSamplingJob.stream_progress
            original_result = modal_backend.PersistentModalSamplingJob.result
            original_teardown = modal_backend.ModalEnvironment.teardown

            def provision(self, *args, **kwargs):
                _send(conn, "provisioning", volume=f"cp-{project}")
                env = original_provision(self, *args, **kwargs)
                # App name is shared by all projects. NEVER use it as a cleanup target.
                _send(conn, "provisioned", volume=f"cp-{project}",
                      control_dict=env._dashboard_dict_name)
                return env

            def running(self):
                if not self._running:
                    run = self._app.run

                    def tracked_run(*args, **kwargs):
                        ctx = run(*args, **kwargs)

                        class TrackedContext:
                            def __enter__(inner):
                                _send(conn, "app.run entry")
                                result = ctx.__enter__()
                                _send(conn, "URL discovery", app_id=self._app.app_id,
                                      control_dict=self._dashboard_dict_name)
                                return result

                            def __exit__(inner, *exit_args):
                                return ctx.__exit__(*exit_args)
                        return TrackedContext()

                    self._app.run = tracked_run
                result = original_running(self)
                _send(conn, "URLs ready", app_id=self._app.app_id,
                      control_dict=self._dashboard_dict_name,
                      dashboard_url=self._dashboard_url,
                      progress_url=(self._progress_fn.get_web_url()
                                    if self._progress_fn else None),
                      stop_url=self._stop_url, token=self._stop_token)
                return result

            def upload(self, *args, **kwargs):
                _send(conn, "payload upload")
                return original_upload(self, *args, **kwargs)

            def stream(self, *args, **kwargs):
                _send(conn, "waiting for first worker event")
                iterator = original_stream(self, *args, **kwargs)
                first = True
                for event in iterator:
                    if first:
                        _send(conn, "first worker event")
                        first = False
                    yield event
                # A generator's return value carries the wire result; preserve it.
                # The original generator also sets _idata_bytes before returning.

            def result(self):
                _send(conn, "result transfer")
                return original_result(self)

            def teardown(self):
                _send(conn, "teardown")
                return original_teardown(self)

            modal_backend.ModalBackend.provision = provision
            modal_backend.ModalEnvironment._ensure_running = running
            modal_backend.ModalEnvironment._upload_if_needed = upload
            modal_backend.PersistentModalSamplingJob.stream_progress = stream
            modal_backend.PersistentModalSamplingJob.result = result
            modal_backend.ModalEnvironment.teardown = teardown

            with pm.Model() as model:
                mu = pm.Normal("mu", 0, 1)
                pm.Normal("obs", mu, 1, observed=np.array([0.5, -0.2, 0.1]))
            session = cp.cloud(model, remote=True, cache=True, dashboard=True,
                               progress=False, project=project)
            _send(conn, "sampling")
            with session:
                idata = pm.sample(draws=200_000, tune=60, chains=2, cores=2,
                                  nuts_sampler="pymc", random_seed=735,
                                  progressbar=False)
            report = cp.convergence.read_report(idata)
            _, lookup = api._cache_lookup(
                True, session.model,
                {"draws": 200_000, "tune": 60, "chains": 2, "cores": 2,
                 "nuts_sampler": "pymc", "random_seed": 735, "progressbar": False},
                overwrite=False, progress=False,
            )
            _send(conn, "result", status=report["status"],
                  draws=idata.posterior.sizes["draw"], cache_hit=lookup.hit)
        except BaseException as exc:
            # Never send exception text: Modal errors can include endpoint URLs.
            _send(conn, "error", error_type=type(exc).__name__,
                  stack="".join(traceback.format_list(traceback.extract_tb(exc.__traceback__))))
        finally:
            conn.close()


def _http_json(url, *, method="GET", origin=None, timeout=1):
    headers = {"Origin": origin} if origin else {}
    with urlopen(Request(url, method=method, headers=headers), timeout=timeout) as response:
        return json.load(response), response.headers


def _targeted_cleanup(conn, identities):
    """Isolated process: only identities emitted by this child may be touched."""
    import modal

    outcomes = {}
    app_id = identities.get("app_id")
    control_dict = identities.get("control_dict")
    volume = identities.get("volume")
    if app_id:
        # CLI accepts exact app ID; do not stop by the shared App name.
        try:
            subprocess.run([str(Path(sys.executable).with_name("modal")), "app", "stop", app_id], check=True,
                           capture_output=True, timeout=15)
            outcomes["app"] = "stopped"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            outcomes["app"] = type(exc).__name__
        try:
            listed = subprocess.run([str(Path(sys.executable).with_name("modal")), "app", "list", "--json"], check=True,
                                    capture_output=True, text=True, timeout=15)
            matches = [app for app in json.loads(listed.stdout)
                       if app.get("App ID") == app_id]
            outcomes["app_inactive"] = all(
                app.get("State", "").lower() not in {"running", "initializing"}
                and app.get("Tasks") == "0" for app in matches)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError, ValueError) as exc:
            outcomes["app_inactive"] = type(exc).__name__
    if control_dict:
        try:
            modal.Dict.objects.delete(control_dict, allow_missing=True)
            outcomes["dict"] = "deleted"
        except Exception as exc:
            outcomes["dict"] = type(exc).__name__
    if volume:
        try:
            modal.Volume.objects.delete(volume, allow_missing=True)
            outcomes["volume"] = "deleted"
        except Exception as exc:
            outcomes["volume"] = type(exc).__name__
    # from_name returns lazy references; hydration checks actual remote existence.
    for label, name, cls in (("dict", control_dict, modal.Dict),
                              ("volume", volume, modal.Volume)):
        if name:
            try:
                cls.from_name(name).hydrate()
            except Exception as exc:
                outcomes[f"{label}_absent"] = isinstance(exc, modal.exception.NotFoundError)
            else:
                outcomes[f"{label}_absent"] = False
    conn.send(outcomes)
    conn.close()


@dataclass
class StopOutcome:
    result: dict | None
    stage: str
    stack: str
    error: str | None
    cleanup: dict
    identities: dict
    elapsed: float
    reaped: bool


def _bounded_http(http, url, *, deadline, **kwargs):
    """Bound DNS/socket/custom fake calls even when their timeout is ignored."""
    queue = SimpleQueue()

    def call():
        try:
            queue.put((True, http(url, timeout=min(1, max(0.001, deadline - time.monotonic())),
                                  **kwargs)))
        except BaseException as exc:
            queue.put((False, exc))

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout=min(1, max(0, deadline - time.monotonic())))
    if thread.is_alive():
        raise TimeoutError("HTTP deadline exceeded")
    ok, value = queue.get_nowait()
    if not ok:
        raise value
    return value


def _stop_url(url, token):
    from urllib.parse import parse_qsl, urlunsplit

    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["token"] = token
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def supervise(*, run_seconds=RUN_SECONDS, cleanup_seconds=CLEANUP_SECONDS,
              child_target=_sample_child, cleanup_target=_targeted_cleanup,
              http=_http_json, project=None, stop_grace=2):
    """Bounded supervisor; injected targets and HTTP permit free watchdog tests."""
    ctx = mp.get_context("spawn")
    project = project or f"extraction-{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"extraction-[0-9a-f]{12}", project):
        raise ValueError("project must be a unique extraction test project")
    import tempfile

    started = time.monotonic()
    deadline = started + run_seconds
    cleanup_deadline = deadline + cleanup_seconds
    identities = {"volume": f"cp-{project}"}
    stage = "child startup"
    error = None
    result = None
    stack = ""
    stop_attempted = False
    with tempfile.TemporaryDirectory(prefix="cloudposterior-stop-") as tempdir:
        stack_path = str(Path(tempdir) / "stack.txt")
        Path(stack_path).touch()
        parent, child_conn = ctx.Pipe(duplex=False)
        child = ctx.Process(target=child_target, args=(child_conn, stack_path, project))
        child.start()
        child_conn.close()
        try:
            next_progress = 0.0
            while time.monotonic() < deadline and result is None and error is None:
                now = time.monotonic()
                if parent.poll(min(0.05, max(0, deadline - now))):
                    try:
                        msg = parent.recv()
                    except EOFError:
                        error = "child exited without result"
                        break
                    previous_stage = stage
                    stage = msg["stage"]
                    for key in ("volume", "control_dict", "app_id", "dashboard_url",
                                "progress_url", "stop_url", "token"):
                        if msg.get(key):
                            identities[key] = msg[key]
                    if stage == "error":
                        stack = msg.get("stack", "")
                        stage = previous_stage
                        error = f"child {msg['error_type']} at {stage}"
                    elif stage == "result":
                        result = msg
                    continue
                if not child.is_alive():
                    error = f"child exited ({child.exitcode}) before result"
                    break
                if identities.get("progress_url") and not stop_attempted and now >= next_progress:
                    next_progress = now + 0.5
                    origin_parts = urlsplit(identities["dashboard_url"])
                    origin = f"{origin_parts.scheme}://{origin_parts.netloc}"
                    url = _stop_url(identities["progress_url"], identities["token"])
                    try:
                        data, headers = _bounded_http(http, url, origin=origin,
                                                      deadline=deadline)
                        if headers.get("Access-Control-Allow-Origin") != origin or headers.get("Vary") != "Origin":
                            raise AssertionError("progress CORS mismatch")
                        if (data.get("sampling") or {}).get("total_draws", 0) >= 10:
                            stage = "live progress"
                            try:
                                _bounded_http(http, _stop_url(identities["progress_url"], "invalid"),
                                              origin=origin, deadline=deadline)
                            except HTTPError as exc:
                                if exc.code != 403 or exc.headers.get("Access-Control-Allow-Origin") != origin:
                                    raise AssertionError("invalid token response mismatch") from exc
                            else:
                                raise AssertionError("progress allowed invalid token")
                            stop_attempted = True
                            stage = "stop request"
                            response, stop_headers = _bounded_http(
                                http, _stop_url(identities["stop_url"], identities["token"]),
                                method="POST", origin=origin, deadline=deadline)
                            if response.get("stopped") is not True or stop_headers.get("Access-Control-Allow-Origin") != origin:
                                raise AssertionError("stop acknowledgement failed")
                            stage = "stop acknowledged"
                    except (HTTPError, TimeoutError, OSError):
                        # Startup endpoints may transiently be unavailable; a stop
                        # request itself is not retried, even on HTTP failure.
                        if stop_attempted:
                            error = "stop request failed"
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
            if result and not stop_attempted:
                error = "sampling finished without authenticated stop"
            if result and (result["status"] not in {"stopped", "manual_stop"}
                           or not 0 < result["draws"] < 200_000 or result["cache_hit"]):
                error = "partial result/status/cache assertion failed"
            if result is None and error is None:
                error = f"deadline exceeded at {stage}"
        finally:
            # The parent never waits for arbitrary child teardown or remote calls.
            if error and child.is_alive():
                if hasattr(signal, "SIGUSR1"):
                    os.kill(child.pid, signal.SIGUSR1)
                    time.sleep(min(0.05, max(0, cleanup_deadline - time.monotonic())))
                stack = Path(stack_path).read_text(encoding="utf-8") or stack or "child stack unavailable (signal handler not installed or process exited)"
                if identities.get("stop_url") and not stop_attempted:
                    stop_attempted = True
                    try:
                        _bounded_http(http, _stop_url(identities["stop_url"], identities["token"]),
                                      method="POST", deadline=cleanup_deadline)
                    except Exception:
                        pass
            # Terminate child before deleting its app/Dict/Volume so it cannot
            # recreate resources after cleanup. Grace only if already settling.
            child.join(timeout=min(stop_grace if not error else 0.1,
                                   max(0, cleanup_deadline - time.monotonic())))
            if child.is_alive():
                child.terminate()
                child.join(timeout=min(1, max(0, cleanup_deadline - time.monotonic())))
            if child.is_alive():
                child.kill()
                child.join(timeout=min(1, max(0, cleanup_deadline - time.monotonic())))
            reaped = not child.is_alive()
            cleanup_parent, cleanup_conn = ctx.Pipe(duplex=False)
            targets = {k: identities[k] for k in ("app_id", "control_dict", "volume")
                       if k in identities}
            cleaner = ctx.Process(target=cleanup_target, args=(cleanup_conn, targets))
            cleaner.start()
            cleanup_conn.close()
            cleaner.join(timeout=max(0, cleanup_deadline - time.monotonic()))
            if cleaner.is_alive():
                cleaner.terminate()
                cleaner.join(timeout=min(1, max(0, cleanup_deadline - time.monotonic())))
            if cleaner.is_alive():
                cleaner.kill()
                cleaner.join(timeout=min(1, max(0, cleanup_deadline - time.monotonic())))
            if cleanup_parent.poll(0):
                try:
                    cleanup = cleanup_parent.recv()
                except EOFError:
                    cleanup = {"error": "cleanup process exited without verification"}
            else:
                cleanup = {"error": "cleanup deadline exceeded or process exited"}
            cleanup_parent.close()
            parent.close()
    return StopOutcome(result, stage, stack, error, cleanup,
                       targets, time.monotonic() - started, reaped)


def assert_stop_outcome(outcome):
    # Never include token-bearing URLs in assertion output.
    missing = [key for key in ("app_inactive", "dict_absent", "volume_absent")
               if ((key == "app_inactive" and "app_id" in outcome.identities)
                   or (key.replace("_absent", "") in outcome.cleanup))
               and outcome.cleanup.get(key) is not True]
    if "app_id" not in outcome.identities:
        missing.append("app identity unavailable: remote app state unverified")
    if outcome.error or not outcome.reaped or missing or outcome.cleanup.get("error"):
        raise AssertionError(
            f"manual-stop smoke failed: stage={outcome.stage}; error={outcome.error}; "
            f"cleanup={outcome.cleanup}; remote leak/unverified={missing or outcome.cleanup.get('error')}; "
            f"app_id={outcome.identities.get('app_id', 'not recorded')}; "
            f"child reaped={outcome.reaped}; child stack:\n{outcome.stack}"
        )
