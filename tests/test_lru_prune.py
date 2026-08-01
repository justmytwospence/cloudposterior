"""Unit-test the Volume LRU prune logic without hitting Modal.

The prune helper lives on ModalEnvironment but it only touches
``self._volume.listdir`` and ``self._volume.remove_file`` -- both easy to
fake. This avoids the Modal cost of actually populating a real Volume.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cloudposterior.backends.modal_backend import (
    ModalEnvironment,
    _PAYLOAD_KEEP_PER_MODEL,
)


def _entry(name: str, mtime: float):
    """Build a fake Modal FileEntry."""
    return SimpleNamespace(path=name, mtime=mtime)


def _env_with_fake_volume(entries):
    volume = MagicMock()
    volume.listdir.return_value = entries
    env = ModalEnvironment.__new__(ModalEnvironment)
    env._volume = volume
    return env, volume


def test_prune_no_op_when_under_threshold():
    entries = [_entry(f"payload-{i:02x}.bin", float(i)) for i in range(_PAYLOAD_KEEP_PER_MODEL)]
    env, volume = _env_with_fake_volume(entries)

    env._prune_old_payloads("model_x/payload-newest.bin")

    volume.remove_file.assert_not_called()


def test_prune_removes_oldest_when_over_threshold():
    # 8 payloads, mtime ascending. With keep=5, the 3 oldest (mtimes 0,1,2)
    # should be removed.
    entries = [_entry(f"payload-{i:02x}.bin", float(i)) for i in range(8)]
    env, volume = _env_with_fake_volume(entries)

    env._prune_old_payloads("model_x/payload-newest.bin")

    removed = sorted(call.args[0] for call in volume.remove_file.call_args_list)
    assert removed == [
        "/model_x/payload-00.bin",
        "/model_x/payload-01.bin",
        "/model_x/payload-02.bin",
    ]


def test_prune_ignores_non_payload_files():
    """Stray non-payload files in the model directory must not be deleted."""
    entries = [
        _entry("payload-aa.bin", 10.0),
        _entry("payload-bb.bin", 11.0),
        _entry("payload-cc.bin", 12.0),
        _entry("README.txt", 0.1),  # not a payload
        _entry("scratch.bin", 0.2),  # also not a payload (wrong prefix)
    ]
    env, volume = _env_with_fake_volume(entries)

    env._prune_old_payloads("model_x/payload-cc.bin")

    # Only 3 payloads, well under the keep threshold -- no removal.
    volume.remove_file.assert_not_called()


def test_prune_swallows_listdir_errors():
    """Volume listdir failures (e.g. transient network) must not crash the
    upload path -- prune is best-effort."""
    env = ModalEnvironment.__new__(ModalEnvironment)
    env._volume = MagicMock()
    env._volume.listdir.side_effect = RuntimeError("network blip")

    # Should not raise.
    env._prune_old_payloads("model_x/payload-newest.bin")
    env._volume.remove_file.assert_not_called()


def test_prune_swallows_remove_errors():
    """Individual remove_file failures must not abort the rest of the prune."""
    entries = [_entry(f"payload-{i:02x}.bin", float(i)) for i in range(8)]
    env, volume = _env_with_fake_volume(entries)
    volume.remove_file.side_effect = [RuntimeError("first fail"), None, None]

    # Should not raise.
    env._prune_old_payloads("model_x/payload-newest.bin")
    assert volume.remove_file.call_count == 3


def test_prune_handles_dir_prefixed_entry_paths():
    """Some listdir shapes return dir-prefixed paths; prune must still match
    payloads and not double the directory in the remove path."""
    entries = [_entry(f"model_x/payload-{i:02x}.bin", float(i)) for i in range(7)]
    env, volume = _env_with_fake_volume(entries)

    env._prune_old_payloads("model_x/payload-newest.bin")

    removed = sorted(call.args[0] for call in volume.remove_file.call_args_list)
    assert removed == ["/model_x/payload-00.bin", "/model_x/payload-01.bin"]


def test_teardown_deletes_control_dict(monkeypatch):
    """The per-provision cp-dash-* Dict is a named persistent Modal object;
    teardown must delete it or orphans accumulate."""
    import contextlib
    from unittest.mock import MagicMock, patch

    from cloudposterior.backends.modal_backend import ModalEnvironment

    env = ModalEnvironment.__new__(ModalEnvironment)
    env._exit_stack = contextlib.ExitStack()
    env._dashboard_dict_name = "cp-dash-deadbeef"
    env._running = True

    fake_modal = MagicMock()
    with patch.dict("sys.modules", {"modal": fake_modal}):
        env.teardown()

    fake_modal.Dict.objects.delete.assert_called_once_with("cp-dash-deadbeef")
    assert env._running is False


def test_teardown_without_control_dict_skips_delete():
    import contextlib
    from unittest.mock import MagicMock, patch

    from cloudposterior.backends.modal_backend import ModalEnvironment

    env = ModalEnvironment.__new__(ModalEnvironment)
    env._exit_stack = contextlib.ExitStack()
    env._dashboard_dict_name = None
    env._running = True

    fake_modal = MagicMock()
    with patch.dict("sys.modules", {"modal": fake_modal}):
        env.teardown()

    fake_modal.Dict.objects.delete.assert_not_called()


def test_prune_never_deletes_the_payload_just_uploaded():
    """Volume mtimes have one-second granularity, so a tie could otherwise
    sort the fresh payload into the stale tail -- deleting the file the next
    submit is about to load."""
    # Every payload shares an mtime, and the fresh one sorts last by name.
    entries = [_entry(f"payload-{i:02x}.bin", 100.0) for i in range(8)]
    entries.append(_entry("payload-zz.bin", 100.0))
    env, volume = _env_with_fake_volume(entries)

    env._prune_old_payloads("model_x/payload-zz.bin")

    removed = [call.args[0] for call in volume.remove_file.call_args_list]
    assert "/model_x/payload-zz.bin" not in removed
    # The kept payload occupies one of the N slots.
    assert len(entries) - len(removed) == _PAYLOAD_KEEP_PER_MODEL
