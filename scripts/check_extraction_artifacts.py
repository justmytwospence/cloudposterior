"""Validate the composed wheels and sdists before isolated installation.

    python scripts/check_extraction_artifacts.py --monitor DIST --cache DIST --cloud DIST
"""

from __future__ import annotations

import argparse
import email
import tarfile
import zipfile
from pathlib import Path


def _contents(path: Path) -> tuple[set[str], bytes | None]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            metadata = next((archive.read(name) for name in names
                             if name.endswith(".dist-info/METADATA")), None)
            return names, metadata
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        member = next((name for name in names if name.endswith("/PKG-INFO")), None)
        extracted = archive.extractfile(member) if member else None
        return names, extracted.read() if extracted else None


def _check_artifact(path: Path, package: str, *, assets=()) -> None:
    names, metadata = _contents(path)
    assert any(name.endswith(f"{package}/__init__.py") or
               name == f"{package}/__init__.py" for name in names), path
    assert any(name.endswith(f"{package}/py.typed") for name in names), path
    for asset in assets:
        assert any(name.endswith(f"{package}/{asset}") for name in names), (path, asset)
    if package == "cloudposterior":
        assert not any(name.endswith(("cloudposterior/static/uPlot.iife.min.js",
                                      "cloudposterior/static/uPlot.min.css"))
                       for name in names), path
    if metadata is not None:
        msg = email.message_from_bytes(metadata)
        assert msg.get("Name", "").replace("-", "_") == package, path
        requirements = msg.get_all("Requires-Dist", [])
        for requirement in requirements:
            assert "git+" not in requirement and "file://" not in requirement, (
                path, requirement
            )
            assert "/Users/" not in requirement and ".ci-deps" not in requirement, (
                path, requirement
            )
        deps = " ".join(requirements).lower()
        if package == "cloudposterior":
            assert "pymc-monitor" in deps and "pymc-cache" in deps, path
        else:
            assert "cloudposterior" not in deps and "modal" not in deps, path
            if package == "pymc_cache":
                assert "pymc-monitor" not in deps, path
            else:
                assert "pymc-cache" not in deps, path
    print(f"checked {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for package in ("monitor", "cache", "cloud"):
        parser.add_argument(f"--{package}", type=Path, required=True)
    args = parser.parse_args()
    for directory, package, assets in (
        (args.monitor, "pymc_monitor", ("static/dashboard.html", "static/dashboard.js",
                                          "static/dashboard.css", "static/uPlot.iife.min.js",
                                          "static/uPlot.LICENSE")),
        (args.cache, "pymc_cache", ("io.py", "session.py")),
        (args.cloud, "cloudposterior", ("remote/worker.py",)),
    ):
        for suffix in ("*.whl", "*.tar.gz"):
            files = list(directory.glob(suffix))
            assert len(files) == 1, f"expected one {suffix} in {directory}, got {files}"
            _check_artifact(files[0], package, assets=assets)


if __name__ == "__main__":
    main()
