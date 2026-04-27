"""Pytest configuration: split free local tests from paid Modal tests.

Tests marked ``@pytest.mark.modal`` actually hit the Modal API and incur
cloud costs. They are skipped by default. To run them::

    uv run pytest tests/ -v --run-modal
    uv run pytest tests/test_modal_e2e.py -v --run-modal   # Modal-only

To run only the free tests (the default)::

    uv run pytest tests/ -v
    uv run pytest tests/ -v -m "not modal"   # explicit equivalent
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-modal",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.modal (incurs Modal cloud costs).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "modal: tests that hit Modal and cost real money. Skipped unless --run-modal is set.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-modal"):
        return
    skip_modal = pytest.mark.skip(reason="needs --run-modal (incurs Modal cloud costs)")
    for item in items:
        if "modal" in item.keywords:
            item.add_marker(skip_modal)
