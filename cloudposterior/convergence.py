"""Compatibility exports for the pymc-monitor convergence policy and diagnostics.

New integrations should import :mod:`pymc_monitor.convergence` directly. The
cloud API also writes its historical report attribute for older readers.
"""

from pymc_monitor.convergence import (  # noqa: F401
    PRESETS,
    REPORT_ATTR as MONITOR_REPORT_ATTR,
    SUPERCHAIN_ATTR as MONITOR_SUPERCHAIN_ATTR,
    Controller,
    Draws,
    _jsonable,
    assess,
    attach_report as _attach_monitor_report,
    bfmi,
    check_convergence,
    diagnose,
    draws_from_arrays,
    draws_from_idata,
    effective_ess_target,
    element_labels,
    ess_basic,
    ess_bulk,
    ess_mean,
    ess_quantile,
    ess_tail,
    failing,
    make_report,
    mcse_quantile,
    nested_rhat,
    normalize_target,
    projected_draws,
    read_report,
    rhat_rank,
    rhat_threshold,
    split_chains,
    summarize,
    verdict_message,
    z_scale,
)

REPORT_ATTR = "cloudposterior_convergence"
SUPERCHAIN_ATTR = "cloudposterior_superchains"


def attach_report(idata, report: dict) -> None:
    """Write both the monitor report and the historical cloudposterior alias."""
    from cloudposterior._idata import get_group

    _attach_monitor_report(idata, report)
    posterior = get_group(idata, "posterior")
    if posterior is not None and MONITOR_REPORT_ATTR in posterior.attrs:
        posterior.attrs[REPORT_ATTR] = posterior.attrs[MONITOR_REPORT_ATTR]
