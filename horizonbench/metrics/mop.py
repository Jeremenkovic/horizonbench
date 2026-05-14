"""MOP — Meltdown Onset Point.

The smallest k at which RDC(k) < 0.5.
If no such k exists in the tested range, report None (equivalent to MOP > max_k).
"""

from __future__ import annotations


def mop(rdc_curve: dict[int, float], threshold: float = 0.5) -> int | None:
    """Return the smallest k where success rate drops below *threshold*.

    Parameters
    ----------
    rdc_curve: {k: mean_success_rate}  — from metrics.rdc.rdc()
    threshold: default 0.5

    Returns
    -------
    int k  — the meltdown onset point, or None if never below threshold.
    """
    for k in sorted(rdc_curve):
        if rdc_curve[k] < threshold:
            return k
    return None
