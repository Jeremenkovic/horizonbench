"""GDS — Graceful Degradation Score.

GDS(k) = mean partial credit on *failed* runs at step count k.

High GDS → failures preserve progress (good for human-handoff).
Low GDS  → failures are catastrophic (bad for autonomy).

Headline: GDS_overall = mean GDS across all k.
"""

from __future__ import annotations

import numpy as np


def gds(
    successes_by_k: dict[int, list[int]],
    partial_credits_by_k: dict[int, list[float]],
) -> dict[int, float | None]:
    """Compute per-k GDS.

    Parameters
    ----------
    successes_by_k:      {k: [0/1, ...]}
    partial_credits_by_k:{k: [credit, ...]}  — parallel to successes_by_k.

    Returns
    -------
    {k: gds_or_None}  — None if no failed runs at that k.
    """
    result: dict[int, float | None] = {}
    for k in successes_by_k:
        s_list = successes_by_k[k]
        p_list = partial_credits_by_k.get(k, [])
        failed_credits = [p for s, p in zip(s_list, p_list) if not s]
        result[k] = float(np.mean(failed_credits)) if failed_credits else None
    return result


def gds_overall(gds_curve: dict[int, float | None]) -> float | None:
    """Mean GDS across all k values that have a value."""
    vals = [v for v in gds_curve.values() if v is not None]
    return float(np.mean(vals)) if vals else None
