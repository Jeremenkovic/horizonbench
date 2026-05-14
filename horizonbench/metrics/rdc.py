"""RDC — Reliability Decay Curve.

RDC(k) = mean_i S(k, i)   where S ∈ {0, 1}

Single-number summary: RDC_AUC (area under curve, trapezoidal, normalised to [0,1]).
"""

from __future__ import annotations

import numpy as np


def rdc(successes_by_k: dict[int, list[int]]) -> dict[int, float]:
    """Compute per-k mean success rate.

    Parameters
    ----------
    successes_by_k: {k: [0/1, ...]}  — binary success flags per k.

    Returns
    -------
    {k: mean_success_rate}
    """
    return {k: float(np.mean(vals)) for k, vals in successes_by_k.items() if vals}


def rdc_auc(curve: dict[int, float]) -> float:
    """Trapezoidal AUC of the RDC curve, normalised to [0, 1].

    The x-axis spans from min(k) to max(k); y ∈ [0, 1].
    Returns 0.0 if fewer than 2 points.
    """
    if len(curve) < 2:
        return float(np.mean(list(curve.values()))) if curve else 0.0
    ks = sorted(curve)
    ys = [curve[k] for k in ks]
    raw_auc = float(np.trapezoid(ys, ks))
    k_range = ks[-1] - ks[0]
    return raw_auc / k_range if k_range > 0 else float(ys[0])
