"""VAF — Variance Amplification Factor.

VAF(k) = std_i(S(k, i)) / std_i(S(5, i))

Headline value: VAF(50).
Returns None for k values with fewer than 2 observations or zero baseline std.
"""

from __future__ import annotations

import numpy as np


def vaf(successes_by_k: dict[int, list[int]], baseline_k: int = 5) -> dict[int, float | None]:
    """Compute per-k variance amplification relative to baseline_k.

    Parameters
    ----------
    successes_by_k: {k: [0/1, ...]}
    baseline_k:     the k used as the denominator (default 5).

    Returns
    -------
    {k: vaf_value_or_None}
    """
    baseline = successes_by_k.get(baseline_k, [])
    if len(baseline) < 2:
        return {k: None for k in successes_by_k}

    baseline_std = float(np.std(baseline, ddof=1))
    if baseline_std == 0.0:
        return {k: None for k in successes_by_k}

    result: dict[int, float | None] = {}
    for k, vals in successes_by_k.items():
        if len(vals) < 2:
            result[k] = None
        else:
            result[k] = float(np.std(vals, ddof=1)) / baseline_std
    return result
