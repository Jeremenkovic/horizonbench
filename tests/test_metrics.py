"""Tests for RDC, VAF, GDS, MOP metrics."""

from __future__ import annotations

import pytest

from horizonbench.metrics import gds, gds_overall, mop, rdc, rdc_auc, vaf


# ── RDC ──────────────────────────────────────────────────────────────────────


def test_rdc_all_success():
    curve = rdc({5: [1, 1, 1], 10: [1, 1, 1]})
    assert curve[5] == pytest.approx(1.0)
    assert curve[10] == pytest.approx(1.0)


def test_rdc_all_fail():
    curve = rdc({5: [0, 0, 0]})
    assert curve[5] == pytest.approx(0.0)


def test_rdc_mixed():
    curve = rdc({5: [1, 0, 1, 0]})
    assert curve[5] == pytest.approx(0.5)


def test_rdc_empty_k_excluded():
    curve = rdc({5: [1, 1], 10: []})
    assert 5 in curve
    assert 10 not in curve


def test_rdc_auc_flat():
    curve = {5: 0.8, 10: 0.8, 20: 0.8}
    assert rdc_auc(curve) == pytest.approx(0.8)


def test_rdc_auc_decreasing():
    curve = {5: 1.0, 10: 0.5, 20: 0.0}
    auc = rdc_auc(curve)
    assert 0.0 < auc < 1.0


def test_rdc_auc_single_point():
    assert rdc_auc({5: 0.7}) == pytest.approx(0.7)


def test_rdc_auc_empty():
    assert rdc_auc({}) == 0.0


# ── VAF ──────────────────────────────────────────────────────────────────────


def test_vaf_baseline_returns_one():
    v = vaf({5: [1, 0, 1, 0, 1, 0], 10: [1, 1, 0, 0]}, baseline_k=5)
    assert v[5] == pytest.approx(1.0)


def test_vaf_no_variance_returns_none():
    # All successes at baseline → std=0 → cannot divide
    v = vaf({5: [1, 1, 1], 10: [1, 0]}, baseline_k=5)
    assert v[10] is None


def test_vaf_higher_variance_at_k():
    # baseline has some variance, k=50 has higher
    baseline = [1, 0] * 5          # std = 0.5
    k50 = [1, 0] * 5               # same std → VAF=1
    v = vaf({5: baseline, 50: k50}, baseline_k=5)
    assert v[50] == pytest.approx(1.0)


def test_vaf_insufficient_baseline():
    v = vaf({5: [1], 10: [1, 0]}, baseline_k=5)
    assert all(val is None for val in v.values())


# ── GDS ──────────────────────────────────────────────────────────────────────


def test_gds_no_failures():
    g = gds({5: [1, 1, 1]}, {5: [1.0, 1.0, 1.0]})
    assert g[5] is None


def test_gds_all_failures_zero_credit():
    g = gds({5: [0, 0]}, {5: [0.0, 0.0]})
    assert g[5] == pytest.approx(0.0)


def test_gds_partial_progress():
    g = gds({5: [0, 0, 1]}, {5: [0.4, 0.6, 1.0]})
    # Only the two failed runs: mean(0.4, 0.6) = 0.5
    assert g[5] == pytest.approx(0.5)


def test_gds_overall_averages():
    g = {5: 0.4, 10: 0.6, 20: None}
    assert gds_overall(g) == pytest.approx(0.5)


def test_gds_overall_all_none():
    assert gds_overall({5: None, 10: None}) is None


# ── MOP ──────────────────────────────────────────────────────────────────────


def test_mop_first_k():
    # drops below 0.5 at k=10
    curve = {5: 0.9, 10: 0.4, 20: 0.2}
    assert mop(curve) == 10


def test_mop_never_below_threshold():
    curve = {5: 0.8, 10: 0.7, 20: 0.6}
    assert mop(curve) is None


def test_mop_custom_threshold():
    curve = {5: 0.9, 10: 0.75, 20: 0.5}
    assert mop(curve, threshold=0.8) == 10


def test_mop_empty():
    assert mop({}) is None
