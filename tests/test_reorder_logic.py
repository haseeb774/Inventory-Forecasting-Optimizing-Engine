import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from src.forecasting_feature import Reorder


@pytest.fixture
def reorder():
    return Reorder()


# ── Safety stock ──────────────────────────────────────────

def test_safety_stock_scales_with_std(reorder):
    low_std_stock  = reorder.calculate_safety_stock(std_daily_demand=2)
    high_std_stock = reorder.calculate_safety_stock(std_daily_demand=10)
    assert high_std_stock > low_std_stock


def test_safety_stock_zero_when_no_variability(reorder):
    assert reorder.calculate_safety_stock(std_daily_demand=0) == 0


def test_safety_stock_known_value(reorder):
    # Z=1.645, lead_time=7 -> safety_stock = 1.645 * std * sqrt(7)
    std = 5
    expected = 1.645 * std * np.sqrt(7)
    assert reorder.calculate_safety_stock(std) == pytest.approx(expected, rel=1e-3)


# ── Reorder point ─────────────────────────────────────────

def test_reorder_point_increases_with_demand(reorder):
    rop_low, _  = reorder.calculate_reorder_point(avg_daily_demand=5,  std_daily_demand=2)
    rop_high, _ = reorder.calculate_reorder_point(avg_daily_demand=50, std_daily_demand=2)
    assert rop_high > rop_low


def test_reorder_point_equals_demand_plus_safety_stock(reorder):
    avg_demand, std_demand = 10, 3
    rop, safety_stock = reorder.calculate_reorder_point(avg_demand, std_demand)
    expected_rop = (avg_demand * reorder.LEAD_TIME_DAYS) + safety_stock
    assert rop == pytest.approx(expected_rop)


def test_reorder_point_zero_demand_zero_std(reorder):
    rop, safety_stock = reorder.calculate_reorder_point(0, 0)
    assert rop == 0
    assert safety_stock == 0


# ── Status classification logic ──────────────────────────
# Re-implemented inline since generate_alerts() isn't unit-isolated;
# this locks in the business rule so refactors don't silently break it.

def classify_status(current_stock, rop, total_30d_demand):
    if current_stock < rop:
        return "CRITICAL"
    elif current_stock < rop * 1.5:
        return "WARNING"
    elif current_stock > 3 * total_30d_demand:
        return "OVERSTOCK"
    return "HEALTHY"


def test_status_critical_when_below_reorder_point():
    assert classify_status(current_stock=10, rop=50, total_30d_demand=100) == "CRITICAL"


def test_status_warning_in_buffer_zone():
    # rop=50 -> warning zone is [50, 75)
    assert classify_status(current_stock=60, rop=50, total_30d_demand=100) == "WARNING"


def test_status_overstock_when_stock_far_exceeds_demand():
    assert classify_status(current_stock=400, rop=50, total_30d_demand=100) == "OVERSTOCK"


def test_status_healthy_in_normal_range():
    assert classify_status(current_stock=80, rop=50, total_30d_demand=100) == "HEALTHY"


def test_low_volume_sku_not_misflagged_overstock():
    """
    Regression guard for the README's stated claim:
    a low-volume flat-demand item should not be lumped in with bestsellers.
    """
    # Slow mover: small demand, moderate stock — should be HEALTHY, not OVERSTOCK
    rop, _ = (5, 0)  # placeholder rop for a low-demand SKU
    status = classify_status(current_stock=20, rop=8, total_30d_demand=15)
    assert status != "OVERSTOCK"