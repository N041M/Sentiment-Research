"""Unit tests for the relevance-aware, point-in-time statement->market pairing.

These cover the leakage-critical helpers that replace the discredited event_context
attribution: returns must use only prior days (no look-ahead), and a statement must be
paired with a market move that happens strictly AFTER it.
"""

import numpy as np
import pandas as pd
import pytest

from sentiment_signal.features.market_response import (
    compute_daily_returns,
    pair_statements_with_returns,
)


def _prices(market: str, dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"market": market, "date": pd.to_datetime(dates), "close": closes})


class TestComputeDailyReturns:
    def test_pct_change(self):
        df = _prices("M", ["2020-01-01", "2020-01-02", "2020-01-03"], [100, 110, 99])
        out = compute_daily_returns(df, vol_window=2).set_index("date")
        assert np.isnan(out.loc["2020-01-01", "ret_pct"])
        assert out.loc["2020-01-02", "ret_pct"] == pytest.approx(10.0)
        assert out.loc["2020-01-03", "ret_pct"] == pytest.approx(-10.0)

    def test_abnormal_is_point_in_time(self):
        # abnormal at row i may only use returns strictly before i (prior-vol, shifted).
        df = _prices(
            "M",
            [f"2020-01-{d:02d}" for d in range(1, 7)],
            [100, 101, 103, 100, 104, 102],
        )
        out = compute_daily_returns(df, vol_window=2).reset_index(drop=True)
        # rows 0-2 cannot have a prior 2-day vol estimate -> NaN (no future leak)
        assert out["abnormal"][:3].isna().all()
        assert np.isfinite(out["abnormal"].iloc[3])

    def test_empty(self):
        out = compute_daily_returns(pd.DataFrame(columns=["market", "date", "close"]))
        assert out.empty


def _rets(market: str, dates: list[str], r: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": market,
            "date": pd.to_datetime(dates),
            "ret_pct": r,
            "abnormal": r,
        }
    )


def _stmts(market: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"market": market, "date": pd.to_datetime(dates)})


class TestPairStatementsWithReturns:
    def test_pairs_with_next_day_strictly_after(self):
        sdf = _stmts("M", ["2020-01-02", "2020-01-06"])
        rets = _rets("M", ["2020-01-02", "2020-01-03", "2020-01-07"], [1.0, 2.0, 3.0])
        out = pair_statements_with_returns(sdf, rets, tolerance_days=5).sort_values("date")
        # 01-02 -> 01-03 (NOT same-day 01-02); 01-06 -> 01-07
        assert list(out["ret_pct"]) == [2.0, 3.0]

    def test_no_lookahead_never_pairs_with_prior_day(self):
        # A statement after every market day must get NaN, never an earlier return.
        sdf = _stmts("M", ["2020-01-10"])
        rets = _rets("M", ["2020-01-02", "2020-01-03"], [1.0, 2.0])
        out = pair_statements_with_returns(sdf, rets, tolerance_days=5)
        assert out["ret_pct"].isna().all()

    def test_tolerance_caps_the_gap(self):
        sdf = _stmts("M", ["2020-01-02"])
        rets = _rets("M", ["2020-01-20"], [9.0])  # 18 days later, beyond tolerance
        out = pair_statements_with_returns(sdf, rets, tolerance_days=5)
        assert out["ret_pct"].isna().all()

    def test_strictly_after_false_allows_same_day(self):
        sdf = _stmts("M", ["2020-01-02"])
        rets = _rets("M", ["2020-01-02", "2020-01-03"], [1.0, 2.0])
        out = pair_statements_with_returns(sdf, rets, tolerance_days=5, strictly_after=False)
        assert out["ret_pct"].iloc[0] == pytest.approx(1.0)

    def test_empty_statements(self):
        out = pair_statements_with_returns(
            pd.DataFrame(columns=["market", "date"]), _rets("M", ["2020-01-02"], [1.0])
        )
        assert out.empty
        assert "ret_pct" in out.columns
