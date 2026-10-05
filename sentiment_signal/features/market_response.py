"""Relevance-aware, point-in-time pairing of statements with subsequent market moves.

The clean replacement for the discredited `event_context` window-attribution (see
architecture_v2 "Rework" and build-log §13.26): instead of linking every market event
to every signal in a 48 h window across all 16 markets (geography-blind, pseudo-
replicated), each in-domain statement is paired with the *next* trading-day move of the
*one* market its speaker plausibly affects (`primary_market_for_institution`), strictly
after the statement (no look-ahead). One row per statement — no 16× replication.

Daily data only, so a day's move is a noisy proxy for one statement's impact: treat the
output as exploratory, not causal. The pure helpers (`compute_daily_returns`,
`pair_statements_with_returns`) are unit-tested; the DB wrappers just feed them.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sqlalchemy import select

from sentiment_signal.db.models import PriceData
from sentiment_signal.features.geography import primary_market_for_institution
from sentiment_signal.features.sequence import load_event_sequence

DEFAULT_VOL_WINDOW = 60  # trading days for the prior-volatility estimate

_RETURN_COLS = ["market", "date", "ret_pct", "abnormal"]


def compute_daily_returns(
    price_df: pd.DataFrame, *, vol_window: int = DEFAULT_VOL_WINDOW
) -> pd.DataFrame:
    """Per-market daily % return and point-in-time abnormal return.

    `price_df` has columns market, date (tz-naive, normalised), close. `abnormal` =
    return / (rolling `vol_window`-day std of return, shifted one day so only PRIOR
    days inform it — no look-ahead). Returns columns market, date, ret_pct, abnormal.
    """
    if price_df.empty:
        return pd.DataFrame(columns=_RETURN_COLS)
    parts = []
    for _, g in price_df.groupby("market"):
        g = g.sort_values("date").copy()
        ret = g["close"].astype(float).pct_change() * 100
        prior_std = ret.rolling(vol_window).std().shift(1)  # prior days only
        g["ret_pct"] = ret
        g["abnormal"] = ret / prior_std
        parts.append(g[_RETURN_COLS])
    return pd.concat(parts, ignore_index=True)


def pair_statements_with_returns(
    sdf: pd.DataFrame,
    rets: pd.DataFrame,
    *,
    tolerance_days: int = 5,
    strictly_after: bool = True,
) -> pd.DataFrame:
    """As-of join each statement to the first market return on/after its date.

    With `strictly_after=True` (default) a statement is paired with the next trading
    day STRICTLY after it — the move is wholly subsequent to the statement, the
    no-look-ahead guarantee. `tolerance_days` caps the gap (a statement with no market
    day within the window gets NaN ret_pct/abnormal). One row per input statement.
    """
    if sdf.empty:
        return sdf.assign(ret_pct=pd.Series(dtype=float), abnormal=pd.Series(dtype=float))
    tol = pd.Timedelta(days=int(tolerance_days))
    parts = []
    for market, g in sdf.groupby("market"):
        pm = rets[rets["market"] == market][["date", "ret_pct", "abnormal"]].sort_values("date")
        left = g.sort_values("date")
        if pm.empty:
            merged = left.assign(ret_pct=float("nan"), abnormal=float("nan"))
        else:
            merged = pd.merge_asof(
                left,
                pm,
                on="date",
                direction="forward",
                allow_exact_matches=not strictly_after,
                tolerance=tol,
            )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def market_daily_returns(
    session, symbols: Sequence[str], *, vol_window: int = DEFAULT_VOL_WINDOW
) -> pd.DataFrame:
    """Load daily closes for `symbols` and compute returns/abnormal returns."""
    if not symbols:
        return pd.DataFrame(columns=_RETURN_COLS)
    rows = session.execute(
        select(PriceData.symbol, PriceData.timestamp, PriceData.close)
        .where(PriceData.symbol.in_(list(symbols)), PriceData.granularity == "1d")
        .order_by(PriceData.symbol, PriceData.timestamp)
    ).all()
    df = pd.DataFrame(rows, columns=["market", "ts", "close"])
    if df.empty:
        return pd.DataFrame(columns=_RETURN_COLS)
    df["date"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    return compute_daily_returns(df[["market", "date", "close"]], vol_window=vol_window)


def statement_market_pairs(
    session,
    *,
    source_types: Sequence[str] | None = ("speech",),
    tolerance_days: int = 5,
    vol_window: int = DEFAULT_VOL_WINDOW,
    strictly_after: bool = True,
) -> pd.DataFrame:
    """Relevance-aware, point-in-time statement→market-move pairs.

    Columns: timestamp, date, person, institution, market, sentiment_score,
    hawkish_score, ret_pct (next-day % move of the speaker's mandated market),
    abnormal (vol-normalised). Statements whose institution maps to no collected
    index, or with no sentiment_score, are dropped.
    """
    events = load_event_sequence(session, source_types=source_types)
    recs = []
    for e in events:
        market = primary_market_for_institution(e.institution)
        if market is None or e.sentiment_score is None:
            continue
        ts = pd.Timestamp(e.timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        recs.append(
            {
                "timestamp": ts,
                "person": e.person,
                "institution": e.institution,
                "market": market,
                "sentiment_score": float(e.sentiment_score),
                "hawkish_score": (None if e.hawkish_score is None else float(e.hawkish_score)),
                "sentiment_label": e.sentiment_label,
                "hawkish_label": e.hawkish_label,
            }
        )
    sdf = pd.DataFrame(recs)
    if sdf.empty:
        return sdf
    sdf["date"] = sdf["timestamp"].dt.tz_localize(None).dt.normalize()
    rets = market_daily_returns(session, sorted(sdf["market"].unique()), vol_window=vol_window)
    return pair_statements_with_returns(
        sdf, rets, tolerance_days=tolerance_days, strictly_after=strictly_after
    )
