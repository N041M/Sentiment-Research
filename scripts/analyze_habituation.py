#!/usr/bin/env python3
"""Habituation test (architecture_v2 engineered baseline): does a market's |response|
to a speech shrink as recent same-topic statements accumulate ("the market is used to
it")?

Honest by construction:
  - relevance-aware pairing (speaker -> its mandated index only; no geography-blind
    attribution),
  - point-in-time abnormal response (return / prior-window vol, prior data only),
  - point-in-time habituation pressure (recency-weighted prior same-topic statements),
  - day-clustered standard errors (one market-day is not 16 independent points),
  - a permutation/placebo check (shuffle the pressure; the real slope should be an
    outlier).

Daily data only, so a day's move is a noisy proxy for one speech's impact — read this as
suggestive, not causal proof. Run:  python scripts/analyze_habituation.py
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from loguru import logger
from scipy.stats import spearmanr

from sentiment_signal.db.session import SessionLocal
from sentiment_signal.features.geography import primary_market_for_institution
from sentiment_signal.features.market_response import market_daily_returns
from sentiment_signal.features.novelty import recent_similar_pressure
from sentiment_signal.features.sequence import load_event_sequence


def _market_abnormal(session, symbols: list[str]) -> pd.DataFrame:
    """Per-market point-in-time abnormal return (shared with the dashboard pairing)."""
    df = market_daily_returns(session, symbols)
    return df[["market", "date", "abnormal"]].dropna(subset=["abnormal"])


def _perm_pressure_coef(
    y: np.ndarray, press: np.ndarray, ctrl: np.ndarray, n_perm: int = 2000
) -> float:
    """Two-sided permutation p-value for the pressure coefficient, controlling for ctrl,
    by shuffling the pressure column."""
    x = np.column_stack([np.ones_like(press), press, ctrl])
    obs = np.linalg.lstsq(x, y, rcond=None)[0][1]
    rng = np.random.default_rng(42)
    null = np.empty(n_perm)
    for k in range(n_perm):
        xp = x.copy()
        xp[:, 1] = rng.permutation(press)
        null[k] = np.linalg.lstsq(xp, y, rcond=None)[0][1]
    return float((np.abs(null) >= abs(obs)).mean())


def main() -> None:
    session = SessionLocal()
    logger.info("Loading in-domain speeches (source_type='speech')…")
    events = load_event_sequence(session, source_types=("speech",))
    pressure = recent_similar_pressure(events)

    recs = []
    for e, pr in zip(events, pressure):
        market = primary_market_for_institution(e.institution)
        if market is None or e.sentiment_score is None:
            continue
        recs.append(
            {
                "market": market,
                "date": pd.Timestamp(e.timestamp).tz_convert("UTC").tz_localize(None).normalize(),
                "pressure": pr,
                "abs_sent": abs(e.sentiment_score),
            }
        )
    sdf = pd.DataFrame(recs)
    logger.info(
        f"{len(sdf)} speeches with a relevant market; markets={sorted(sdf['market'].unique())}"
    )

    abn = _market_abnormal(session, sorted(sdf["market"].unique()))
    session.close()

    parts = []
    for m, g in sdf.groupby("market"):
        pm = abn[abn.market == m][["date", "abnormal"]].sort_values("date")
        merged = pd.merge_asof(
            g.sort_values("date"), pm, on="date", direction="forward", tolerance=pd.Timedelta("5D")
        )
        parts.append(merged)
    df = pd.concat(parts).dropna(subset=["abnormal"])
    df["abs_resp"] = df["abnormal"].abs()
    for c in ("pressure", "abs_sent"):
        df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std()

    print(f"\nN={len(df)} statement->market-day pairs, {df['date'].nunique()} distinct days")
    r, p = spearmanr(df["pressure"], df["abs_resp"])
    print(f"raw Spearman(pressure, |response|): r={r:+.4f} p={p:.4f}  (negative = habituation)")

    model = smf.ols("abs_resp ~ pressure_z + abs_sent_z", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["date"]}
    )
    b = model.params["pressure_z"]
    ci = model.conf_int().loc["pressure_z"]
    print("\nOLS  |response| ~ pressure + |sentiment|   (day-clustered SE):")
    print(
        f"  pressure coef = {b:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={model.pvalues['pressure_z']:.4f}"
    )
    print(
        "  (negative => more recent same-topic statements -> smaller market response = habituation)"
    )

    p_perm = _perm_pressure_coef(
        df["abs_resp"].to_numpy(), df["pressure_z"].to_numpy(), df["abs_sent_z"].to_numpy()
    )
    print(f"  permutation p (shuffle pressure): {p_perm:.4f}")


if __name__ == "__main__":
    main()
