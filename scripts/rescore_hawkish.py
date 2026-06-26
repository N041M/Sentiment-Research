#!/usr/bin/env python3
"""(Re)score central-bank speeches with hawkish/dovish stance.

Hawkish/dovish is a MONETARY-POLICY signal, so it is only applied to central-bank
speeches (source_type='speech'). Any hawkish score previously written to other
document types (executive orders, UN news, etc.) is nulled — applying a monetary
lexicon to a sanctions proclamation is a category error and produced a spurious
−1.0 cluster.

Uses FOMC-RoBERTa when available, otherwise the smoothed rule-based lexicon.
Re-scores all speeches each run so lexicon changes take effect.

By default it also:
  1. snapshots the current speech scores into `hawkish_score_snapshot` BEFORE
     overwriting (guarded to capture the lexicon baseline exactly once), so the
     thesis can show a lexicon-vs-model comparison;
  2. backfills `sentiment_signal.hawkish_score` from `statement_analysis`;
  3. prints a lexicon-vs-model agreement report (Pearson r, label agreement,
     sign flips) when a baseline snapshot exists.

Run from project root with venv active:
    python scripts/rescore_hawkish.py
    python scripts/rescore_hawkish.py --no-snapshot --no-backfill   # bare rescore
"""

import argparse
import sys

sys.path.insert(0, ".")

from collections import Counter

import numpy as np
from loguru import logger
from sqlalchemy import select, text, update

from sentiment_signal.db.models import Statement, StatementAnalysis
from sentiment_signal.db.session import SessionLocal

# Auxiliary (non-core) table created on demand; holds point-in-time snapshots of the
# stance scores so the lexicon baseline can be compared against the FOMC-RoBERTa run.
SNAPSHOT_DDL = """
CREATE TABLE IF NOT EXISTS hawkish_score_snapshot (
    id BIGSERIAL PRIMARY KEY,
    statement_id UUID NOT NULL,
    hawkish_score DOUBLE PRECISION,
    hawkish_label VARCHAR(10),
    method VARCHAR(40) NOT NULL,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def snapshot_current_scores(session, method: str) -> None:
    """Copy current speech stance scores into the snapshot table, once per `method`.

    Guarded: if a snapshot already exists for `method`, skip — so re-running with the
    model can never clobber the original lexicon baseline we want to compare against.
    """
    session.execute(text(SNAPSHOT_DDL))
    already = session.execute(
        text("SELECT 1 FROM hawkish_score_snapshot WHERE method = :m LIMIT 1"),
        {"m": method},
    ).first()
    if already:
        logger.info(f"Snapshot '{method}' already exists — keeping it, not re-snapshotting")
        session.commit()
        return
    n = session.execute(
        text(
            """
            INSERT INTO hawkish_score_snapshot (statement_id, hawkish_score, hawkish_label, method)
            SELECT sa.statement_id, sa.hawkish_score, sa.hawkish_label, :m
            FROM statement_analysis sa
            JOIN statements s ON s.id = sa.statement_id
            WHERE s.source_type = 'speech' AND sa.hawkish_score IS NOT NULL
            """
        ),
        {"m": method},
    ).rowcount
    session.commit()
    logger.info(f"Snapshotted {n} speech stance scores as baseline '{method}'")


def backfill_sentiment_signal(session) -> None:
    """Propagate statement_analysis.hawkish_score onto sentiment_signal (NULL for
    non-speeches, since those were cleared on statement_analysis above)."""
    n = session.execute(
        text(
            """
            UPDATE sentiment_signal ss
            SET hawkish_score = sa.hawkish_score
            FROM statement_analysis sa
            WHERE sa.statement_id = ss.statement_id
            """
        )
    ).rowcount
    session.commit()
    logger.info(f"Backfilled hawkish_score onto {n} sentiment_signal rows")


def report_comparison(session, method: str) -> None:
    """Print lexicon-vs-model agreement on speeches present in both the baseline
    snapshot and the freshly written statement_analysis scores."""
    rows = session.execute(
        text(
            """
            SELECT snap.hawkish_score AS old_s, snap.hawkish_label AS old_l,
                   sa.hawkish_score   AS new_s, sa.hawkish_label   AS new_l
            FROM hawkish_score_snapshot snap
            JOIN statement_analysis sa ON sa.statement_id = snap.statement_id
            WHERE snap.method = :m
              AND snap.hawkish_score IS NOT NULL
              AND sa.hawkish_score IS NOT NULL
            """
        ),
        {"m": method},
    ).all()
    if not rows:
        logger.info(f"No overlapping rows to compare against baseline '{method}' — skipping report")
        return

    old = np.array([r.old_s for r in rows], dtype=float)
    new = np.array([r.new_s for r in rows], dtype=float)
    label_agree = sum(1 for r in rows if r.old_l == r.new_l) / len(rows)
    sign_flips = int(np.sum(np.sign(old) != np.sign(new)))
    pearson = (
        float(np.corrcoef(old, new)[0, 1]) if old.std() > 0 and new.std() > 0 else float("nan")
    )

    logger.info(f"=== Stance comparison vs baseline '{method}' (n={len(rows)}) ===")
    logger.info(f"Pearson r(old, new) = {pearson:.3f}")
    logger.info(f"Label agreement     = {label_agree:.1%}")
    logger.info(f"Sign flips          = {sign_flips} ({sign_flips / len(rows):.1%})")
    logger.info(f"Mean score: baseline {old.mean():+.3f} -> new {new.mean():+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-snapshot", action="store_true", help="skip the baseline snapshot")
    parser.add_argument(
        "--no-backfill", action="store_true", help="skip the sentiment_signal backfill"
    )
    parser.add_argument(
        "--snapshot-method",
        default="lexicon_baseline",
        help="tag for the baseline snapshot (default: lexicon_baseline)",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        # 1. Null hawkish scores wrongly applied to non-speech documents.
        cleared = session.execute(
            update(StatementAnalysis)
            .where(
                StatementAnalysis.statement_id.in_(
                    select(Statement.id).where(Statement.source_type != "speech")
                )
            )
            .values(hawkish_score=None, hawkish_label=None)
        ).rowcount
        session.commit()
        logger.info(f"Cleared hawkish scores on {cleared} non-speech statements")

        # 2. Snapshot the current (pre-overwrite) speech scores as the baseline.
        if not args.no_snapshot:
            snapshot_current_scores(session, args.snapshot_method)

        # 3. Score all central-bank speeches.
        speeches = session.execute(
            select(Statement.id, Statement.raw_text)
            .join(StatementAnalysis, StatementAnalysis.statement_id == Statement.id)
            .where(Statement.source_type == "speech")
        ).all()
        logger.info(f"Speeches to score: {len(speeches)}")
        if not speeches:
            return

        ids = [str(r.id) for r in speeches]
        texts = [r.raw_text for r in speeches]

        try:
            from sentiment_signal.nlp.pipeline import NLPPipeline

            logger.info(f"Scoring {len(texts)} speeches with FOMC-RoBERTa")
            results = NLPPipeline().score_hawkish_dovish(texts)
            method = "FOMC-RoBERTa"
        except Exception as exc:
            logger.warning(
                f"FOMC-RoBERTa unavailable ({exc.__class__.__name__}), "
                f"falling back to rule-based lexicon"
            )
            from sentiment_signal.nlp.hawkish_lexicon import score_batch

            results = score_batch(texts)
            method = "lexicon"

        for stmt_id, result in zip(ids, results):
            analysis = session.scalar(
                select(StatementAnalysis).where(StatementAnalysis.statement_id == stmt_id)
            )
            if analysis:
                analysis.hawkish_score = result["hawkish_score"]
                analysis.hawkish_label = result["hawkish_label"]
        session.commit()
        logger.info(f"Done — scored {len(results)} speeches via {method}")
        logger.info(f"Label distribution: {dict(Counter(r['hawkish_label'] for r in results))}")

        # 4. Backfill sentiment_signal from the freshly written statement_analysis scores.
        if not args.no_backfill:
            backfill_sentiment_signal(session)

        # 5. Lexicon-vs-model comparison (only meaningful when the model actually ran).
        if not args.no_snapshot and method != args.snapshot_method:
            report_comparison(session, args.snapshot_method)
    finally:
        session.close()


if __name__ == "__main__":
    main()
