#!/usr/bin/env python3
"""Store a lexicon baseline for central-bank speeches in `hawkish_score_snapshot`.

The `lexicon_baseline` snapshot in rescore_hawkish.py copies whatever
`statement_analysis` holds when it runs. Once FOMC-RoBERTa has written its scores
there, that copy can no longer capture a lexicon run. This script therefore scores
the speech text with `hawkish_lexicon` directly and inserts the results under its own
method tag. It does not change `statement_analysis`, `sentiment_signal` or any
existing snapshot.

The script stops if rows for the tag already exist. After changing the lexicon, store
the new run under a new tag so that earlier baselines stay available for comparison.

Before writing, it prints how the lexicon scores compare with the current
`statement_analysis` scores and with the previous baseline, using the same measures
as rescore_hawkish.py.

Run from project root with venv active:
    python scripts/snapshot_lexicon_baseline.py --dry-run   # score and compare only
    python scripts/snapshot_lexicon_baseline.py             # store as lexicon_v2
    python scripts/snapshot_lexicon_baseline.py --method lexicon_v3 --previous lexicon_v2
"""

import argparse
import sys

sys.path.insert(0, ".")

from collections import Counter

import numpy as np
from loguru import logger
from sqlalchemy import select, text

from scripts.rescore_hawkish import SNAPSHOT_DDL
from sentiment_signal.db.models import Statement, StatementAnalysis
from sentiment_signal.db.session import SessionLocal
from sentiment_signal.nlp.hawkish_lexicon import score_batch

# statement_id -> (hawkish_score, hawkish_label)
Scores = dict[str, tuple[float, str]]


def _table_exists(session) -> bool:
    return session.execute(
        text("SELECT to_regclass('hawkish_score_snapshot') IS NOT NULL")
    ).scalar()


def _snapshot_exists(session, method: str) -> bool:
    return (
        session.execute(
            text("SELECT 1 FROM hawkish_score_snapshot WHERE method = :m LIMIT 1"), {"m": method}
        ).first()
        is not None
    )


def _load_snapshot(session, method: str) -> Scores:
    rows = session.execute(
        text(
            "SELECT statement_id::text, hawkish_score, hawkish_label FROM hawkish_score_snapshot "
            "WHERE method = :m AND hawkish_score IS NOT NULL"
        ),
        {"m": method},
    ).all()
    return {r[0]: (r[1], r[2]) for r in rows}


def _compare(name: str, base: Scores, lexicon: Scores) -> None:
    """Log Pearson r, label agreement, sign flips and means on the speeches in both."""
    ids = [i for i in lexicon if i in base]
    if not ids:
        logger.info(f"No speeches in common with {name}, skipping comparison")
        return
    old = np.array([base[i][0] for i in ids], dtype=float)
    new = np.array([lexicon[i][0] for i in ids], dtype=float)
    label_agree = sum(base[i][1] == lexicon[i][1] for i in ids) / len(ids)
    sign_flips = int(np.sum(np.sign(old) != np.sign(new)))
    pearson = (
        float(np.corrcoef(old, new)[0, 1]) if old.std() > 0 and new.std() > 0 else float("nan")
    )
    logger.info(f"=== Lexicon vs {name} (n={len(ids)}) ===")
    logger.info(f"Pearson r       = {pearson:.3f}")
    logger.info(f"Label agreement = {label_agree:.1%}")
    logger.info(f"Sign flips      = {sign_flips} ({sign_flips / len(ids):.1%})")
    logger.info(f"Mean score: {name} {old.mean():+.3f}, lexicon {new.mean():+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", default="lexicon_v2", help="tag for the new rows (default: lexicon_v2)"
    )
    parser.add_argument(
        "--previous",
        default="lexicon_baseline",
        help="snapshot tag to compare against (default: lexicon_baseline)",
    )
    parser.add_argument("--dry-run", action="store_true", help="score and compare, write nothing")
    args = parser.parse_args()

    session = SessionLocal()
    try:
        has_table = _table_exists(session)
        if has_table and _snapshot_exists(session, args.method):
            if not args.dry_run:
                logger.error(f"Snapshot '{args.method}' already exists. Choose a new --method.")
                sys.exit(1)
            logger.warning(f"Snapshot '{args.method}' already exists, so a real run would stop")

        speeches = session.execute(
            select(
                Statement.id,
                Statement.raw_text,
                StatementAnalysis.hawkish_score,
                StatementAnalysis.hawkish_label,
            )
            .join(StatementAnalysis, StatementAnalysis.statement_id == Statement.id)
            .where(Statement.source_type == "speech")
        ).all()
        logger.info(f"Speeches to score: {len(speeches)}")
        if not speeches:
            return

        results = score_batch([r.raw_text or "" for r in speeches])
        lexicon = {
            str(r.id): (res["hawkish_score"], res["hawkish_label"])
            for r, res in zip(speeches, results)
        }
        logger.info(f"Label distribution: {dict(Counter(label for _, label in lexicon.values()))}")

        current = {
            str(r.id): (r.hawkish_score, r.hawkish_label)
            for r in speeches
            if r.hawkish_score is not None
        }
        _compare("statement_analysis", current, lexicon)
        if has_table:
            _compare(f"snapshot '{args.previous}'", _load_snapshot(session, args.previous), lexicon)

        if args.dry_run:
            logger.info("Dry run, nothing written")
            return

        session.execute(text(SNAPSHOT_DDL))
        session.execute(
            text(
                "INSERT INTO hawkish_score_snapshot "
                "(statement_id, hawkish_score, hawkish_label, method) "
                "VALUES (CAST(:sid AS uuid), :score, :label, :m)"
            ),
            [
                {"sid": sid, "score": s, "label": label, "m": args.method}
                for sid, (s, label) in lexicon.items()
            ],
        )
        session.commit()
        logger.info(f"Stored {len(lexicon)} lexicon scores as '{args.method}'")
    finally:
        session.close()


if __name__ == "__main__":
    main()
