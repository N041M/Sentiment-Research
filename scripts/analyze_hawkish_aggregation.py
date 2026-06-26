#!/usr/bin/env python3
"""Diagnostic (report-only): compare hawkish/dovish document-aggregation schemes.

Mean-pooling FOMC-RoBERTa chunk probabilities collapses long speeches toward
neutral (the live run labelled 80% of speeches neutral, mean score ~0). FOMC-RoBERTa
is a *sentence*-level classifier, so a single confidently-hawkish passage is buried
when token-weighted-averaged against neutral filler.

This scores each speech's 512-token chunks ONCE (cached to scratchpad), then applies
several aggregation schemes to the cached per-chunk probabilities and reports the
resulting stance distribution / spread / correlation with the lexicon baseline — to
decide whether a non-mean aggregation recovers discrimination. Writes nothing to the
DB; the lexicon baseline is read from the `hawkish_score_snapshot` table.

Run from project root with venv active:
    python scripts/analyze_hawkish_aggregation.py
    python scripts/analyze_hawkish_aggregation.py --band 0.10 --recompute
"""

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
from loguru import logger
from sqlalchemy import select, text

from sentiment_signal.db.models import Statement, StatementAnalysis
from sentiment_signal.db.session import SessionLocal
from sentiment_signal.utils.terminal import progress

# FOMC-RoBERTa resolved class indices (verified live: hawk=1, dove=0, neutral=2).
H, D, N = 1, 0, 2

# Kept in the system temp dir (stable across runs, never inside the public repo).
CACHE = Path(tempfile.gettempdir()) / "sentiment_hawkish_chunk_probs.pkl"


def compute_chunk_probs(
    speeches: list[tuple[str, str]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per speech, the FOMC-RoBERTa softmax probabilities of every 512-token chunk
    and the chunk token counts. The single expensive (model) pass."""
    from sentiment_signal.nlp.pipeline import NLPPipeline

    pipe = NLPPipeline()
    pipe._ensure_fomc()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid, raw in progress(speeches, "chunk-score", every=50):
        chunks = pipe._chunk_text(raw, tokenizer=pipe._fomc_tokenizer) or [("", 1)]
        probs = pipe._fomc_probs([c for c, _ in chunks])
        out[sid] = (np.array(probs, dtype=float), np.array([n for _, n in chunks], dtype=float))
    return out


# --- aggregation schemes: (chunk_probs [n,3], token_weights [n]) -> doc score in [-1,1] ---
def agg_mean(p: np.ndarray, w: np.ndarray) -> float:
    """Current behaviour: token-weighted mean of chunk probabilities, then hawk-dove."""
    wm = np.average(p, axis=0, weights=w if w.sum() > 0 else None)
    return float(wm[H] - wm[D])


def agg_stance_weighted(p: np.ndarray, w: np.ndarray) -> float:
    """Token-weighted mean of per-chunk (hawk-dove), but each chunk weighted by its
    non-neutral mass (1 - P[neutral]) so neutral filler barely contributes."""
    sw = w * (1.0 - p[:, N])
    if sw.sum() <= 0:
        return 0.0
    return float(np.average(p[:, H] - p[:, D], weights=sw))


def agg_drop_neutral(p: np.ndarray, w: np.ndarray) -> float:
    """Mean of (hawk-dove) over only the chunks whose argmax label is not neutral."""
    mask = p.argmax(axis=1) != N
    if not mask.any():
        return 0.0
    s, ww = (p[:, H] - p[:, D])[mask], w[mask]
    return float(np.average(s, weights=ww if ww.sum() > 0 else None))


def agg_max_abs(p: np.ndarray, w: np.ndarray) -> float:
    """The single most stance-bearing chunk (max |hawk-dove|)."""
    s = p[:, H] - p[:, D]
    return float(s[np.argmax(np.abs(s))])


def agg_net_label(p: np.ndarray, w: np.ndarray) -> float:
    """Net hawkish chunk fraction: (#hawkish - #dovish) / #chunks (label-count index,
    the Trillion Dollar Words style)."""
    lab = p.argmax(axis=1)
    return float((np.sum(lab == H) - np.sum(lab == D)) / len(lab))


SCHEMES = {
    "mean (current)": agg_mean,
    "stance_weighted": agg_stance_weighted,
    "drop_neutral": agg_drop_neutral,
    "max_abs": agg_max_abs,
    "net_label_frac": agg_net_label,
}


def summarize(name: str, scores: dict[str, float], baseline: dict[str, float], band: float) -> None:
    sc = np.array(list(scores.values()))
    hawk = int(np.sum(sc > band))
    dove = int(np.sum(sc < -band))
    neut = len(sc) - hawk - dove
    common = [s for s in scores if s in baseline]
    a = np.array([scores[s] for s in common])
    b = np.array([baseline[s] for s in common])
    r = (
        float(np.corrcoef(a, b)[0, 1])
        if len(common) > 2 and a.std() > 0 and b.std() > 0
        else float("nan")
    )
    logger.info(
        f"{name:16s} std={sc.std():.3f} mean={sc.mean():+.3f}  "
        f"hawk/neut/dove={hawk:4d}/{neut:4d}/{dove:4d}  r_vs_lexicon={r:+.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--band", type=float, default=0.05, help="neutral band: |score|<=band -> neutral"
    )
    parser.add_argument(
        "--recompute", action="store_true", help="ignore the cache and re-run the model pass"
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        conn = session.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
        baseline = {
            r[0]: float(r[1])
            for r in conn.execute(
                text(
                    "SELECT statement_id::text, hawkish_score FROM hawkish_score_snapshot "
                    "WHERE method = 'lexicon_baseline' AND hawkish_score IS NOT NULL"
                )
            ).all()
        }
        logger.info(f"Lexicon baseline: {len(baseline)} speeches")

        speeches = [
            (str(r.id), r.raw_text)
            for r in session.execute(
                select(Statement.id, Statement.raw_text)
                .join(StatementAnalysis, StatementAnalysis.statement_id == Statement.id)
                .where(Statement.source_type == "speech")
            ).all()
        ]
        logger.info(f"Speeches to aggregate: {len(speeches)}")
    finally:
        session.close()

    if CACHE.exists() and not args.recompute:
        logger.info(f"Loading cached chunk probabilities from {CACHE.name}")
        chunk_probs = pickle.loads(CACHE.read_bytes())
    else:
        chunk_probs = compute_chunk_probs(speeches)
        CACHE.write_bytes(pickle.dumps(chunk_probs))
        logger.info(f"Cached chunk probabilities to {CACHE.name} ({len(chunk_probs)} speeches)")

    logger.info(f"=== Aggregation comparison (neutral band ±{args.band}) ===")
    logger.info("(higher std = more discrimination; lexicon baseline mean -0.198, std for ref)")
    base_sc = np.array(list(baseline.values()))
    logger.info(f"{'lexicon':16s} std={base_sc.std():.3f} mean={base_sc.mean():+.3f}  (reference)")
    for name, fn in SCHEMES.items():
        scores = {sid: fn(p, w) for sid, (p, w) in chunk_probs.items()}
        summarize(name, scores, baseline, args.band)


if __name__ == "__main__":
    main()
