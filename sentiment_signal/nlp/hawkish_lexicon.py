"""Rule-based hawkish/dovish scorer for central bank text.

Based on the monetary policy lexicon approach used in the FOMC NLP
literature (Apel & Blix Grimaldi 2012; Schmeling & Wagner 2019).

hawkish_score = (hawkish_hits - dovish_hits) / (total_hits + SMOOTHING)
  > 0 → hawkish (restrictive / inflation-fighting)
  < 0 → dovish  (accommodative / growth-supporting)
  = 0 → neutral or no signal

Without SMOOTHING, (hawk - dove) / total gives every one-sided document a score of
exactly ±1.0, so a single matched keyword would score the same as a strongly
one-sided speech. Additive (Laplace) smoothing pulls thin evidence toward neutral so
that the magnitude reflects how much evidence there is. One dovish hit scores −0.33
and ten score −0.83.

Terms match as whole words, and the words of a phrase may be separated by any
whitespace, including line breaks. The last word may also carry a plural "s", so
"rate hike" matches "rate hikes". Every occurrence counts. A match that lies entirely
inside a longer match is dropped, so "quantitative tightening" counts once and the
"accommodative" inside "less accommodative" does not count as dovish.
"""

from __future__ import annotations

import re

# Pseudo-counts of neutral evidence; regularises thin/one-sided matches toward 0.
SMOOTHING = 2

# Words/phrases strongly associated with tightening/restrictive stance
HAWKISH = frozenset(
    [
        "raise rates",
        "rate hike",
        "rate increase",
        "increase rates",
        "tighten",
        "tightening",
        "restrictive",
        "above neutral",
        "upside risk",
        "inflation risk",
        "inflation expectations",
        "overheating",
        "overheat",
        "price stability",
        "price pressures",
        "inflationary",
        "hawkish",
        "less accommodative",
        "remove accommodation",
        "normalize",
        "normalisation",
        "normalization",
        "reduce balance sheet",
        "quantitative tightening",
        "qt",
        "above target",
        "persistent inflation",
        "wage growth",
        "labor market tight",
        "labour market tight",
        "above-target",
        "inflation above",
    ]
)

# Words/phrases strongly associated with easing/accommodative stance
DOVISH = frozenset(
    [
        "cut rates",
        "rate cut",
        "rate reduction",
        "lower rates",
        "ease",
        "easing",
        "accommodative",
        "below neutral",
        "downside risk",
        "recession risk",
        "unemployment",
        "labor market slack",
        "labour market slack",
        "below target",
        "inflation below",
        "deflationary",
        "dovish",
        "more accommodation",
        "additional support",
        "quantitative easing",
        "qe",
        "asset purchases",
        "forward guidance",
        "lower for longer",
        "support the economy",
        "economic support",
        "weak growth",
        "subdued inflation",
        "below-target",
    ]
)


def _pattern(term: str) -> re.Pattern[str]:
    words = r"\s+".join(re.escape(w) for w in term.split())
    return re.compile(rf"\b{words}s?\b")


# (pattern, is_hawkish) for every lexicon term, compiled once at import.
_PATTERNS = [(_pattern(t), True) for t in sorted(HAWKISH)] + [
    (_pattern(t), False) for t in sorted(DOVISH)
]


def _hits(text_lower: str) -> tuple[int, int]:
    """Count hawkish and dovish matches, skipping any match inside a longer one."""
    # Sorted by start, longest first, so a containing match is seen before its contents.
    matches = sorted(
        (m.start(), -m.end(), is_hawk)
        for pattern, is_hawk in _PATTERNS
        for m in pattern.finditer(text_lower)
    )
    hawk = dove = 0
    reach = -1  # furthest end of any match kept so far
    for _, neg_end, is_hawk in matches:
        end = -neg_end
        if end <= reach:
            continue
        reach = end
        if is_hawk:
            hawk += 1
        else:
            dove += 1
    return hawk, dove


def score(text: str) -> dict:
    """Return hawkish_score ∈ [-1, 1], label, and hit counts."""
    hawk_hits, dove_hits = _hits(text.lower())
    total = hawk_hits + dove_hits
    # Smoothed net tone in (-1, 1); never clamps to ±1 on thin one-sided evidence.
    raw_score = (hawk_hits - dove_hits) / (total + SMOOTHING)

    if total == 0:
        label = "neutral"
    elif raw_score > 0.1:
        label = "hawkish"
    elif raw_score < -0.1:
        label = "dovish"
    else:
        label = "neutral"

    return {
        "hawkish_score": float(raw_score),
        "hawkish_label": label,
        "hawk_hits": hawk_hits,
        "dove_hits": dove_hits,
    }


def score_batch(texts: list[str]) -> list[dict]:
    return [score(t) for t in texts]
