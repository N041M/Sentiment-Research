"""Unit tests for the hawkish/dovish lexicon scorer.

Regression coverage for the spurious −1.0 cluster: a one-sided document must NOT
clamp to exactly ±1.0 (additive smoothing pulls thin evidence toward neutral).
TestTermMatching covers whole-word matching, which replaced substring matching after
"ease" was found to match inside "increase", "release" and "please".
"""

import pytest

from sentiment_signal.nlp.hawkish_lexicon import DOVISH, HAWKISH, SMOOTHING, score


class TestHawkishLexicon:
    def test_no_keywords_is_neutral_zero(self):
        r = score("The meeting covered procedural and administrative matters.")
        assert r["hawkish_score"] == 0.0
        assert r["hawkish_label"] == "neutral"

    def test_single_dovish_term_is_mild_not_minus_one(self):
        # The bug: one dovish term used to score exactly -1.0
        r = score("We discussed a rate cut.")
        assert r["hawkish_score"] == pytest.approx(-1 / (1 + SMOOTHING))  # -0.333
        assert r["hawkish_score"] > -1.0
        assert r["hawkish_label"] == "dovish"

    def test_single_hawkish_term_is_mild_not_plus_one(self):
        r = score("We may consider a rate hike.")
        assert r["hawkish_score"] == pytest.approx(1 / (1 + SMOOTHING))  # +0.333
        assert r["hawkish_score"] < 1.0
        assert r["hawkish_label"] == "hawkish"

    def test_one_sided_document_never_reaches_extremes(self):
        # Even a heavily one-sided document stays strictly inside (-1, 1)
        heavy = "rate hike rate increase tighten restrictive hawkish overheating"
        r = score(heavy)
        assert 0 < r["hawkish_score"] < 1.0

    def test_more_evidence_gives_stronger_score(self):
        weak = score("We discussed a rate cut.")["hawkish_score"]
        strong = score("rate cut lower rates accommodative recession risk")["hawkish_score"]
        assert strong < weak < 0  # more dovish hits -> more negative

    def test_balanced_is_neutral(self):
        r = score("They weighed a rate hike against a rate cut.")
        assert r["hawkish_score"] == pytest.approx(0.0)
        assert r["hawkish_label"] == "neutral"


def _hits(text: str) -> tuple[int, int]:
    r = score(text)
    return r["hawk_hits"], r["dove_hits"]


class TestTermMatching:
    def test_rate_increase_is_hawkish_only(self):
        # "ease" used to match inside "increase" and cancel the hawkish hit
        r = score("The Committee approved a rate increase.")
        assert (r["hawk_hits"], r["dove_hits"]) == (1, 0)
        assert r["hawkish_label"] == "hawkish"

    @pytest.mark.parametrize(
        "text",
        [
            "The press release is attached.",
            "Please see the minutes.",
            "Prices continued to increase.",
            "Output may decrease.",
        ],
    )
    def test_ease_inside_other_words_does_not_match(self, text):
        assert _hits(text) == (0, 0)

    def test_ease_policy_is_dovish(self):
        assert _hits("We stand ready to ease policy.") == (0, 1)

    def test_tightening_matches_once(self):
        # "tighten" must not also fire inside "tightening"
        assert _hits("Further tightening is appropriate.") == (1, 0)

    def test_phrase_suppresses_term_inside_it(self):
        assert _hits("Quantitative tightening continues.") == (1, 0)
        assert _hits("Quantitative easing continues.") == (0, 1)

    def test_less_accommodative_is_not_dovish(self):
        assert _hits("Policy will become less accommodative.") == (1, 0)

    def test_partly_overlapping_phrases_both_count(self):
        # "inflation above" and "above target" share a word but neither contains the other
        assert _hits("Inflation above target is a concern.") == (2, 0)

    def test_counts_occurrences(self):
        assert _hits("A rate hike now, and another rate hike later.") == (2, 0)

    def test_plural_matches(self):
        assert _hits("Further rate hikes may be needed.") == (1, 0)
        assert _hits("Markets expect rate cuts.") == (0, 1)

    def test_phrase_matches_across_line_break(self):
        assert _hits("a rate\nhike") == (1, 0)

    @pytest.mark.parametrize("term", sorted(HAWKISH))
    def test_each_hawkish_term_scores_one_hawkish_hit(self, term):
        assert _hits(term) == (1, 0)

    @pytest.mark.parametrize("term", sorted(DOVISH))
    def test_each_dovish_term_scores_one_dovish_hit(self, term):
        assert _hits(term) == (0, 1)
