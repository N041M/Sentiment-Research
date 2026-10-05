"""Tests for the deterministic two-level topic labeller (nlp/topic_lexicon)."""

import pytest

from sentiment_signal.nlp.topic_lexicon import THEMES, _clean_terms, classify


def test_monetary_policy_main_and_sub():
    text = "The FOMC weighed a rate hike of 25 basis points as inflation stayed elevated."
    main, secondary = classify(text)
    assert main == "Monetary policy"
    assert secondary in {"Rate decisions", "Inflation & prices", "Outlook & labour market"}


def test_sanctions_main():
    text = (
        "By the authority vested in me I declare a national emergency and order blocked "
        "property of persons in Russia and Ukraine under OFAC sanction."
    )
    main, _ = classify(text)
    assert main == "Sanctions & emergencies"


def test_trade_tariffs_main():
    text = "Under section 232 I adjust imports of steel and aluminum with a new tariff schedule."
    main, _ = classify(text)
    assert main == "Trade & tariffs"


def test_ceremonial_proclamation_main():
    text = "Now therefore I hereby proclaim Flag Day and Flag Week as a national observance."
    main, _ = classify(text)
    assert main == "Ceremonial proclamations"


def test_no_keywords_is_other_with_tfidf_secondary():
    main, secondary = classify("the quick brown fox jumped over", tfidf_terms="fox / brown")
    assert main == "Other"
    assert secondary == "fox / brown"


def test_other_without_tfidf_is_general():
    main, secondary = classify("the quick brown fox jumped over")
    assert main == "Other"
    assert secondary == "general"


def test_matched_main_always_returns_named_subtheme():
    # When a main theme matches, the secondary is always one of its sub-themes
    # (never the TF-IDF fallback), even if junk TF-IDF terms are supplied.
    main, secondary = classify("inflation and price stability", tfidf_terms="pdf / vol")
    assert main == "Monetary policy"
    assert secondary != "pdf / vol"
    assert secondary in {
        "Rate decisions",
        "Inflation & prices",
        "Outlook & labour market",
        "Financial stability",
    }


def test_strongest_theme_wins():
    # Two themes present; the one with more keyword hits should win.
    text = "inflation inflation inflation cpi disinflation. one mention of steel tariff."
    main, _ = classify(text)
    assert main == "Monetary policy"


def test_clean_terms_drops_boilerplate_and_digits():
    assert _clean_terms("return text / pdf / 2024") == "general"
    assert _clean_terms(["inflation", "pdf", "42", "cpi"]) == "inflation / cpi"


def test_clean_terms_empty_is_general():
    assert _clean_terms(None) == "general"
    assert _clean_terms("") == "general"


def test_classify_is_case_insensitive():
    lower = classify("section 232 steel aluminum tariff")
    upper = classify("SECTION 232 STEEL ALUMINUM TARIFF")
    assert lower == upper


def test_central_bank_reserve_not_defense():
    # 'reserve' must not drag Fed/RBA speeches into Defense via "Federal Reserve" etc.
    text = (
        "The Federal Reserve and the Reserve Bank discussed bank reserves, "
        "inflation, the labour market, and the policy rate at the FOMC meeting."
    )
    main, _ = classify(text)
    assert main == "Monetary policy"


def test_launch_facility_not_space():
    # 'launch' must not drag a CB speech about launching a facility into Space.
    text = (
        "The central bank decided to launch a new lending facility to support "
        "financial stability and the transmission of monetary policy."
    )
    main, _ = classify(text)
    assert main == "Monetary policy"


@pytest.mark.parametrize(
    "text",
    [
        "Chair Powell spoke to reporters.",  # "pow"
        "Supply chains lost power for a week.",  # "pow"
        "The minutes are kept in the archive.",  # "hiv"
        "Economic development slowed.",  # "opm"
        "Clear guidance aids market functioning.",  # "AIDS"
        "The meeting was officially sanctioned.",  # "sanction"
        "Prices kept increasing.",  # "easing"
    ],
)
def test_keyword_does_not_match_unrelated_word(text):
    assert classify(text)[0] == "Other"


def test_powell_speech_is_monetary_policy():
    # "pow" used to match "Powell" and "power" three times and outvote two monetary hits
    text = "Chair Powell said inflation is easing. Powell expects power prices to fall."
    assert classify(text)[0] == "Monetary policy"


def test_aids_matches_only_in_capitals():
    assert classify("World AIDS Day")[0] == "Public health"
    assert classify("It aids recovery.")[0] == "Other"


@pytest.mark.parametrize(
    ("text", "main"),
    [
        ("New cybersecurity rules.", "Technology & cyber"),
        ("Russian forces crossed the border.", "Sanctions & emergencies"),
        ("Executive Order 13660 remains in effect.", "Sanctions & emergencies"),
        ("Inflationary pressures persist.", "Monetary policy"),
    ],
)
def test_starred_keyword_matches_longer_words(text, main):
    assert classify(text)[0] == main


def test_plural_matches():
    assert classify("New tariffs on imports.")[0] == "Trade & tariffs"


def test_phrase_matches_across_line_break():
    assert classify("a rate\nhike")[0] == "Monetary policy"


def test_hts_matches_next_to_punctuation():
    # "hts" used to be padded with spaces, which missed "HTS," and "(HTS)"
    assert classify("See heading 9903.88.15 of the HTS, as amended.")[0] == "Trade & tariffs"
    assert classify("Human rights and insights.")[0] == "Other"


def test_counts_every_occurrence():
    # Three mentions of one keyword outweigh two different keywords.
    assert classify("steel steel steel. inflation and cpi.")[0] == "Trade & tariffs"


@pytest.mark.parametrize(
    ("keyword", "main"),
    [(k, main) for main, subs in THEMES.items() for kws in subs.values() for k in sorted(kws)],
)
def test_each_keyword_matches_its_own_theme(keyword, main):
    assert classify(keyword.rstrip("*"))[0] == main
