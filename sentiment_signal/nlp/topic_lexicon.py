"""Deterministic two-level topic labelling for speech/document clusters.

Maps a cluster's text to a readable **main headline** (broad theme) and a
**secondary context** (sub-theme) using a curated keyword lexicon. The approach is
rule-based and reproducible, like `hawkish_lexicon`. Expand `THEMES` as new topics
appear in the corpus.

    main, secondary = classify(cluster_text, tfidf_terms)

Keywords match as whole words, so "pow" does not match "Powell" and "opm" does not
match "development". The words of a phrase may be separated by any whitespace,
including line breaks, and the last word may carry a plural "s". A keyword ending in
"*" also matches longer words that start with it, so "cyber*" matches "cybersecurity".
A keyword written in capitals matches only in capitals, so "AIDS" does not match the
verb "aids". Every occurrence of every keyword counts.
"""

from __future__ import annotations

import re

# Boilerplate / scraping artefacts that should never drive a label.
STOPWORDS = frozenset(
    {
        "return text",
        "return to text",
        "pdf",
        "html",
        "accessible version",
        "federal register",
        "presidential documents",
        "vol",
        "page",
        "no",
    }
)

# main theme -> {sub-theme -> keyword set}. A cluster's score for a theme is the
# total keyword hits across its sub-themes. The sub-theme with the most hits is the
# secondary.
THEMES: dict[str, dict[str, set[str]]] = {
    "Monetary policy": {
        "Rate decisions": {
            "rate hike",
            "rate cut",
            "federal funds",
            "basis points",
            "cash rate",
            "tightening",
            "easing",
            "policy rate",
            "bank rate",
        },
        "Inflation & prices": {
            "inflation*",
            "price stability",
            "disinflation*",
            "consumer prices",
            "cpi",
        },
        "Outlook & labour market": {
            "labour market",
            "labor market",
            "unemployment",
            "economic outlook",
            "fomc",
            "monetary policy*",
            "mpc",
        },
        "Financial stability": {
            "financial stability",
            "systemic*",
            "stress test*",
            "bank capital",
            "prudential*",
            "macroprudential",
        },
    },
    "Sanctions & emergencies": {
        "Iran": {"respect iran", "iran*"},
        "Russia & Ukraine": {"russia*", "ukraine", "crimea*"},
        "Counter-narcotics": {"narcotics", "traffickers", "drug trafficking"},
        "Asset blocking": {
            "blocked property",
            "interests property",
            "property interests",
            "national emergency",
            "executive order 13*",
            "ofac",
            "sanction",
        },
        "Regional emergencies": {
            "sudan*",
            "somalia",
            "congo*",
            "zimbabwe*",
            "balkans",
            "libya*",
            "belarus*",
            "syria*",
            "lebanon",
            "nicaragua*",
        },
    },
    "Trade & tariffs": {
        "Steel & aluminium (232)": {"section 232", "steel", "aluminum", "aluminium"},
        "Tariff schedule": {
            "tariff",
            "harmonized tariff",
            "hts",
            "import duty",
            "section 301",
            "quota",
            "clause proclamation",
        },
        "Trade policy": {"trade agreement", "trade deficit", "free trade", "export control"},
    },
    "Defense & security": {
        "Flags at half-staff": {"half staff", "flown half", "naval vessels", "shall flown"},
        # Bare "reserve" and "drawdown" are left out because CB speeches use them in
        # "Federal Reserve", "Reserve Bank", "bank reserves" and liquidity "drawdown".
        "Armed forces": {
            "armed forces",
            "national guard",
            "reservists",
            "reserve component",
            "naval",
            "defense",
            "servicemembers",
            "section 506",
            "active duty",
        },
        "Veterans": {"veteran", "caregivers", "military spouses", "pearl harbor", "pow"},
    },
    "Ceremonial proclamations": {
        "Patriotic observances": {
            "flag day",
            "flag week",
            "independence day",
            "constitution day",
            "wright brothers",
            "leif erikson",
            "pulaski",
        },
        "Awareness & recognition": {
            "awareness month",
            "awareness week",
            "national day",
            "observance",
            "recognition day",
            "hereby proclaim",
            "proclaim*",
            "prayer",
            "thanksgiving",
        },
        "Remembrance": {"remembrance", "holocaust", "memorial*"},
    },
    "Public health": {
        "Disease awareness": {
            "cancer",
            "prostate",
            "ovarian",
            "hiv",
            "AIDS",
            "alzheimer",
            "diabetes",
        },
        "Substance & pandemic": {"influenza", "pandemic", "overdose", "substance", "opioid"},
    },
    "Civil rights & social policy": {
        "Equality & justice": {
            "civil rights",
            "equal pay",
            "disabilities",
            "lgbtqi",
            "sexual assault",
            "vawa",
            "crime victims",
            "dating violence",
            "stalking",
            "race sex",
        },
        "Education": {"charter schools", "hbcus", "apprenticeship"},
    },
    "Immigration": {
        "Refugees & visas": {
            "refugee admissions",
            "refugee",
            "visa overstay",
            "overstay",
            "noncitizen",
            "asylum",
        },
        "Border": {"border facilities", "border security", "permittee"},
    },
    "Technology & cyber": {
        "AI & infrastructure": {
            "ai infrastructure",
            "artificial intelligence",
            "generative",
            "frontier",
            "data center",
        },
        "Cyber & intelligence": {
            "cyber*",
            "signals intelligence",
            "spyware",
            "critical technology",
            "semiconductor",
            "reactor",
        },
        # "launch" alone collides with launching CB facilities/programs.
        "Space": {"national space", "spaceflight", "space launch", "rocket launch", "satellite"},
    },
    "Energy & environment": {
        "Public lands": {"public lands", "great outdoors", "national monument", "wildlife"},
        "Energy & climate": {"energy", "climate", "emissions", "recycling", "conservation"},
    },
    "Government operations": {
        "Delegation of authority": {
            "functions authorities",
            "authorities vested",
            "delegate*",
            "succession",
            "order shall",
            "insofar",
        },
        "Civil service": {"competitive service", "probationary", "opm", "federal workforce"},
    },
}


def _pattern(keyword: str) -> re.Pattern[str]:
    stem = keyword.endswith("*")
    words = r"\s+".join(re.escape(w) for w in keyword.rstrip("*").split())
    flags = 0 if keyword.isupper() else re.IGNORECASE
    return re.compile(rf"\b{words}\w*" if stem else rf"\b{words}s?\b", flags)


# THEMES with every keyword compiled once at import.
_PATTERNS = {
    main: {sub: [_pattern(k) for k in sorted(kws)] for sub, kws in subs.items()}
    for main, subs in THEMES.items()
}

_TERM_SPLIT = re.compile(r"\s*/\s*|\s*,\s*")


def _clean_terms(tfidf_terms: str | list[str] | None) -> str:
    """Turn the TF-IDF label into a tidy secondary fallback, dropping boilerplate."""
    if not tfidf_terms:
        return "general"
    parts = _TERM_SPLIT.split(tfidf_terms) if isinstance(tfidf_terms, str) else list(tfidf_terms)
    kept = [
        p.strip()
        for p in parts
        if p.strip() and p.strip().lower() not in STOPWORDS and not p.strip().isdigit()
    ]
    return " / ".join(kept[:3]) if kept else "general"


def classify(text: str, tfidf_terms: str | list[str] | None = None) -> tuple[str, str]:
    """Return (main_headline, secondary_context) for a cluster's text.

    `text` is a representative sample of the cluster's documents. `tfidf_terms` is
    the cluster's TF-IDF label, used as the secondary fallback when no sub-theme matches.
    """
    best_main, best_score, best_sub = "Other", 0, None
    for main, subs in _PATTERNS.items():
        main_score = 0
        top_sub, top_hits = None, 0
        for sub, patterns in subs.items():
            hits = sum(len(p.findall(text)) for p in patterns)
            main_score += hits
            if hits > top_hits:
                top_sub, top_hits = sub, hits
        if main_score > best_score:
            best_main, best_score, best_sub = main, main_score, top_sub

    # A matched main always has a winning sub-theme because main_score is the sum of
    # the sub-theme hits. Only the "Other" path falls back to the cleaned TF-IDF terms.
    if best_main == "Other":
        return "Other", _clean_terms(tfidf_terms)
    return best_main, best_sub
