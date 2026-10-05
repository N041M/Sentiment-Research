"""APScheduler job definitions.

Each job runs its blocking work in a thread (`asyncio.to_thread`) so the async
scheduler loop stays responsive. Speech scrapers fetch only the current year on
their interval (the full back-catalogue is a one-off via scripts/run_phase1.py).

Market-data and pipeline jobs delegate to the step functions in
scripts/run_phase1.py. Those fetch the full date range and rely on
on_conflict_do_nothing for idempotency — correct but not incremental; an
incremental fetch is a future optimisation.

Channel 1B (reaction) jobs are Phase 2 stubs — the Reddit/YouTube scrapers do
not exist yet.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

scheduler = AsyncIOScheduler()


def setup_scheduler() -> AsyncIOScheduler:
    # ── Channel 1A — statement scrapers ──────────────────────────────────────
    scheduler.add_job(fed_speeches_job, "interval", hours=6, id="fed_speeches")
    scheduler.add_job(federal_register_job, "interval", hours=6, id="federal_register")
    scheduler.add_job(bis_speeches_job, "interval", hours=6, id="bis_speeches")
    scheduler.add_job(boe_speeches_job, "interval", hours=6, id="boe_speeches")
    scheduler.add_job(rba_speeches_job, "interval", hours=12, id="rba_speeches")
    scheduler.add_job(un_news_job, "interval", hours=6, id="un_news")

    # ── Channel 1B — reaction scrapers (Phase 2, not yet implemented) ────────
    scheduler.add_job(praw_reactions_job, "interval", minutes=30, id="praw_reactions")
    scheduler.add_job(youtube_reactions_job, "interval", hours=4, id="youtube_reactions")

    # ── Channel 2 — NLP scoring of newly scraped statements ──────────────────
    scheduler.add_job(nlp_scoring_job, "interval", hours=2, id="nlp_scoring")

    # ── Channel 3 — market data + event detection (daily) ────────────────────
    scheduler.add_job(yfinance_daily_job, "cron", hour=0, minute=0, id="yfinance_daily")
    scheduler.add_job(fred_daily_job, "cron", hour=0, minute=15, id="fred_daily")
    scheduler.add_job(event_detection_job, "cron", hour=0, minute=30, id="event_detection")

    return scheduler


def _run_scraper(factory) -> None:
    """Open a session, run the scraper built by `factory(session)`, then close."""
    from sentiment_signal.db.session import SessionLocal

    session = SessionLocal()
    try:
        factory(session).run()
    finally:
        session.close()


# ── Channel 1A — statement scrapers ──────────────────────────────────────────


async def fed_speeches_job() -> None:
    logger.info("fed_speeches_job: starting")
    from sentiment_signal.collectors.fed_speeches import FedSpeechesScraper

    year = datetime.now().year
    await asyncio.to_thread(_run_scraper, lambda s: FedSpeechesScraper(s, start_year=year))


async def federal_register_job() -> None:
    logger.info("federal_register_job: starting")
    from sentiment_signal.collectors.federal_register import FederalRegisterScraper

    year = datetime.now().year
    await asyncio.to_thread(_run_scraper, lambda s: FederalRegisterScraper(s, start_year=year))


async def bis_speeches_job() -> None:
    logger.info("bis_speeches_job: starting")
    from sentiment_signal.collectors.bis_speeches import BISSpeechesScraper

    await asyncio.to_thread(_run_scraper, BISSpeechesScraper)


async def boe_speeches_job() -> None:
    logger.info("boe_speeches_job: starting")
    from sentiment_signal.collectors.boe_speeches import BoESpeechesScraper

    await asyncio.to_thread(_run_scraper, BoESpeechesScraper)


async def rba_speeches_job() -> None:
    logger.info("rba_speeches_job: starting")
    from sentiment_signal.collectors.rba_speeches import RBASpeechesScraper

    year = datetime.now().year
    await asyncio.to_thread(_run_scraper, lambda s: RBASpeechesScraper(s, start_year=year))


async def un_news_job() -> None:
    logger.info("un_news_job: starting")
    from sentiment_signal.collectors.un_sg_speeches import UNSGSpeechesScraper

    year = datetime.now().year
    await asyncio.to_thread(_run_scraper, lambda s: UNSGSpeechesScraper(s, start_year=year))


# ── Channel 1B — reaction scrapers (Phase 2 stubs) ───────────────────────────


async def praw_reactions_job() -> None:
    logger.info("praw_reactions_job: Phase 2 — Reddit/PRAW reaction scraper not yet implemented")


async def youtube_reactions_job() -> None:
    logger.info("youtube_reactions_job: Phase 2 — YouTube reaction scraper not yet implemented")


# ── Channel 2 — NLP scoring ──────────────────────────────────────────────────


async def nlp_scoring_job() -> None:
    """Score newly scraped statements with FinBERT and (re)build signal rows."""
    logger.info("nlp_scoring_job: starting")
    from scripts.run_phase1 import step3_score_statements, step4_compute_signals

    await asyncio.to_thread(step3_score_statements)
    await asyncio.to_thread(step4_compute_signals)


# ── Channel 3 — market data + event detection ────────────────────────────────


async def yfinance_daily_job() -> None:
    logger.info("yfinance_daily_job: starting")
    from scripts.run_phase1 import step5_fetch_prices

    await asyncio.to_thread(step5_fetch_prices)


async def fred_daily_job() -> None:
    logger.info("fred_daily_job: starting")
    from scripts.run_phase1 import step6_fetch_macro

    await asyncio.to_thread(step6_fetch_macro)


async def event_detection_job() -> None:
    logger.info("event_detection_job: starting")
    # step9 (event_context window-attribution) is retired per architecture_v2 — it is
    # non-causal/geography-blind (build-log §13.26) and no longer feeds the dashboard,
    # so we no longer build it on a schedule (stale/invalid pairs were accumulating).
    # Event *detection* (step8) is independent and still useful, so it stays.
    from scripts.run_phase1 import step8_detect_events

    await asyncio.to_thread(step8_detect_events)
