"""Point-in-time sentiment event sequence (architecture_v2 Stage 2.5).

Emits a time-ordered sequence of per-statement sentiment events and an **as-of**
selector that returns only events strictly before a reference time — the no-look-
ahead primitive every downstream learner (the engineered novelty baseline, the
interpreter transformer, the LSTM ablation) must build on.

Deliberately does NOT pre-aggregate into window means: the raw ordered sequence,
with each event's topic and recency intact, is exactly what lets a sequence model
learn habituation (repeated similar statements -> diminishing impact). Collapsing
to a window mean is what made the old `event_context` attribution unusable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class SentimentEvent:
    """One statement as a point-in-time sentiment event."""

    statement_id: str
    timestamp: datetime  # tz-aware UTC; when the information became public
    person: str | None
    institution: str | None
    influence_tier: int | None
    source_type: str | None
    sentiment_score: float | None
    hawkish_score: float | None
    topic_main: str | None
    topic_id: int | None  # BERTopic topic / cluster_id; cheap "same topic" key
    sentiment_label: str | None = None  # FinBERT argmax class (positive/neutral/negative)
    hawkish_label: str | None = None  # FOMC-RoBERTa stance (hawkish/neutral/dovish); speech-only


def _utc(ts: datetime) -> datetime:
    """Normalise to tz-aware UTC (naive timestamps are assumed already UTC)."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def order_events(events: Iterable[SentimentEvent]) -> list[SentimentEvent]:
    """Return events sorted ascending by timestamp."""
    return sorted(events, key=lambda e: _utc(e.timestamp))


def events_asof(
    events: Sequence[SentimentEvent],
    t: datetime,
    *,
    lookback: timedelta | None = None,
    inclusive: bool = False,
) -> list[SentimentEvent]:
    """Events available **as of** time `t`.

    Strictly before `t` by default (point-in-time; no look-ahead). `inclusive=True`
    also admits events exactly at `t`. `lookback` bounds how far back to include.
    Output preserves the input order (pass an `order_events` result for a sorted
    sequence). This is the single guard against feature leakage downstream.
    """
    t = _utc(t)
    lo_time = t - lookback if lookback is not None else None
    out: list[SentimentEvent] = []
    for e in events:
        et = _utc(e.timestamp)
        before = et <= t if inclusive else et < t
        if before and (lo_time is None or et >= lo_time):
            out.append(e)
    return out


def load_event_sequence(
    session, *, source_types: Sequence[str] | None = None
) -> list[SentimentEvent]:
    """Load time-ordered sentiment events from the DB (Stage 1 output).

    `source_types` filters to in-domain text (e.g. ("speech",)) — the inferential
    path should exclude ceremonial presidential documents (out-of-domain for the
    sentiment models); pass None to load everything.
    """
    from sqlalchemy import select

    from sentiment_signal.db.models import Person, Statement, StatementAnalysis

    query = (
        select(
            Statement.id,
            Statement.published_at,
            Statement.source_type,
            Person.canonical_name,
            Person.institution,
            Person.influence_tier,
            StatementAnalysis.sentiment_score,
            StatementAnalysis.hawkish_score,
            StatementAnalysis.topic_main,
            StatementAnalysis.cluster_id,
            StatementAnalysis.sentiment_label,
            StatementAnalysis.hawkish_label,
        )
        .join(Person, Person.id == Statement.person_id)
        .join(StatementAnalysis, StatementAnalysis.statement_id == Statement.id)
    )
    if source_types:
        query = query.where(Statement.source_type.in_(list(source_types)))

    events = [
        SentimentEvent(
            statement_id=str(r[0]),
            timestamp=r[1],
            source_type=r[2],
            person=r[3],
            institution=r[4],
            influence_tier=r[5],
            sentiment_score=r[6],
            hawkish_score=r[7],
            topic_main=r[8],
            topic_id=r[9],
            sentiment_label=r[10],
            hawkish_label=r[11],
        )
        for r in session.execute(query).all()
    ]
    return order_events(events)
