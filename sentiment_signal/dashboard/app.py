import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sqlalchemy import func, select

from sentiment_signal.db.models import (
    ContextPeriod,
    Event,
    Person,
    SentimentSignalRecord,
    Statement,
    StatementAnalysis,
)
from sentiment_signal.db.session import SessionLocal
from sentiment_signal.features.geography import country_for_institution
from sentiment_signal.features.market_response import statement_market_pairs

st.set_page_config(page_title="Sentiment Signal", layout="wide")
st.title("Sentiment Signal — Research Dashboard")

session = SessionLocal()
# Read-only dashboard: run in AUTOCOMMIT so each query releases its transaction
# immediately. Otherwise every Streamlit rerun leaks an 'idle in transaction'
# connection that holds locks; those then pile up behind any concurrent ALTER
# (e.g. cluster_speeches), and Postgres queues all readers behind the ALTER —
# hanging the whole dashboard.
session.connection(execution_options={"isolation_level": "AUTOCOMMIT"})

# Band colours for macro-context regimes, by category (used to shade time charts)
CONTEXT_CATEGORY_COLORS = {
    "pandemic": "#8b5cf6",
    "war_conflict": "#dc2626",
    "monetary_policy": "#2563eb",
    "fiscal_trade": "#0d9488",
    "financial_crisis": "#ea580c",
    "political": "#65a30d",
    "energy": "#ca8a04",
    "technology": "#db2777",
}

# ── Glossary ───────────────────────────────────────────────────────────────────
with st.expander("Glossary — click to expand"):
    st.markdown("""
| Term | Definition |
|---|---|
| **Sentiment score** | FinBERT financial sentiment output. Range −1 to +1: −1 = maximally negative / pessimistic language, 0 = neutral, +1 = maximally positive / optimistic. |
| **Hawkish score** | Monetary policy stance. +1 = hawkish (favours higher interest rates, tighter credit conditions), −1 = dovish (favours lower rates, looser policy). Computed by a rule-based lexicon until FOMC-RoBERTa model access is granted. |
| **Pre-event signal** | The mean sentiment score (or sharpe_analog if available) across all tracked speeches in the 48 hours before a market event. This is the x-axis in the Phase 1 scatter chart. |
| **sharpe_analog** | Novel feature: `engagement-weighted sentiment delta ÷ reaction standard deviation`. Captures how surprising a statement was relative to the public reaction it generated. NULL until Phase 2 reaction data is collected. |
| **Pearson r** | Linear correlation coefficient, −1 to +1. Measures how well a straight line describes the relationship. Values near 0 = no linear relationship; ±1 = perfect linear relationship. |
| **Spearman r** | Rank-order correlation, −1 to +1. Does not assume a linear relationship; compares the ranks of values rather than the values themselves, making it more robust to outliers. |
| **p-value** | The probability of observing a correlation this large purely by chance if no real relationship exists. p < 0.05 is the conventional threshold for statistical significance. |
| **OLS trendline** | Ordinary Least Squares regression line — the straight line that minimises the total squared vertical distance from all data points. |
| **UMAP** | Uniform Manifold Approximation and Projection. Compresses high-dimensional embeddings (768 numbers per speech) into 2 numbers so they can be plotted. Axis values have no standalone meaning — only the distance between points matters. Nearby speeches discuss similar topics. |
| **BERTopic** | Topic model: embeds each speech with a topic-semantic sentence-transformer, groups them with UMAP + HDBSCAN, and names each topic with class-based TF-IDF. Replaces the earlier FinBERT-embedding clustering, which grouped by sentiment/tone rather than topic. |
| **HDBSCAN** | Hierarchical Density-Based Spatial Clustering of Applications with Noise. Finds natural groupings without a pre-specified cluster count; used inside BERTopic. |
| **c-TF-IDF** | Class-based Term Frequency–Inverse Document Frequency. Identifies the most distinctive words of each topic relative to all other topics — the "secondary" label shown per topic. |
| **FinBERT** | A BERT-based transformer pre-trained on financial text. Used to compute sentiment scores (and the 768-d embeddings behind agreement/signal); topic clustering uses sentence-transformer embeddings instead. |
| **Embedding (768-dim)** | A list of 768 numbers encoding the semantic meaning of a speech as a point in high-dimensional space. The cosine distance between two embeddings reflects their topical similarity. |
| **Event context** | A record linking one market event to all speeches that occurred in the 48 hours before it. The mean signal across those speeches is the x-axis value in the Phase 1 chart. |
| **Influence tier** | Speaker authority: 1 = head of institution (e.g. Fed Chair, Prime Minister), 2 = deputy/vice-chair, 3 = governor/MPC member, 4 = senior staff. |
| **US Equity** | S&P 500 (`^GSPC`), Nasdaq Composite (`^IXIC`), Dow Jones (`^DJI`). Events = daily moves ≥ ±1%. |
| **European Equity** | DAX (`^GDAXI`), FTSE 100 (`^FTSE`), CAC 40 (`^FCHI`), Euro Stoxx 50 (`^STOXX50E`). Events = daily moves ≥ ±1%. |
| **Asia-Pacific Equity** | Nikkei 225 (`^N225`), Hang Seng (`^HSI`), Shanghai Composite (`000001.SS`), KOSPI (`^KS11`), ASX 200 (`^AXJO`). Events = daily moves ≥ ±1%. |
| **Major FX** | EUR/USD, USD/JPY, GBP/USD, USD/CNY. Events = daily moves ≥ ±0.5% (FX moves less in percentage terms than equity indices). |
""")

st.divider()

# ── Shared sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    persons = session.scalars(select(Person).order_by(Person.canonical_name)).all()
    person_names = ["All"] + [p.canonical_name for p in persons]
    selected_person = st.selectbox("Person", person_names)

# Check whether clustering columns have been migrated yet (incl. two-level topic labels)
try:
    session.execute(select(StatementAnalysis.cluster_id, StatementAnalysis.topic_main).limit(1))
    _clustering_cols_exist = True
except Exception:
    session.rollback()
    _clustering_cols_exist = False

# Check whether the context_periods table exists (migration 002 + seed)
try:
    context_periods = session.scalars(
        select(ContextPeriod).order_by(ContextPeriod.start_date)
    ).all()
except Exception:
    session.rollback()
    context_periods = []

# ── Status metrics ─────────────────────────────────────────────────────────────
_m = st.columns(7)
_m[0].metric(
    "Statements",
    session.scalar(select(func.count()).select_from(Statement)),
    help="Total speech documents collected across all sources (Fed, BoE, RBA, BIS).",
)
_m[1].metric(
    "Scored",
    session.scalar(select(func.count()).select_from(StatementAnalysis)),
    help="Speeches with FinBERT sentiment scores and 768-dim embeddings computed.",
)
_m[2].metric(
    "Clustered",
    session.scalar(
        select(func.count())
        .select_from(StatementAnalysis)
        .where(StatementAnalysis.cluster_id.isnot(None))
    )
    if _clustering_cols_exist
    else "—",
    help="Speeches assigned to a BERTopic topic (sentence-transformer embeddings). "
    "Run: python scripts/cluster_speeches.py",
)
_m[3].metric(
    "Reactions",
    0,
    help="Public reactions (Reddit, YouTube comments) linked to speeches. Collected in Phase 2.",
)
_m[4].metric(
    "With sharpe_analog",
    session.scalar(
        select(func.count())
        .select_from(SentimentSignalRecord)
        .where(SentimentSignalRecord.sharpe_analog.isnot(None))
    ),
    help="sharpe_analog = engagement-weighted sentiment delta ÷ reaction std. "
    "Requires Phase 2 reaction data — NULL for all rows until then.",
)
_m[5].metric(
    "Market events",
    session.scalar(select(func.count()).select_from(Event)),
    help="Price moves above the detection threshold across all tracked markets 2015–present. "
    "Equity: ≥±1%. FX: ≥±0.5%.",
)
_m[6].metric(
    "In-domain speeches",
    session.scalar(
        select(func.count()).select_from(Statement).where(Statement.source_type == "speech")
    ),
    help="Central-bank speeches (source_type='speech') — the in-domain population paired with "
    "their mandated market in the Phase 1 Results tab. Ceremonial/other docs are excluded.",
)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_results, tab_clusters, tab_sentiment, tab_context, tab_speeches, tab_adhoc = st.tabs(
    [
        "Phase 1 Results",
        "Topic Clusters",
        "Sentiment Over Time",
        "Macro Context",
        "Speech Browser",
        "Ad-hoc Scoring",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Phase 1 scatter chart
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("Statement Signal vs Subsequent Market Move")
    st.caption(
        "Each point is **one in-domain speech**, paired with the **next trading-day move** "
        "of the single market its speaker plausibly affects (Fed→S&P 500, ECB→Euro Stoxx, "
        "BoJ→Nikkei, RBA→ASX, …). **X-axis:** the speech's sentiment (−1 negative … +1 "
        "positive). **Y-axis:** that market's return on the first session *strictly after* "
        "the speech (%). This is relevance-aware and point-in-time — one row per speech, no "
        "look-ahead — replacing the retired geography-blind 48 h window attribution. Daily "
        "data, so a day's move is a noisy proxy for one speech: read as **exploratory**."
    )

    df_pairs = statement_market_pairs(session)

    if df_pairs.empty:
        st.info(
            "No relevance-aware pairs yet. Needs scored speeches (steps 3,4) and price data "
            "(step 5) for the mapped indices."
        )
    else:
        df_pairs = df_pairs.dropna(subset=["ret_pct", "sentiment_score"]).copy()
        fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
        with fcol1:
            market_opts = ["All"] + sorted(df_pairs["market"].unique())
            sel_market = st.selectbox(
                "Filter by market (speaker's mandated index)",
                market_opts,
                key="results_market",
                help="Each speaker is paired only with the index its institution plausibly moves.",
            )
        with fcol2:
            hide_neutral_sent_r = st.checkbox(
                "Hide neutral sentiment",
                value=False,
                key="results_hide_sent",
                help="Drop points whose FinBERT sentiment label is neutral.",
            )
        with fcol3:
            hide_neutral_stance_r = st.checkbox(
                "Hide neutral stance",
                value=False,
                key="results_hide_stance",
                help="Drop points whose FOMC-RoBERTa stance is neutral.",
            )
        view = df_pairs if sel_market == "All" else df_pairs[df_pairs["market"] == sel_market]
        if hide_neutral_sent_r:
            view = view[view["sentiment_label"] != "neutral"]
        if hide_neutral_stance_r:
            view = view[view["hawkish_label"] != "neutral"]

        if len(view) < 3:
            st.info("Too few pairs to correlate for this selection.")
        else:
            r_dir, p_dir = stats.pearsonr(view["sentiment_score"], view["ret_pct"])
            r_spear, p_spear = stats.spearmanr(view["sentiment_score"], view["ret_pct"])
            n_days = view["date"].nunique()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "N (speech→market pairs)",
                len(view),
                help="One row per in-domain speech mapped to its relevant index (no replication).",
            )
            c2.metric(
                "Distinct market-days",
                n_days,
                help="Independent units. N pairs ≫ market-days means clustering inflates naive "
                "p-values — treat significance cautiously.",
            )
            c3.metric(
                "Pearson r — directional",
                f"{r_dir:.3f}",
                help=f"Linear correlation between speech sentiment and the next-day signed move. "
                f"Range −1 to +1. p = {p_dir:.4f}",
            )
            c4.metric(
                "Spearman r",
                f"{r_spear:.3f}",
                help=f"Rank-order correlation, more robust to outliers. p = {p_spear:.4f}",
            )

            sig = "significant (p < 0.05)" if p_dir < 0.05 else "not significant (p >= 0.05)"
            st.caption(
                f"Directional Pearson: {sig} — p = {p_dir:.4f}. "
                f"Naive p-values assume independence; with {len(view)} pairs over only "
                f"{n_days} market-days, prefer day-clustered inference (see analyze_habituation.py)."
            )

            fig = px.scatter(
                view,
                x="sentiment_score",
                y="ret_pct",
                color="market" if sel_market == "All" else None,
                hover_data={"timestamp": "|%Y-%m-%d", "person": True, "market": True},
                labels={
                    "sentiment_score": "Speech sentiment  (−1 negative … +1 positive)",
                    "ret_pct": "Next-day move of speaker's market (%)",
                    "market": "Market",
                    "person": "Speaker",
                },
                trendline="ols",
                trendline_scope="overall",
                trendline_color_override="#1d4ed8",
                height=520,
            )
            fig.update_traces(marker=dict(size=7, opacity=0.7))
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
            fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
            st.plotly_chart(fig, width="stretch")

            if sel_market == "All":
                st.subheader("Results by Market")
                st.caption("Pearson r per mapped index (markets with fewer than 3 pairs excluded).")
                breakdown = []
                for grp, gdf in view.groupby("market"):
                    if len(gdf) < 3:
                        continue
                    rp, pp = stats.pearsonr(gdf["sentiment_score"], gdf["ret_pct"])
                    breakdown.append(
                        {
                            "Market": grp,
                            "Pairs (N)": len(gdf),
                            "Pearson r": round(rp, 3),
                            "p-value": round(pp, 4),
                            "Significant (p < 0.05)": "yes" if pp < 0.05 else "no",
                        }
                    )
                if breakdown:
                    st.dataframe(pd.DataFrame(breakdown), hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Topic clusters
# ══════════════════════════════════════════════════════════════════════════════
with tab_clusters:
    st.subheader("Speech Topic Clusters")
    st.caption(
        "Topics from **BERTopic** (sentence-transformer embeddings + UMAP + HDBSCAN + class-based "
        "TF-IDF). Each topic has a **secondary** label (its most distinctive words) and a **main "
        "headline** (a broad theme). Points are coloured by main headline — filter by headline, then "
        "drill into a specific topic. See the Glossary for BERTopic/UMAP/c-TF-IDF."
    )

    if not _clustering_cols_exist:
        cluster_rows = []
    else:
        cluster_rows = session.execute(
            select(
                StatementAnalysis.topic_main,
                StatementAnalysis.topic_classification,
                StatementAnalysis.umap_x,
                StatementAnalysis.umap_y,
                StatementAnalysis.sentiment_score,
                Statement.published_at,
                Statement.raw_text,
                Person.canonical_name.label("person"),
                Person.institution,
            )
            .join(Statement, StatementAnalysis.statement_id == Statement.id)
            .join(Person, Statement.person_id == Person.id)
            .where(StatementAnalysis.cluster_id.isnot(None))
            .where(StatementAnalysis.umap_x.isnot(None))
        ).all()

    if not cluster_rows:
        st.info(
            "Clusters not yet computed. Run:\n"
            "```\n"
            "psql $DATABASE_URL < db/migrations/001_add_clustering_columns.sql\n"
            "psql $DATABASE_URL < db/migrations/003_add_topic_main.sql\n"
            "python scripts/cluster_speeches.py\n"
            "```"
        )
    else:
        df_cl = pd.DataFrame(
            cluster_rows,
            columns=[
                "main",
                "secondary",
                "umap_x",
                "umap_y",
                "sentiment_score",
                "published_at",
                "raw_text",
                "person",
                "institution",
            ],
        )
        df_cl["main"] = df_cl["main"].fillna("Other")
        df_cl["secondary"] = df_cl["secondary"].fillna("general")
        df_cl["published_at"] = pd.to_datetime(df_cl["published_at"], utc=True).dt.date
        df_cl["preview"] = df_cl["raw_text"].str[:120] + "…"

        # Two-level filter. Colour is driven by the filter: focusing on a single main
        # theme colours by its BERTopic sub-topics — otherwise every monetary-policy
        # speech is one colour and looks homogeneous despite being split underneath.
        fc1, fc2 = st.columns(2)
        with fc1:
            main_opts = sorted(df_cl["main"].unique())
            sel_main = st.multiselect("Main theme (empty = all)", main_opts, default=[])
        scope = df_cl if not sel_main else df_cl[df_cl["main"].isin(sel_main)]
        with fc2:
            sec_opts = sorted(scope["secondary"].unique())
            sel_sec = st.multiselect("Sub-topic (empty = all)", sec_opts, default=[])
        df_show = scope if not sel_sec else scope[scope["secondary"].isin(sel_sec)]

        color_field = "secondary" if len(sel_main) == 1 else "main"
        color_by = "Sub-topic" if color_field == "secondary" else "Main theme"
        n_colors = df_show[color_field].nunique()
        fig_cl = px.scatter(
            df_show,
            x="umap_x",
            y="umap_y",
            color=color_field,
            hover_data={
                "main": True,
                "secondary": True,
                "person": True,
                "institution": True,
                "published_at": True,
                "preview": True,
                "umap_x": False,
                "umap_y": False,
            },
            labels={
                "umap_x": "Semantic space — horizontal (UMAP 1)",
                "umap_y": "Semantic space — vertical (UMAP 2)",
                "main": "Main theme",
                "secondary": "Sub-topic",
            },
            height=560,
        )
        fig_cl.update_traces(marker=dict(size=5, opacity=0.75))
        fig_cl.update_layout(
            showlegend=n_colors <= 18,  # hide when too many to be a usable key (rely on hover)
            legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title_text=color_by),
            margin=dict(b=30),
        )
        st.plotly_chart(fig_cl, width="stretch")
        st.caption(
            f"Colour = {color_by.lower()}. Select one main theme to fan it into its BERTopic "
            "sub-topics — e.g. Monetary policy splits into inflation / financial stability / "
            "growth & labour / housing / CBDC / FX. Axis distance is what matters; hover for details."
        )

        st.subheader("Cluster Summary")
        st.caption("Grouped main theme → secondary context. Sentiment −1 (neg) to +1 (pos).")
        summary = (
            df_cl.groupby(["main", "secondary"])
            .agg(
                Speeches=("person", "count"),
                Speakers=("person", "nunique"),
                **{"Avg sentiment": ("sentiment_score", "mean")},
                Earliest=("published_at", "min"),
                Latest=("published_at", "max"),
            )
            .round(3)
            .sort_values("Speeches", ascending=False)
            .reset_index()
            .rename(columns={"main": "Main theme", "secondary": "Secondary context"})
        )
        st.dataframe(summary, hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Sentiment over time
# ══════════════════════════════════════════════════════════════════════════════
with tab_sentiment:
    st.subheader("Statement Sentiment — Individual Speeches")
    st.caption(
        "Each point is one speech, **labelled by speaker** (hover, or click for the full "
        "details below). Y-axis: sentiment −1 (negative) · 0 (neutral) · +1 (positive). "
        "The lines above the data mark macro-context regimes (crises, wars, policy cycles) "
        "coloured by category. Use the filters to narrow by country, institution, source, "
        "tier, period, or sentiment."
    )

    query = (
        select(
            SentimentSignalRecord.timestamp,
            SentimentSignalRecord.statement_sentiment,
            Person.canonical_name.label("person"),
            Person.institution,
            Statement.raw_text,
            Statement.source_type,
            Statement.influence_tier,
            StatementAnalysis.topic_classification,
            StatementAnalysis.sentiment_label,
            StatementAnalysis.hawkish_label,
        )
        .join(Person, SentimentSignalRecord.person_id == Person.id)
        .join(Statement, SentimentSignalRecord.statement_id == Statement.id)
        .outerjoin(StatementAnalysis, StatementAnalysis.statement_id == Statement.id)
        .order_by(SentimentSignalRecord.timestamp)
    )
    if selected_person != "All":
        person_obj = next((p for p in persons if p.canonical_name == selected_person), None)
        if person_obj:
            query = query.where(SentimentSignalRecord.person_id == person_obj.id)

    rows = session.execute(query).all()
    if not rows:
        st.info("No scored statements yet. Run: python scripts/run_phase1.py --steps 3,4")
    else:
        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "statement_sentiment",
                "person",
                "institution",
                "raw_text",
                "source_type",
                "influence_tier",
                "cluster",
                "sentiment_label",
                "hawkish_label",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["institution"] = df["institution"].fillna("Unknown")
        df["country"] = df["institution"].map(country_for_institution)
        df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        df["text_short"] = df["raw_text"].str[:400] + "…"
        df = df.sort_values("timestamp")

        # ── Filters: narrow the data instead of hiding/bucketing it ─────────────
        with st.expander("Filters", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                country_opts = sorted(df["country"].unique())
                sel_countries = st.multiselect("Country / bloc", country_opts, default=country_opts)
                inst_opts = sorted(df["institution"].unique())
                sel_insts = st.multiselect("Institution", inst_opts, default=inst_opts)
            with fc2:
                src_opts = sorted(df["source_type"].dropna().unique())
                sel_srcs = st.multiselect("Source type", src_opts, default=src_opts)
                tier_opts = sorted(int(t) for t in df["influence_tier"].dropna().unique())
                sel_tiers = st.multiselect(
                    "Influence tier (1 = head)", tier_opts, default=tier_opts
                )
            with fc3:
                yr_lo, yr_hi = (
                    int(df["timestamp"].dt.year.min()),
                    int(df["timestamp"].dt.year.max()),
                )
                yr_sel = (
                    st.slider("Year range", yr_lo, yr_hi, (yr_lo, yr_hi))
                    if yr_hi > yr_lo
                    else (yr_lo, yr_hi)
                )
                sent_sel = st.slider("Sentiment range", -1.0, 1.0, (-1.0, 1.0), 0.05)
                regime_sel = st.selectbox(
                    "Only during regime", ["(any)"] + [p.name for p in context_periods]
                )
                hide_neutral_sent_t = st.checkbox(
                    "Hide neutral sentiment",
                    value=False,
                    key="time_hide_sent",
                    help="Drop speeches whose FinBERT sentiment label is neutral.",
                )
                hide_neutral_stance_t = st.checkbox(
                    "Hide neutral stance",
                    value=False,
                    key="time_hide_stance",
                    help="Drop speeches whose FOMC-RoBERTa stance is neutral. "
                    "Non-monetary documents (no stance) stay visible.",
                )

        mask = (
            df["country"].isin(sel_countries)
            & df["institution"].isin(sel_insts)
            & df["source_type"].isin(sel_srcs)
            & df["influence_tier"].isin(sel_tiers)
            & df["timestamp"].dt.year.between(yr_sel[0], yr_sel[1])
            & df["statement_sentiment"].between(sent_sel[0], sent_sel[1])
        )
        if hide_neutral_sent_t:
            mask &= df["sentiment_label"] != "neutral"
        if hide_neutral_stance_t:
            mask &= df["hawkish_label"] != "neutral"
        if regime_sel != "(any)":
            rp = next((p for p in context_periods if p.name == regime_sel), None)
            if rp is not None:
                r0 = pd.to_datetime(rp.start_date, utc=True)
                r1 = pd.to_datetime(rp.end_date, utc=True) if rp.end_date else df["timestamp"].max()
                mask &= df["timestamp"].between(r0, r1)
        view = df[mask]
        st.caption(f"{len(view)} of {len(df)} speeches match the filters.")

        if view.empty:
            st.info("No speeches match the current filters — widen them above.")
        else:
            people = sorted(view["person"].unique())
            palette = px.colors.qualitative.Dark24
            person_colors = {pp: palette[i % len(palette)] for i, pp in enumerate(people)}
            regime_categories = sorted({p.category for p in context_periods})
            default_cats = sorted({p.category for p in context_periods if p.impact_tier == 1})

            # Selections come from the clickable lists rendered BELOW the chart. Read the
            # previous run's state here so those lists drive the chart (Streamlit reruns
            # on every click), then render the widgets lower down with the same keys.
            sel_cats = [
                c
                for c in st.session_state.get("sel_regime_cats", default_cats)
                if c in regime_categories
            ]
            sel_speakers = [s for s in st.session_state.get("sel_speakers", []) if s in people]
            plot_view = view[view["person"].isin(sel_speakers)] if sel_speakers else view

            fig_px = px.scatter(
                plot_view,
                x="timestamp",
                y="statement_sentiment",
                color="person",
                color_discrete_map=person_colors,
                custom_data=[
                    "person",
                    "institution",
                    "date_str",
                    "statement_sentiment",
                    "text_short",
                ],
            )
            fig_px.update_traces(
                marker=dict(size=6, opacity=0.75),
                hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                "%{customdata[2]} · sentiment %{y:.3f}<extra></extra>",
            )

            y_label = "Sentiment score  (−1 neg · 0 neutral · +1 pos)"
            cats_present = [c for c in regime_categories if c in sel_cats]
            chosen = [p for p in context_periods if p.category in cats_present]
            if cats_present and chosen and not plot_view.empty:
                # One lane per regime CATEGORY in a strip above the scatter (separate
                # subplot row, shared x). The strip height scales with the number of
                # categories, so it never compresses the scatter below.
                cat_lane = {c: i for i, c in enumerate(cats_present)}
                strip_px = 26 * len(cats_present) + 16
                fig_scatter = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    row_heights=[strip_px / (strip_px + 400), 400 / (strip_px + 400)],
                    vertical_spacing=0.04,
                )
                for tr in fig_px.data:
                    fig_scatter.add_trace(tr, row=2, col=1)

                xmax = plot_view["timestamp"].max()
                for p in chosen:
                    start = pd.to_datetime(p.start_date, utc=True)
                    end = pd.to_datetime(p.end_date, utc=True) if p.end_date else xmax
                    color = CONTEXT_CATEGORY_COLORS.get(p.category, "#9ca3af")
                    end_label = "ongoing" if p.end_date is None else end.date()
                    hover = f"<b>{p.name}</b><br>{p.category}<br>{start.date()} – {end_label}"
                    xs = list(pd.date_range(start, end, periods=12))
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=xs,
                            y=[cat_lane[p.category]] * len(xs),
                            mode="lines",
                            line=dict(color=color, width=4),
                            hovertemplate=hover + "<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                # Strip y-axis: one labelled lane per category; the scatter keeps its
                # own natural y-range below.
                fig_scatter.update_yaxes(
                    row=1,
                    col=1,
                    range=[-0.6, len(cats_present) - 0.4],
                    tickvals=list(range(len(cats_present))),
                    ticktext=[c.replace("_", " ") for c in cats_present],
                    tickfont=dict(size=9),
                    showgrid=False,
                    zeroline=False,
                )
                fig_scatter.update_yaxes(row=2, col=1, title_text=y_label)
                fig_scatter.update_xaxes(row=2, col=1, title_text="Date")
                fig_scatter.add_hline(
                    y=0, line_dash="dot", line_color="gray", opacity=0.4, row=2, col=1
                )
                fig_scatter.update_layout(
                    showlegend=False, height=strip_px + 400, margin=dict(t=10, b=10)
                )
            else:
                fig_scatter = fig_px
                fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig_scatter.update_yaxes(title_text=y_label)
                fig_scatter.update_xaxes(title_text="Date")
                fig_scatter.update_layout(showlegend=False, height=480, margin=dict(t=10, b=10))

            event = st.plotly_chart(
                fig_scatter, width="stretch", on_select="rerun", key="sentiment_scatter"
            )

            # ── Clickable lists that drive the chart (selections read at the top next run) ──
            list_regimes, list_speakers = st.columns([1, 2])
            with list_regimes:
                st.caption("Macro regimes — click a category to show its markers")
                st.pills(
                    "regime categories",
                    regime_categories,
                    selection_mode="multi",
                    default=default_cats,
                    format_func=lambda c: c.replace("_", " "),
                    key="sel_regime_cats",
                    label_visibility="collapsed",
                )
            with list_speakers:
                st.caption("Speakers — click to focus the chart (none selected = show all)")
                st.pills(
                    "speakers",
                    people,
                    selection_mode="multi",
                    key="sel_speakers",
                    label_visibility="collapsed",
                )

            # Click a point -> details in a fixed panel below (never over the data)
            points = []
            try:
                points = event["selection"]["points"]
            except (TypeError, KeyError, AttributeError):
                sel = getattr(event, "selection", None)
                points = getattr(sel, "points", None) or []
            speech_points = [pt for pt in points if pt.get("customdata")]
            if speech_points:
                person, inst, date_str, score, text = speech_points[0]["customdata"][:5]
                try:
                    score_txt = f"{float(score):+.3f}"
                except (TypeError, ValueError):
                    score_txt = "n/a"
                st.markdown(f"**{person}** · {inst} · {date_str} · sentiment `{score_txt}`")
                st.text_area(
                    "Selected speech (preview)",
                    text,
                    height=120,
                    disabled=True,
                    key="sentiment_selected_text",
                )
                st.caption("Full text is in the Speech Browser tab.")
            else:
                st.caption("Click a point to see its speech details here.")

            st.divider()
            st.subheader("30-Day Rolling Mean Sentiment")
            st.caption("Smoothed average of the shown speeches within each 30-day window.")
            df_ts = plot_view.set_index("timestamp")[["statement_sentiment"]]
            st.line_chart(df_ts["statement_sentiment"].resample("30D").mean().dropna())

            st.subheader("Average Sentiment by Speaker")
            st.caption("Mean / count / variability of sentiment per speaker, over the shown set.")
            by_person = (
                plot_view.groupby("person")["statement_sentiment"]
                .agg(["mean", "count", "std"])
                .rename(
                    columns={
                        "mean": "avg sentiment (−1 to +1)",
                        "count": "n speeches",
                        "std": "std dev",
                    }
                )
                .sort_values("avg sentiment (−1 to +1)", ascending=False)
                .round(3)
            )
            st.dataframe(by_person, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Macro context
# ══════════════════════════════════════════════════════════════════════════════
with tab_context:
    st.subheader("Macro Context — High-Impact Events & Regimes")
    st.caption(
        "Curated catalog of high-impact real-world events (wars, pandemics, policy "
        "regimes, crises) with time windows. A statement, market event, or signal is "
        "'in' a period when its timestamp falls inside the window. Use these to "
        "stratify analysis by regime — e.g. does pre-event sentiment predict moves "
        "during crises but not in calm periods?"
    )

    if not context_periods:
        st.info(
            "Context periods not yet loaded. Run:\n"
            "```\n"
            "psql $DATABASE_URL < db/migrations/002_add_context_periods.sql\n"
            "python scripts/seed_context_periods.py\n"
            "```"
        )
    else:
        now_ts = pd.Timestamp.now(tz="UTC")
        df_ctx = pd.DataFrame(
            [
                {
                    "name": p.name,
                    "category": p.category,
                    "start": p.start_date,
                    "end": p.end_date or now_ts,
                    "ongoing": p.end_date is None,
                    "impact_tier": p.impact_tier,
                    "geography": p.geography,
                    "description": p.description,
                }
                for p in context_periods
            ]
        )
        # px.timeline cannot handle timezone-aware datetimes — normalise to tz-naive
        df_ctx["start"] = pd.to_datetime(df_ctx["start"], utc=True).dt.tz_localize(None)
        df_ctx["end"] = pd.to_datetime(df_ctx["end"], utc=True).dt.tz_localize(None)

        # Gantt-style timeline of all periods, coloured by category
        fig_tl = px.timeline(
            df_ctx,
            x_start="start",
            x_end="end",
            y="name",
            color="category",
            hover_data={
                "description": True,
                "impact_tier": True,
                "geography": True,
                "start": True,
                "end": True,
                "name": False,
            },
            labels={"name": "", "category": "Category"},
            height=max(400, 22 * len(df_ctx)),
        )
        fig_tl.update_yaxes(autorange="reversed")
        fig_tl.add_vline(x=now_ts.tz_convert(None), line_dash="dot", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_tl, width="stretch")
        st.caption(
            "Bars extending to the right edge are ongoing (end date open). Dotted line = today."
        )

        # Counts by category
        st.subheader("Catalog")
        cat_counts = (
            df_ctx["category"].value_counts().rename_axis("category").reset_index(name="count")
        )
        st.dataframe(cat_counts, hide_index=True, width="content")

        # Full catalog table
        table = df_ctx[
            [
                "name",
                "category",
                "start",
                "end",
                "ongoing",
                "impact_tier",
                "geography",
                "description",
            ]
        ].copy()
        table["start"] = table["start"].dt.date
        table["end"] = table["end"].dt.date
        st.dataframe(table, hide_index=True, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Speech browser
# ══════════════════════════════════════════════════════════════════════════════
with tab_speeches:
    st.subheader("Speech Browser")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        browser_person = st.selectbox("Speaker", person_names, key="browser_person")
    with col2:
        sentiment_filter = st.select_slider(
            "Sentiment score range",
            options=[-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
            value=(-1.0, 1.0),
            help="Filter by FinBERT sentiment score. −1 = most negative language, +1 = most positive.",
        )
    with col3:
        search_text = st.text_input("Search text", placeholder="e.g. inflation, rate hike")

    # Two independent toggles — hide neutral by sentiment and/or by stance, separately.
    tcol1, tcol2, _tcol3 = st.columns([2, 2, 2])
    with tcol1:
        hide_neutral_sentiment = st.checkbox(
            "Hide neutral sentiment",
            value=False,
            help="Drop speeches whose FinBERT sentiment label is neutral.",
        )
    with tcol2:
        hide_neutral_stance = st.checkbox(
            "Hide neutral stance",
            value=False,
            help="Drop speeches whose FOMC-RoBERTa hawkish/dovish stance is neutral. "
            "Non-monetary documents (which have no stance) stay visible.",
        )

    speech_query = (
        select(
            Statement.id,
            Statement.published_at,
            Statement.url,
            Statement.source_type,
            Person.canonical_name.label("person"),
            StatementAnalysis.sentiment_score,
            StatementAnalysis.sentiment_label,
            StatementAnalysis.hawkish_score,
            StatementAnalysis.hawkish_label,
            StatementAnalysis.topic_classification,
            Statement.raw_text,
        )
        .join(Person, Statement.person_id == Person.id)
        .join(StatementAnalysis, StatementAnalysis.statement_id == Statement.id)
        .where(StatementAnalysis.sentiment_score >= sentiment_filter[0])
        .where(StatementAnalysis.sentiment_score <= sentiment_filter[1])
        .order_by(Statement.published_at.desc())
    )
    if browser_person != "All":
        person_obj = next((p for p in persons if p.canonical_name == browser_person), None)
        if person_obj:
            speech_query = speech_query.where(Statement.person_id == person_obj.id)
    if search_text.strip():
        speech_query = speech_query.where(Statement.raw_text.ilike(f"%{search_text.strip()}%"))
    # Independent neutral filters. is_distinct_from keeps NULL-stance (non-monetary)
    # rows visible when hiding neutral stance — `!= 'neutral'` would drop them.
    if hide_neutral_sentiment:
        speech_query = speech_query.where(
            StatementAnalysis.sentiment_label.is_distinct_from("neutral")
        )
    if hide_neutral_stance:
        speech_query = speech_query.where(
            StatementAnalysis.hawkish_label.is_distinct_from("neutral")
        )

    speech_rows = session.execute(speech_query.limit(200)).all()
    st.caption(f"{len(speech_rows)} speeches shown (max 200)")

    if speech_rows:
        df_sp = pd.DataFrame(
            speech_rows,
            columns=[
                "id",
                "date",
                "url",
                "source_type",
                "person",
                "sentiment_score",
                "sentiment_label",
                "hawkish_score",
                "hawkish_label",
                "topic_cluster",
                "raw_text",
            ],
        )
        df_sp["date"] = pd.to_datetime(df_sp["date"], utc=True).dt.date
        df_sp["sentiment_score"] = df_sp["sentiment_score"].round(3)
        df_sp["hawkish_score"] = df_sp["hawkish_score"].round(3)
        df_sp["preview"] = df_sp["raw_text"].str[:200] + "…"

        st.dataframe(
            df_sp[
                [
                    "date",
                    "person",
                    "sentiment_score",
                    "sentiment_label",
                    "hawkish_score",
                    "hawkish_label",
                    "topic_cluster",
                    "preview",
                    "url",
                ]
            ],
            width="stretch",
            column_config={
                "url": st.column_config.LinkColumn("Link"),
                "sentiment_score": st.column_config.NumberColumn(
                    "Sentiment score",
                    format="%.3f",
                    help="FinBERT financial sentiment: −1 = most negative, 0 = neutral, +1 = most positive.",
                ),
                "sentiment_label": st.column_config.TextColumn(
                    "Sentiment label",
                    help="Categorical class assigned by FinBERT: positive, neutral, or negative.",
                ),
                "hawkish_score": st.column_config.NumberColumn(
                    "Hawkish score",
                    format="%.3f",
                    help="FOMC-RoBERTa monetary-policy stance: +1 = hawkish, −1 = dovish, 0 = neutral. "
                    "Central-bank speeches only; blank for non-monetary documents.",
                ),
                "hawkish_label": st.column_config.TextColumn(
                    "Stance",
                    help="FOMC-RoBERTa stance class: hawkish, neutral, or dovish (central-bank speeches only).",
                ),
                "topic_cluster": st.column_config.TextColumn(
                    "Topic cluster",
                    help="BERTopic secondary label: the most distinctive words (class-based TF-IDF) of this speech's topic. "
                    "Run scripts/cluster_speeches.py to populate.",
                ),
            },
        )

        st.subheader("Full Text")
        selected_idx = st.selectbox(
            "Select speech",
            options=df_sp.index,
            format_func=lambda i: (
                f"{df_sp.loc[i, 'date']}  {df_sp.loc[i, 'person']}  "
                f"(sentiment: {df_sp.loc[i, 'sentiment_score']:+.3f})"
            ),
        )
        row = df_sp.loc[selected_idx]
        st.markdown(
            f"**{row['person']}** · {row['date']} · "
            f"sentiment score: `{row['sentiment_score']:+.3f}` · "
            f"[source]({row['url']})"
        )
        st.text_area("Full speech text", value=row["raw_text"], height=400, disabled=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Ad-hoc scoring
# ══════════════════════════════════════════════════════════════════════════════
with tab_adhoc:
    st.subheader("Ad-hoc NLP Scoring")
    st.caption(
        "Paste any text to score it with FinBERT. Long text is split into 512-token "
        "chunks and aggregated, so the whole document is scored (not just the opening). "
        "The bar chart shows the aggregated probability per class; the sentiment score "
        "is P(positive) − P(negative), ranging from −1 to +1."
    )
    input_text = st.text_area("Text to score", height=150)
    if st.button("Analyze") and input_text.strip():
        with st.spinner("Running FinBERT…"):
            from sentiment_signal.nlp.pipeline import NLPPipeline

            pipe = NLPPipeline()
            result = pipe.analyze_documents([input_text])[0]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Sentiment Score",
            f"{result['sentiment_score']:+.3f}",
            help="P(positive) − P(negative). Range −1 to +1.",
        )
        c2.metric(
            "Label",
            result["sentiment_label"].upper(),
            help="The sentiment class with the highest aggregated probability.",
        )
        c3.metric(
            "Chunks",
            result["n_chunks"],
            help="Number of 512-token chunks the text was split into and aggregated over.",
        )
        # Label the bars in the model's actual class order (not a hardcoded guess)
        id2label = pipe.model.config.id2label
        labels = [id2label[i].lower() for i in range(len(result["probs"]))]
        fig = px.bar(
            x=labels,
            y=result["probs"],
            labels={"x": "Sentiment class", "y": "Model probability (0 to 1)"},
            color=labels,
            color_discrete_map={"positive": "#16a34a", "neutral": "#6b7280", "negative": "#dc2626"},
            height=300,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
