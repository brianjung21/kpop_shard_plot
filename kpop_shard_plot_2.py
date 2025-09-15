"""
kpop_shard_plot_2.py

K-pop Reddit brand intelligence dashboard:
- Core: Load pivoted daily/weekly brand mention counts, plot Raw or SoV, top brands, top posts/subreddits
- New: Spike detector + drivers, Engagement-weighted mentions, Subreddit mix/diffusion,
       Sentiment overlay (optional, if cache exists), Momentum vs Attention quadrant

Assumptions:
- Pivot file: kpop_alias_pivoted_brand_counts.csv with columns ['date', brand1, brand2, ...]
- Raw matches (optional): reddit_matches_raw.csv with at least ['date','keyword','alias','subreddit','title','score','num_comments']
- Optional sentiment cache (if you precomputed): sentiment_cache.csv with ['id' or 'hash','date','keyword','compound']

Data directory:
- Looks for files in:  Path(__file__).parent.parent / "data"
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# Config & file discovery
# -----------------------------
DATA_CANDIDATES = [
    Path(__file__).parent.parent / "data"
]
PIVOT_FILE = "kpop_alias_pivoted_brand_counts.csv"
RAW_MATCHES = "reddit_matches_raw.csv"
SENTIMENT_CACHE = "sentiment_cache.csv"  # optional (see notes below)

DEFAULT_TOPN = 5

# -----------------------------
# Helpers
# -----------------------------
def find_file(fname: str) -> Path:
    for base in DATA_CANDIDATES:
        p = base / fname
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {fname} in any of: {[str(b) for b in DATA_CANDIDATES]}"
    )

@st.cache_data(show_spinner=False)
def load_pivot() -> pd.DataFrame:
    p = find_file(PIVOT_FILE)
    df = pd.read_csv(p, parse_dates=['date'])
    for c in df.columns:
        if c != 'date':
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df.sort_values('date').reset_index(drop=True)

def resample_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return df.set_index("date").asfreq("D").fillna(0).reset_index()
    return (df.set_index("date").resample("W").sum().reset_index())

def to_long(df: pd.DataFrame, brands: List[str]) -> pd.DataFrame:
    cols = ['date'] + brands
    sub = df[cols].copy()
    return sub.melt(id_vars='date', var_name='brand', value_name='mentions')

def default_top_brands(df_freq: pd.DataFrame, topn: int) -> List[str]:
    if df_freq.empty:
        return []
    totals = df_freq.drop(columns=['date']).sum(axis=0).sort_values(ascending=False)
    return totals.head(topn).index.tolist()

def compute_sov(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Share of Voice (%) from long df [date|brand|mentions]."""
    totals = (long_df.groupby("date", as_index=False)["mentions"].sum()
                      .rename(columns={"mentions": "total"}))
    x = long_df.merge(totals, on="date", how="left")
    x["value"] = (x["mentions"] / x["total"]).fillna(0) * 100.0
    return x[["date", "brand", "value"]]

def compute_window_sov(df_window: pd.DataFrame, brands_to_show: List[str]) -> pd.DataFrame:
    """Window SoV = 100 * sum(brand_mentions) / sum(total_mentions_all_brands) over the window."""
    if df_window.empty:
        return pd.DataFrame({"brand": [], "value": []})
    totals_by_date = df_window.drop(columns=["date"]).sum(axis=1).rename("total")
    totals_df = pd.DataFrame({"date": df_window["date"], "total": totals_by_date})
    brands = [b for b in brands_to_show if b in df_window.columns]
    if not brands:
        return pd.DataFrame({"brand": [], "value": []})
    m = df_window[["date"] + brands].melt("date", var_name="brand", value_name="mentions")
    m = m.merge(totals_df, on="date", how="left")
    sums = m.groupby("brand", as_index=False).agg({"mentions": "sum", "total": "sum"})
    sums["value"] = (sums["mentions"] / sums["total"]).fillna(0) * 100.0
    return sums[["brand", "value"]]

def apply_rolling(long_df: pd.DataFrame, freq: str, win_daily=7, win_weekly=4) -> pd.DataFrame:
    win = win_daily if freq == "D" else win_weekly
    if win <= 1:
        long_df['smoothed'] = long_df['mentions']
        return long_df
    out = []
    for b, g in long_df.groupby("brand", as_index=False):
        g = g.sort_values('date').copy()
        g['smoothed'] = g['mentions'].rolling(win, min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_raw_matches_or_none() -> Optional[pd.DataFrame]:
    try:
        p = find_file(RAW_MATCHES)
    except FileNotFoundError:
        return None
    try:
        raw = pd.read_csv(p, parse_dates=['date'])
        keep = ['date', 'keyword', 'alias', 'subreddit', 'title', 'score', 'num_comments']
        cols = [c for c in keep if c in raw.columns]
        return raw[cols].copy().sort_values(['date', 'score'], ascending=[True, False])
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_sentiment_cache_or_none() -> Optional[pd.DataFrame]:
    """
    Load and normalize sentiment cache to minimal schema: ['date','brand','compound'].
    Accepts either 'compound' (transformer) or 'compound_vader' fallback.
    Creates 'brand' from 'keyword' if present; otherwise keeps existing 'brand'.
    """
    try:
        p = find_file(SENTIMENT_CACHE)
    except FileNotFoundError:
        return None
    try:
        sc = pd.read_csv(p, parse_dates=['date'])
    except Exception:
        return None

    # Determine compound column
    compound_col = None
    if 'compound' in sc.columns:
        compound_col = 'compound'
    elif 'compound_vader' in sc.columns:
        compound_col = 'compound_vader'
    else:
        return None  # no usable sentiment signal

    # Determine brand column
    if 'brand' in sc.columns:
        sc['brand'] = sc['brand'].astype(str)
    elif 'keyword' in sc.columns:
        sc['brand'] = sc['keyword'].astype(str)
    elif 'alias' in sc.columns:
        sc['brand'] = sc['alias'].astype(str)
    else:
        return None

    # Keep only necessary columns and clean
    keep = ['date', 'brand', compound_col]
    sc = sc[keep].rename(columns={compound_col: 'compound'})
    sc = sc.dropna(subset=['date', 'brand', 'compound']).copy()

    # Ensure date is naive timestamp and compound clipped to [-1,1]
    sc['date'] = pd.to_datetime(sc['date']).dt.tz_localize(None)
    sc['compound'] = sc['compound'].astype(float).clip(-1.0, 1.0)

    return sc

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="K-pop Reddit Intelligence (Sharded)", layout="wide")
st.title("K-pop Reddit Intelligence — Sharded (Interactive)")

try:
    df = load_pivot()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

raw = load_raw_matches_or_none()
sent_cache = load_sentiment_cache_or_none()

all_brands = [c for c in df.columns if c != "date"]

with st.sidebar:
    st.header("Controls")
    freq_label = st.radio("Frequency", ["Daily", "Weekly"], index=0)
    freq = "D" if freq_label == "Daily" else "W"

    df_freq = resample_freq(df, freq)

    norm_mode = st.radio("Normalization", ["Raw counts", "Share of Voice (SoV)"], index=0)

    # Date window
    min_d, max_d = df_freq["date"].min().date(), df_freq["date"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, date_range

    mask = (df_freq["date"].dt.date >= start_date) & (df_freq["date"].dt.date <= end_date)
    df_win = df_freq.loc[mask].copy()

    defaults = default_top_brands(df_win, DEFAULT_TOPN)
    selected = st.multiselect("Brands", options=all_brands, default=defaults)

    do_smooth = st.checkbox("Show rolling average", value=True)
    win_daily = st.number_input("Rolling window (days)", min_value=2, max_value=30, value=7) if (do_smooth and freq == "D") else 7
    win_weekly = st.number_input("Rolling window (weeks)", min_value=2, max_value=12, value=4) if (do_smooth and freq == "W") else 4

# Guards
if df_win.empty:
    st.info("No data in the selected date range.")
    st.stop()
if not selected:
    st.info("Pick at least one brand to plot from the sidebar.")
    st.stop()

# -----------------------------
# Core time-series plot (Raw or SoV)
# -----------------------------
long_df = to_long(df_win, selected)

if norm_mode == "Raw counts":
    long_metric = long_df.rename(columns={"mentions": "value"}).copy()
    y_label = "Mentions"
else:
    all_brands_in_win = [c for c in df_win.columns if c != "date"]
    long_all = to_long(df_win, all_brands_in_win)
    sov_all = compute_sov(long_all)
    long_metric = sov_all[sov_all["brand"].isin(selected)].copy()
    y_label = "Share of Voice (%)"

if do_smooth:
    long_plot = long_metric.rename(columns={"value": "mentions"})
    long_smooth = apply_rolling(long_plot, freq=freq, win_daily=win_daily, win_weekly=win_weekly)
    fig_main = px.line(
        long_smooth, x="date", y="smoothed", color="brand",
        title=f"K-pop brand trends ({freq_label}) — {norm_mode}",
        labels={"smoothed": y_label, "date": "Date"}
    )
    fig_raw = px.line(long_metric, x="date", y="value", color="brand")
    for tr in fig_raw.data:
        tr.update(opacity=0.25, line={"width": 1}, showlegend=False)
        fig_main.add_trace(tr)
else:
    fig_main = px.line(
        long_metric, x="date", y="value", color="brand",
        title=f"K-pop brand trends ({freq_label}) — {norm_mode}",
        labels={"value": y_label, "date": "Date"}
    )

fig_main.update_layout(hovermode="x unified", legend_title_text="Brand")
st.plotly_chart(fig_main, use_container_width=True)

# -----------------------------
# Summary (current window)
# -----------------------------
st.subheader("Summary (current window)")
if norm_mode == "Raw counts":
    summary = (long_metric.groupby("brand", as_index=False)["value"].sum()
                          .sort_values("value", ascending=False)
                          .rename(columns={"value": "Mentions"}))
else:
    sov_window = compute_window_sov(df_win, selected)
    summary = sov_window.sort_values("value", ascending=False).rename(columns={"value": "SoV over window (%)"})
    summary["SoV over window (%)"] = summary["SoV over window (%)"].round(2)
st.dataframe(summary, use_container_width=True)

# -----------------------------
# 1) Spike detector + drivers
# -----------------------------
st.subheader("Spike detector (z-score, weekly)")

# Build weekly from full data, then limit to current window and selected brands for performance
_df_week_full = resample_freq(df, "W")
mask_w = (_df_week_full["date"].dt.date >= start_date) & (_df_week_full["date"].dt.date <= end_date)
df_week_win = _df_week_full.loc[mask_w, ["date"] + selected].copy()

k = st.number_input("Rolling window (weeks) for baseline (k)", min_value=4, max_value=20, value=8)
z_thr = st.slider("Z-score threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
min_mentions = st.number_input("Min weekly mentions to consider a spike", min_value=0, max_value=50, value=3)
max_spikes = st.number_input("Max spikes to display", min_value=50, max_value=2000, value=400, step=50)

eps = 1e-8
spikes_list = []
weekly_long = df_week_win.melt("date", var_name="brand", value_name="mentions")

for b, g in weekly_long.groupby("brand"):
    g = g.sort_values("date").copy()
    # require some history and non-zero variance
    if len(g) < max(int(k), 6) or g["mentions"].std() == 0:
        continue
    g["mu"] = g["mentions"].rolling(int(k), min_periods=int(k)).mean()
    g["sd"] = g["mentions"].rolling(int(k), min_periods=int(k)).std()
    ok = g["sd"] > eps
    g.loc[ok, "z"] = (g.loc[ok, "mentions"] - g.loc[ok, "mu"]) / g.loc[ok, "sd"]
    hit = g[ok & (g["z"] >= float(z_thr)) & (g["mentions"] >= int(min_mentions))][["date", "brand", "mentions", "z"]]
    if not hit.empty:
        spikes_list.append(hit)

spikes = pd.concat(spikes_list) if spikes_list else pd.DataFrame(columns=["date","brand","mentions","z"])

if spikes.empty:
    st.info("No spikes detected with current parameters.")
else:
    spikes = spikes.sort_values(["z", "mentions"], ascending=[False, False]).head(int(max_spikes)).reset_index(drop=True)
    st.dataframe(spikes, use_container_width=True)

    if raw is not None and not spikes.empty:
        st.markdown("**Top posts driving a selected spike**")
        # Narrow by brand first to keep UI snappy
        bopts = sorted(spikes["brand"].unique())
        bsel = st.selectbox("Brand", bopts)
        sbrand = spikes[spikes["brand"] == bsel].reset_index(drop=True)
        pick = st.selectbox(
            "Choose spike",
            options=sbrand.index,
            format_func=lambda i: f"{sbrand.loc[i,'brand']} @ {sbrand.loc[i,'date'].date()} (z={sbrand.loc[i,'z']:.1f}, m={int(sbrand.loc[i,'mentions'])})"
        )
        sel = sbrand.loc[int(pick)]
        week_end = pd.to_datetime(sel["date"])  # resample('W') gives week end
        week_start = week_end - pd.Timedelta(days=6)
        posts = raw[(raw["keyword"] == sel["brand"]) & (raw["date"].between(week_start, week_end))]
        posts = posts.sort_values(["score", "num_comments"], ascending=False)
        st.caption(f"Top posts for {sel['brand']} during {week_start.date()} → {week_end.date()}")
        st.dataframe(posts.head(30), use_container_width=True)
    else:
        st.caption("raw matches not available; cannot show spike drivers.")

# -----------------------------
# 2) Engagement-weighted mentions
# -----------------------------
with st.expander("Engagement-weighted mentions"):
    if raw is None:
        st.info("raw matches not found.")
    else:
        agg_mode = st.radio("Aggregate by", ["posts (count)", "sum(score)", "sum(num_comments)"], index=0, horizontal=True)
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        g = raw.loc[rmask].copy()
        g["brand"] = g["keyword"]

        if agg_mode == "posts (count)":
            metric = g.groupby(["date","brand"]).size().rename("value")
        elif agg_mode == "sum(score)":
            metric = g.groupby(["date","brand"])["score"].sum().rename("value")
        else:
            metric = g.groupby(["date","brand"])["num_comments"].sum().rename("value")

        long_e = metric.reset_index()
        # Robust weekly resample without duplicate 'brand' during reset_index
        long_w = (long_e
                    .groupby(["brand", pd.Grouper(key="date", freq="W")], as_index=False)["value"].sum())
        fig_e = px.line(long_w[long_w["brand"].isin(selected)],
                        x="date", y="value", color="brand",
                        title=f"Engagement-weighted ({agg_mode}) — weekly")
        st.plotly_chart(fig_e, use_container_width=True)

        rank = (long_w.groupby("brand", as_index=False)["value"].sum()
                        .sort_values("value", ascending=False))
        st.dataframe(rank, use_container_width=True)

# -----------------------------
# 3) Subreddit mix & diffusion
# -----------------------------
with st.expander("Subreddit mix over time (stacked %)"):
    if raw is None:
        st.info("raw matches not found.")
    else:
        brand_pick = st.selectbox("Brand", options=selected if selected else all_brands, key="mix_brand")
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        r = raw.loc[rmask].copy()
        r = r[r["keyword"] == brand_pick][["date", "subreddit"]]
        if r.empty:
            st.info("No posts for this brand/date window.")
        else:
            r["count"] = 1
            weekly = (
                r.groupby(["subreddit", pd.Grouper(key="date", freq="W")], as_index=False)["count"].sum()
            )
            pivot = weekly.pivot(index="date", columns="subreddit", values="count").fillna(0)
            shares = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100
            long_mix = shares.reset_index().melt("date", var_name="subreddit", value_name="share_%")
            fig_mix = px.area(long_mix, x="date", y="share_%", color="subreddit",
                              title=f"{brand_pick} — subreddit mix (%)")
            st.plotly_chart(fig_mix, use_container_width=True)

with st.expander("Subreddit diffusion heatmap (within-brand % shares)"):
    if raw is None:
        st.info("raw matches not found.")
    else:
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        r = raw.loc[rmask].copy()
        r["brand"] = r["keyword"]
        if r.empty:
            st.info("No raw matches in this window.")
        else:
            tab = (r.groupby(["brand","subreddit"]).size().rename("posts").reset_index())
            tot = tab.groupby("brand")["posts"].sum().rename("tot")
            tab = tab.merge(tot, on="brand")
            tab["share_%"] = 100 * tab["posts"] / tab["tot"]
            H = tab.pivot(index="brand", columns="subreddit", values="share_%").fillna(0)
            # sort brands by concentration (HHI)
            shares = H.values / 100.0
            hhi = (shares ** 2).sum(axis=1)
            H = H.iloc[hhi.argsort()[::-1]]
            fig_hm = px.imshow(H, aspect="auto", color_continuous_scale="Viridis",
                               labels=dict(color="Share %"), title="Brand × Subreddit (% within brand)")
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("Sorted by concentration (HHI). Higher HHI ⇒ more niche; lower ⇒ more diffused.")

# -----------------------------
# 4) Sentiment overlay (optional)
# -----------------------------
with st.expander("Sentiment overlay (optional)"):
    if sent_cache is None:
        st.info("Sentiment cache not found. Precompute and save as 'sentiment_cache.csv' "
                "with columns like ['date','keyword','compound'] to enable this section.")
    else:
        # Sentiment cache already normalized to ['date','brand','compound']
        sc = sent_cache.copy()
        # Filter brand and date window
        sc = sc[sc['date'].dt.date.between(start_date, end_date)]
        if selected:
            sc = sc[sc['brand'].isin(selected)]

        if sc.empty:
            st.info("No sentiment rows for the selected window/brands.")
        else:
            # Weekly average sentiment per brand (tidy format)
            sc_w = (
                sc.groupby(['brand', pd.Grouper(key='date', freq='W')], as_index=False)['compound'].mean()
            )
            fig_sent = px.line(sc_w, x='date', y='compound', color='brand',
                               title='Weekly avg sentiment (compound)')
            st.plotly_chart(fig_sent, use_container_width=True)

            # Mentions vs Sentiment scatter (latest common week)
            w = resample_freq(df, 'W')
            long_w = to_long(w, selected).rename(columns={'mentions': 'count'})
            if not sc_w.empty and not long_w.empty:
                latest_week = min(sc_w['date'].max(), long_w['date'].max())
                sw = sc_w[sc_w['date'] == latest_week][['brand', 'compound']]
                mw = (long_w[long_w['date'] == latest_week][['brand', 'count']]
                            .rename(columns={'count': 'mentions'}))
                join = sw.merge(mw, on='brand', how='inner')
                if not join.empty:
                    fig_ms = px.scatter(join, x='mentions', y='compound', text='brand',
                                        title=f'Mentions vs Avg sentiment — week of {latest_week.date()}',
                                        labels={'mentions': 'Weekly mentions', 'compound': 'Avg sentiment'})
                    fig_ms.update_traces(textposition='top center')
                    st.plotly_chart(fig_ms, use_container_width=True)

# -----------------------------
# 6) Momentum vs Attention quadrant
# -----------------------------
with st.expander("Momentum vs Attention (weekly)"):
    w = resample_freq(df, "W")
    long_w = to_long(w, all_brands).rename(columns={"mentions": "count"})
    # last two full weeks
    per_brand_tail2 = (long_w.groupby("brand")
                             .apply(lambda g: g.tail(2))
                             .reset_index(level=0, drop=True))
    two = per_brand_tail2.pivot(index="brand", columns="date", values="count")
    if two.shape[1] >= 2:
        dates = sorted(two.columns.tolist())
        attn = two[dates[-1]].rename("attention").fillna(0)
        mom = (two[dates[-1]].fillna(0) - two[dates[-2]].fillna(0)).rename("momentum")
        quad = pd.concat([attn, mom], axis=1).reset_index()
        # filter brands in selection (optional)
        quad = quad[quad["brand"].isin(selected)] if selected else quad
        # thresholds
        x_thr = st.number_input("Attention threshold (x-axis line)", value=float(quad["attention"].median()))
        y_thr = st.number_input("Momentum threshold (y-axis line)", value=float(quad["momentum"].median()))
        fig_q = px.scatter(quad, x="attention", y="momentum", text="brand",
                           labels={"attention":"This week mentions", "momentum":"Δ vs last week"})
        fig_q.add_vline(x=x_thr, line_dash="dash"); fig_q.add_hline(y=y_thr, line_dash="dash")
        fig_q.update_traces(textposition="top center")
        st.plotly_chart(fig_q, use_container_width=True)
        st.dataframe(quad.sort_values(["momentum","attention"], ascending=False), use_container_width=True)
    else:
        st.info("Need at least 2 weeks of data to compute momentum.")

# -----------------------------
# Top subreddits + Top posts (reuse from your original)
# -----------------------------
with st.expander("Top subreddits for selected brands"):
    if raw is None:
        st.info(f"{RAW_MATCHES} not found.")
    else:
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        rsub = raw.loc[rmask].copy()
        if "keyword" in rsub.columns:
            rsub = rsub[rsub["keyword"].isin(selected)]
        if rsub.empty:
            st.info("No raw matches for the selected brands/date range.")
        else:
            top_subs = (rsub.groupby(["keyword", "subreddit"]).size().reset_index(name="posts"))
            for b in selected:
                sub_b = top_subs[top_subs["keyword"] == b].sort_values("posts", ascending=False).head(10)
                if not sub_b.empty:
                    st.markdown(f"**{b}**")
                    st.dataframe(sub_b[["subreddit", "posts"]], use_container_width=True)

with st.expander("Top posts (by score) in window"):
    if raw is None:
        st.info(f"{RAW_MATCHES} not found.")
    else:
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        rsub = raw.loc[rmask].copy()
        if "keyword" in rsub.columns and selected:
            rsub = rsub[rsub["keyword"].isin(selected)]
        cols = [c for c in ["date", "keyword", "alias", "subreddit", "title", "score", "num_comments"] if c in rsub.columns]
        top_posts = rsub.sort_values("score", ascending=False)[cols].head(50)
        st.dataframe(top_posts, use_container_width=True)