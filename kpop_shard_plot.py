"""
A utility module for data analysis and visualization of brand mentions on Reddit.

This module contains functions and utilities to preprocess, load, and analyze data
related to brand mentions. It includes functionality for computing Share of Voice,
loading datasets, transforming data formats, and generating visualizations using
Streamlit and Plotly.

It also provides functions for resampling data, applying rolling averages, and
handling file paths.

Functions:
- compute_sov: Compute Share of Voice from a DataFrame.
- find_file: Search for a file across predefined directories.
- load_pivot: Load and process the pivoted brand counts dataset.
- resample_freq: Resample the dataset at a given frequency.
- to_long: Convert a wide DataFrame to a long format for plotting.
- default_top_brands: Get the top N brands by mentions.
- apply_rolling: Apply a rolling average to smooth mention counts.
- load_raw_matches_or_none: Load raw match data if available.

The module initializes and renders a Streamlit-based user interface that
includes interactive controls for exploring temporal trends in brand mentions.

"""

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import streamlit as st
import plotly.express as px

# --- SoV helper ---
def compute_sov(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Share of Voice (percent) from long df with columns: date|brand|mentions."""
    totals = (
        long_df.groupby("date", as_index=False)["mentions"].sum()
                .rename(columns={"mentions": "total"})
    )
    x = long_df.merge(totals, on="date", how="left")
    x["value"] = (x["mentions"] / x["total"]).fillna(0) * 100.0
    return x[["date", "brand", "value"]]

# --- SoV over a window (ratio of sums) ---
def compute_window_sov(df_window: pd.DataFrame, brands_to_show: List[str]) -> pd.DataFrame:
    """
    df_window: wide frame with columns: date + brand columns (counts)
    Returns per-brand SoV over the window in percent:
        100 * sum(brand_mentions) / sum(total_mentions_all_brands)
    Only returns rows for brands_to_show.
    """
    if df_window.empty:
        return pd.DataFrame({"brand": [], "value": []})
    # Total across all brands per date
    totals_by_date = df_window.drop(columns=["date"]).sum(axis=1).rename("total")
    totals_df = pd.DataFrame({"date": df_window["date"], "total": totals_by_date})
    # Melt selected brands
    brands = [b for b in brands_to_show if b in df_window.columns]
    if not brands:
        return pd.DataFrame({"brand": [], "value": []})
    m = df_window[["date"] + brands].melt("date", var_name="brand", value_name="mentions")
    m = m.merge(totals_df, on="date", how="left")
    # Ratio of sums across the window
    sums = m.groupby("brand", as_index=False).agg({"mentions": "sum", "total": "sum"})
    sums["value"] = (sums["mentions"] / sums["total"]).fillna(0) * 100.0
    return sums[["brand", "value"]]


DATA_CANDIDATES = [
    Path(__file__).parent.parent / "data"
]
PIVOT_FILE = "kpop_alias_pivoted_brand_counts.csv"
RAW_MATCHES = "reddit_matches_raw.csv"

DEFAULT_TOPN = 5


def find_file(fname:str) -> Path:
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
        if c!= 'date':
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def resample_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return df.set_index("date").asfreq("D").fillna(0).reset_index()
    return (
        df.set_index('date')
        .resample("W")
        .sum()
        .reset_index()
    )


def to_long(df: pd.DataFrame, brands: List[str]) -> pd.DataFrame:
    cols = ['date'] + brands
    sub = df[cols].copy()
    long_df = sub.melt(id_vars='date', var_name='brand', value_name='mentions')
    return long_df


def default_top_brands(df_freq: pd.DataFrame, topn: int) -> List[str]:
    totals = df_freq.drop(columns=['date']).sum(axis=0).sort_values(ascending=False)
    return totals.head(topn).index.tolist()


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


st.set_page_config(page_title="Reddit Brand Mentions (Sharded)", layout="wide")
st.title("Reddit Brand Mentions - Sharded (Interactive)")

try:
    df = load_pivot()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

all_brands = [c for c in df.columns if c != "date"]

with st.sidebar:
    st.header("Controls")
    freq_label = st.radio("Frequency", ["Daily", "Weekly"], index=0)
    freq = "D" if freq_label == "Daily" else "W"

    # We compute frequency FIRST, then pick defaults based on that windowed frame
    df_freq = resample_freq(df, freq)

    norm_mode = st.radio(
        "Normalization",
        ["Raw counts", "Share of Voice (SoV)"],
        index=0
    )

    # Date range selector (based on available dates at this freq)
    min_d, max_d = df_freq["date"].min().date(), df_freq["date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d, max_value=max_d
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, date_range

    # Filter by date first
    mask = (df_freq["date"].dt.date >= start_date) & (df_freq["date"].dt.date <= end_date)
    df_win = df_freq.loc[mask].copy()

    # Defaults: top-N brands by total mentions over filtered window
    defaults = default_top_brands(df_win, DEFAULT_TOPN) if not df_win.empty else []
    selected = st.multiselect("Brands", options=all_brands, default=defaults)

    # Rolling average toggle
    do_smooth = st.checkbox("Show rolling average", value=True)
    win_daily = st.number_input("Rolling window (days)", min_value=2, max_value=30, value=7) if (do_smooth and freq == "D") else 7
    win_weekly = st.number_input("Rolling window (weeks)", min_value=2, max_value=12, value=4) if (do_smooth and freq == "W") else 4

# Guard: no data window or no brands
if df_win.empty:
    st.info("No data in the selected date range.")
    st.stop()

if not selected:
    st.info("Pick at least one brand to plot from the sidebar.")
    st.stop()

# Long format for plotting
long_df = to_long(df_win, selected)

# Build plotting frame according to normalization
if norm_mode == "Raw counts":
    long_metric = long_df.rename(columns={"mentions": "value"}).copy()
    y_label = "Mentions"
else:
    # SoV should be computed over ALL brands in the filtered window, then subset to selected
    all_brands_in_win = [c for c in df_win.columns if c != "date"]
    long_all = to_long(df_win, all_brands_in_win)
    sov_all = compute_sov(long_all)
    long_metric = sov_all[sov_all["brand"].isin(selected)].copy()
    y_label = "Share of Voice (%)"

# Plot
if do_smooth:
    long_plot = long_metric.rename(columns={"value": "mentions"})  # reuse smoother on 'mentions'
    long_smooth = apply_rolling(long_plot, freq=freq, win_daily=win_daily, win_weekly=win_weekly)
    fig = px.line(
        long_smooth,
        x="date", y="smoothed", color="brand",
        title=f"Brand mentions ({freq_label}) — {norm_mode}",
        labels={"smoothed": y_label, "date": "Date"},
    )
    # faint raw lines underneath (hide in legend)
    fig_raw = px.line(long_metric, x="date", y="value", color="brand")
    for tr in fig_raw.data:
        tr.update(opacity=0.25, line={"width": 1}, showlegend=False)
        fig.add_trace(tr)
else:
    fig = px.line(
        long_metric,
        x="date", y="value", color="brand",
        title=f"Brand mentions ({freq_label}) — {norm_mode}",
        labels={"value": y_label, "date": "Date"},
    )

fig.update_layout(hovermode="x unified", legend_title_text="Brand")
st.plotly_chart(fig, use_container_width=True)

# Summary table
st.subheader("Summary (current window)")
if norm_mode == "Raw counts":
    summary = (
        long_metric.groupby("brand", as_index=False)["value"].sum()
                  .sort_values("value", ascending=False)
                  .rename(columns={"value": "Mentions"})
    )
else:
    # True SoV over the window: ratio of sums across all brands
    sov_window = compute_window_sov(df_win, selected)
    summary = sov_window.sort_values("value", ascending=False).rename(columns={"value": "SoV over window (%)"})
    summary["SoV over window (%)"] = summary["SoV over window (%)"].round(2)

st.dataframe(summary, use_container_width=True)

# Top 10 brands (entire dataset)
st.subheader("Top 10 brands — Entire dataset")
if norm_mode == "Raw counts":
    totals_all = df.drop(columns=["date"]).sum(axis=0).sort_values(ascending=False)
    top10_all = totals_all.head(10).reset_index()
    top10_all.columns = ["brand", "value"]
    y_top = "Total mentions"
else:
    # SoV over the full dataset: ratio of sums across all brands
    full_counts = df.copy()
    totals_by_date_full = full_counts.drop(columns=["date"]).sum(axis=1).rename("total")
    totals_df_full = pd.DataFrame({"date": full_counts["date"], "total": totals_by_date_full})
    long_full = full_counts.melt("date", var_name="brand", value_name="mentions")
    long_full = long_full.merge(totals_df_full, on="date", how="left")
    sums_full = long_full.groupby("brand", as_index=False).agg({"mentions": "sum", "total": "sum"})
    sums_full["value"] = (sums_full["mentions"] / sums_full["total"]).fillna(0) * 100.0
    top10_all = sums_full.sort_values("value", ascending=False).head(10)
    y_top = "SoV over dataset (%)"

fig_top10 = px.bar(
    top10_all, x="brand", y="value",
    title=f"Top 10 brands — {y_top}", labels={"value": y_top, "brand": "Brand"}
)
if norm_mode != "Raw counts":
    top10_all["value"] = top10_all["value"].round(2)
st.plotly_chart(fig_top10, use_container_width=True)
st.dataframe(top10_all, use_container_width=True)

# Optional: drilldown from raw matches
raw = load_raw_matches_or_none()

with st.expander("Top subreddits for selected brands"):
    if raw is None:
        st.info(f"{RAW_MATCHES} not found.")
    else:
        # Filter raw matches by the visible date window
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        rsub = raw.loc[rmask].copy()
        if "keyword" in rsub.columns:
            rsub = rsub[rsub["keyword"].isin(selected)]
        if rsub.empty:
            st.info("No raw matches for the selected brands/date range.")
        else:
            # Group by brand (keyword) and subreddit
            top_subs = (
                rsub.groupby(["keyword", "subreddit"])\
                    .size()\
                    .reset_index(name="posts")
            )
            # Show top 10 subreddits per selected brand
            for b in selected:
                sub_b = (
                    top_subs[top_subs["keyword"] == b]
                    .sort_values("posts", ascending=False)
                    .head(10)
                )
                if not sub_b.empty:
                    st.markdown(f"**{b}**")
                    st.dataframe(sub_b[["subreddit", "posts"]], use_container_width=True)

with st.expander("Show top posts (if raw matches available)"):
    if raw is None:
        st.info(f"{RAW_MATCHES} not found.")
    else:
        # Filter by date window and selected brands
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        rsub = raw.loc[rmask].copy()
        if "keyword" in rsub.columns:
            rsub = rsub[rsub["keyword"].isin(selected)]
        # Top posts by score for the current window
        cols = [c for c in ["date", "keyword", "alias", "subreddit", "title", "score", "num_comments"] if c in rsub.columns]
        top_posts = rsub.sort_values("score", ascending=False)[cols].head(50)
        st.dataframe(top_posts, use_container_width=True)
