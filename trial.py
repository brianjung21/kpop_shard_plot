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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Config & file discovery
# -----------------------------
DATA_CANDIDATES = [
    Path(__file__).parent.parent / "data"
]
PIVOT_FILE = "kpop_alias_pivoted_brand_counts_combined.csv"
RAW_MATCHES = "reddit_matches_raw_combined.csv"
SENTIMENT_CACHE = "sentiment_cache.csv"  # optional (see notes below)

# ETF CSV file (for overlay)
ETF_FILE = "tiger_media_contents_prices.csv"  # located in the same data folder
@st.cache_data(show_spinner=False)
def load_etf_or_none() -> Optional[pd.DataFrame]:
    try:
        p = find_file(ETF_FILE)
    except FileNotFoundError:
        return None
    try:
        etf = pd.read_csv(p)
    except Exception:
        return None
    cols = {c.lower(): c for c in etf.columns}
    if "date" not in cols:
        return None
    etf = etf.rename(columns={cols["date"]: "date"})
    if "close" in cols:
        etf = etf.rename(columns={cols["close"]: "close"})
    if "volume" in cols:
        etf = etf.rename(columns={cols["volume"]: "volume"})
    etf["date"] = pd.to_datetime(etf["date"], errors="coerce")
    for c in ("close","volume"):
        if c in etf.columns:
            etf[c] = pd.to_numeric(etf[c], errors="coerce")
    return etf.dropna(subset=["date"]).reset_index(drop=True)

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
st.set_page_config(page_title="Beauty Reddit Intelligence (Sharded)", layout="wide")
st.title("Beauty Reddit Intelligence — Sharded (Interactive)")

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
    lang_choice = st.radio("Language / 언어", ["English", "한국어"], index=0)
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

    # ETF overlay toggle (plaster the line regardless of scale)
    etf_overlay_opt = st.selectbox(
        "ETF overlay",
        ["None", "Close price", "Volume"],
        index=0,
        help="Overlay TIGER Media Contents ETF (from data folder) on the first chart."
    )
    etf_df = load_etf_or_none()

    defaults = default_top_brands(df_win, DEFAULT_TOPN)
    selected = st.multiselect("Brands", options=all_brands, default=defaults)

    do_smooth = st.checkbox("Show rolling average", value=True)
    win_daily = st.number_input("Rolling window (days)", min_value=2, max_value=30, value=7) if (do_smooth and freq == "D") else 7
    win_weekly = st.number_input("Rolling window (weeks)", min_value=2, max_value=12, value=4) if (do_smooth and freq == "W") else 4

# -----------------------------
# Language helper
# -----------------------------
def explain(text_en: str, text_ko: str) -> str:
    return text_en if lang_choice == "English" else text_ko

# Expander details helper
def expander_details(title_en: str, title_ko: str, body_en: str, body_ko: str):
    with st.expander(explain(title_en, title_ko)):
        st.markdown(explain(body_en, body_ko))

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
        title=f"Beauty brand trends ({freq_label}) — {norm_mode}",
        labels={"smoothed": y_label, "date": "Date"}
    )
    fig_raw = px.line(long_metric, x="date", y="value", color="brand")
    for tr in fig_raw.data:
        tr.update(opacity=0.25, line={"width": 1}, showlegend=False)
        fig_main.add_trace(tr)
else:
    fig_main = px.line(
        long_metric, x="date", y="value", color="brand",
        title=f"Beauty brand trends ({freq_label}) — {norm_mode}",
        labels={"value": y_label, "date": "Date"}
    )

fig_main.update_layout(hovermode="x unified", legend_title_text="Brand")

# --- ETF overlay handling (secondary y) ---
def _overlay_etf(fig_px: go.Figure, freq: str) -> go.Figure:
    if etf_overlay_opt == "None" or etf_df is None or etf_df.empty:
        return fig_px
    # Build a new figure with secondary axis and copy existing traces
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for tr in fig_px.data:
        fig.add_trace(tr, secondary_y=False)
    # Determine window based on what is plotted
    try:
        xmins = [pd.to_datetime(tr.x).min() for tr in fig_px.data if hasattr(tr, "x") and len(tr.x)]
        xmaxs = [pd.to_datetime(tr.x).max() for tr in fig_px.data if hasattr(tr, "x") and len(tr.x)]
        xmin = min(xmins) if xmins else None
        xmax = max(xmaxs) if xmaxs else None
    except Exception:
        xmin = df_win["date"].min()
        xmax = df_win["date"].max()

    e = etf_df.copy()
    if freq == "W":
        # resample to weekly: last close, sum volume
        e = (e.set_index("date")
               .resample("W")
               .agg({"close": "last", "volume": "sum"})
               .reset_index())
    # slice to window if available
    if xmin is not None and xmax is not None:
        e = e[(e["date"] >= pd.to_datetime(xmin)) & (e["date"] <= pd.to_datetime(xmax))]
    # add chosen series on secondary y
    if etf_overlay_opt == "Close price" and "close" in e.columns and not e["close"].dropna().empty:
        fig.add_trace(
            go.Scatter(x=e["date"], y=e["close"], mode="lines", name="ETF Close", line=dict(dash="dash")),
            secondary_y=True,
        )
        fig.update_yaxes(title_text=y_label, secondary_y=False)
        fig.update_yaxes(title_text="ETF Close", secondary_y=True)
    elif etf_overlay_opt == "Volume" and "volume" in e.columns and not e["volume"].dropna().empty:
        fig.add_trace(
            go.Scatter(x=e["date"], y=e["volume"], mode="lines", name="ETF Volume", line=dict(dash="dot")),
            secondary_y=True,
        )
        fig.update_yaxes(title_text=y_label, secondary_y=False)
        fig.update_yaxes(title_text="ETF Volume", secondary_y=True)
    # keep layout from original (handle Layout object safely)
    try:
        layout_dict = fig_px.layout.to_plotly_json()
        layout_dict = {k: v for k, v in layout_dict.items() if k not in ("_empty",)}
        fig.update_layout(**layout_dict)
    except Exception:
        fig.update_layout(fig_px.layout)
    return fig

fig_show = _overlay_etf(fig_main, freq)
st.plotly_chart(fig_show, use_container_width=True)

# Add expander after core plot
expander_details(
    "What this shows / How it’s built / Why it matters / Implications / Actions",
    "무엇 / 계산 방식 / 의미 / 시사점 / 활용 방안",
    (
        "**1) What is this?**\n"
        f"Trends in brand mentions by {freq_label.lower()}, shown as either raw counts or Share of Voice (SoV). If smoothing is enabled, the darker line is a rolling average and the faint line is raw.\n\n"
        "**2) How is it calculated?**\n"
        f"Raw: pivoted counts per brand per date. SoV: 100 × brand_mentions / total_mentions on each date. Optional rolling mean: {win_daily if freq=='D' else win_weekly} periods.\n\n"
        "**3) What does it mean?**\n"
        "Reveals directional movement in discussion volume.\n\n"
        "**4) Implications**\n"
        "Uptrends can indicate rising buzz; downtrends, cooling interest.\n\n"
        "**5) Actions for analysts/investors**\n"
        "Watch breakouts, cross-check with spikes and engagement-weighted charts to confirm quality."
    ),
    (
        "**1) 이것은 무엇인가요?**\n"
        f"브랜드 언급 트렌드({ '일간' if freq=='D' else '주간' })를 원자료 또는 SoV로 표시합니다. 스무딩을 켜면 진한 선은 이동평균, 흐린 선은 원시값입니다.\n\n"
        "**2) 어떻게 계산되나요?**\n"
        f"Raw: 날짜별 브랜드 언급수. SoV: 각 날짜의 100×(브랜드 언급/전체 언급). 이동평균: {win_daily if freq=='D' else win_weekly} 기간.\n\n"
        "**3) 무엇을 의미하나요?**\n"
        "담론 볼륨의 방향성을 보여줍니다.\n\n"
        "**4) 시사점**\n"
        "상승은 관심 확산, 하락은 식음을 시사할 수 있습니다.\n\n"
        "**5) 활용 방안**\n"
        "급등 구간을 주시하고, 스파이크/참여가중 차트로 질적 확인을 병행하세요."
    )
)

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

# Add expander after summary table
expander_details(
    "Summary — details",
    "요약 — 설명",
    (
        "**1) What is this?**\n"
        "Aggregate over the selected window: totals (Raw) or window SoV per brand.\n\n"
        "**2) How is it calculated?**\n"
        "Raw: sum of daily mentions in the window. SoV: 100 × sum(brand_mentions)/sum(total_mentions) over the window.\n\n"
        "**3) Meaning**\n"
        "Shows who dominated the window.\n\n"
        "**4) Implications**\n"
        "Use with caution if scraping caps apply; cross-check engagement metrics.\n\n"
        "**5) Actions**\n"
        "Prioritize deep dives on brands with high window share and positive momentum."
    ),
    (
        "**1) 이것은 무엇인가요?**\n"
        "선택한 기간에서의 합계(Raw) 또는 기간 SoV입니다.\n\n"
        "**2) 어떻게 계산되나요?**\n"
        "Raw: 기간 내 일별 언급 합. SoV: 기간 동안 100×(브랜드 언급 합/전체 언급 합).\n\n"
        "**3) 의미**\n"
        "해당 기간을 주도한 브랜드를 보여줍니다.\n\n"
        "**4) 시사점**\n"
        "수집 한도가 있는 경우 왜곡 가능. 참여 지표로 교차확인하세요.\n\n"
        "**5) 활용 방안**\n"
        "기간 점유율이 높고 모멘텀이 좋은 브랜드를 우선 분석하세요."
    )
)

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

    # Expander after spike table
    expander_details(
        "Spike detector — details",
        "스파이크 탐지 — 설명",
        (
            "**1) What is this?**\n"
            "Weekly z-score based spike detection for selected brands.\n\n"
            "**2) How is it calculated?**\n"
            "For each brand, compute rolling mean/SD over k weeks, then z = (week-mean)/SD. Keep weeks with z ≥ threshold and mentions ≥ minimum.\n\n"
            "**3) Meaning**\n"
            "Identifies statistically unusual surges.\n\n"
            "**4) Implications**\n"
            "Investigate drivers (posts, subreddits); corroborate with sentiment/engagement.\n\n"
            "**5) Actions**\n"
            "Tag catalysts, monitor follow-through next week, consider timing promotions."
        ),
        (
            "**1) 이것은 무엇인가요?**\n"
            "브랜드별 주간 z-스코어 기반의 급등 탐지입니다.\n\n"
            "**2) 어떻게 계산되나요?**\n"
            "브랜드별로 k주 이동 평균/표준편차를 계산하고 z = (이번주-평균)/표준편차. 임계치 이상이며 최소 언급수 조건 충족 시 스파이크로 표시합니다.\n\n"
            "**3) 의미**\n"
            "통계적으로 이례적인 급증을 식별합니다.\n\n"
            "**4) 시사점**\n"
            "원인(포스트, 서브레딧)을 조사하고 감성/참여와 교차 확인합니다.\n\n"
            "**5) 활용 방안**\n"
            "촉발 요인을 태깅하고 다음 주 추이를 모니터링, 프로모션 타이밍을 검토합니다."
        )
    )

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
        # Expander after spike driver table
        expander_details(
            "Spike drivers — details",
            "스파이크 기여 포스트 — 설명",
            (
                "**1) What is this?**\nTop posts in the spike week for the chosen brand.\n\n**2) How**\nFiltered raw matches for the spike week; ranked by score / comments.\n\n**3) Meaning**\nThese are likely catalysts.\n\n**4) Implications**\nTie content themes to performance.\n\n**5) Actions**\nReplicate effective formats; outreach to high-impact subreddits/authors.",
                "**1) 이것은 무엇인가요?**\n선택 브랜드의 스파이크 주간 상위 포스트입니다.\n\n**2) 방법**\n스파이크 주간의 raw 매치를 필터링하여 점수/댓글 기준으로 정렬.\n\n**3) 의미**\n급등을 유발한 촉발 요인일 가능성이 큽니다.\n\n**4) 시사점**\n콘텐츠 주제를 성과와 연결하세요.\n\n**5) 활용 방안**\n효과적인 포맷을 재활용하고 영향력 큰 서브레딧/작성자에 협업을 제안하세요."
            )[0],
            (
                "**1) 이것은 무엇인가요?**\n선택 브랜드의 스파이크 주간 상위 포스트입니다.\n\n**2) 방법**\n스파이크 주간의 raw 매치를 필터링하여 점수/댓글 기준으로 정렬.\n\n**3) 의미**\n급등을 유발한 촉발 요인일 가능성이 큽니다.\n\n**4) 시사점**\n콘텐츠 주제를 성과와 연결하세요.\n\n**5) 활용 방안**\n효과적인 포맷을 재활용하고 영향력 큰 서브레딧/작성자에 협업을 제안하세요."
            )[0]
        )
    else:
        st.caption("raw matches not available; cannot show spike drivers.")

# -----------------------------
# 2) Engagement-weighted mentions
# -----------------------------
with st.expander(explain("Engagement-weighted mentions", "참여도 가중 언급량")):
    if raw is None:
        st.info(explain("raw matches not found.", "raw 매치 데이터를 찾을 수 없습니다."))
    else:
        agg_mode = st.radio(
            explain("Aggregate by", "집계 기준"),
            ["posts (count)", "sum(score)", "sum(num_comments)"],
            index=0, horizontal=True
        )
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
                        title=explain(f"Engagement-weighted ({agg_mode}) — weekly", f"참여도 가중 ({agg_mode}) — 주간"))
        st.plotly_chart(fig_e, use_container_width=True)

        # Expander after engagement-weighted plot
        expander_details(
            "Engagement-weighted mentions — details",
            "참여도 가중 언급량 — 설명",
            (
                "**1) What**\nCounts or sums of Reddit post engagement (posts, score, comments) by brand.\n\n**2) How**\nAggregate per brand per week; choose metric with the radio.\n\n**3) Meaning**\nFocuses on attention quality, not just volume.\n\n**4) Implications**\nHigh engagement with low volume can still be commercially meaningful.\n\n**5) Actions**\nDouble down where engagement efficiency is high; analyze subreddit mix.",
                "**1) 무엇**\n브랜드별 레딧 포스트 참여(개수, 점수, 댓글) 집계입니다.\n\n**2) 방법**\n주간 단위로 브랜드별 합산(라디오에서 지표 선택).\n\n**3) 의미**\n단순 볼륨이 아니라 **질**에 초점을 둡니다.\n\n**4) 시사점**\n볼륨이 적어도 참여 효율이 높으면 상업적 의미가 있을 수 있습니다.\n\n**5) 활용 방안**\n효율이 높은 브랜드에 집중하고, 서브레딧 구성을 분석하세요."
            )[0],
            (
                "**1) 무엇**\n브랜드별 레딧 포스트 참여(개수, 점수, 댓글) 집계입니다.\n\n**2) 방법**\n주간 단위로 브랜드별 합산(라디오에서 지표 선택).\n\n**3) 의미**\n단순 볼륨이 아니라 **질**에 초점을 둡니다.\n\n**4) 시사점**\n볼륨이 적어도 참여 효율이 높으면 상업적 의미가 있을 수 있습니다.\n\n**5) 활용 방안**\n효율이 높은 브랜드에 집중하고, 서브레딧 구성을 분석하세요."
            )[0]
        )

        rank = (long_w.groupby("brand", as_index=False)["value"].sum()
                        .sort_values("value", ascending=False))
        st.dataframe(rank, use_container_width=True)
        # Expander after ranking table
        expander_details(
            "Engagement-weighted brand ranking — details",
            "참여도 가중 브랜드 랭킹 — 설명",
            (
                "**1) What is this?**\n"
                "Ranks brands by total engagement-weighted metric in the window.\n\n"
                "**2) How**\n"
                "Sum across all weeks in window for selected engagement metric.\n\n"
                "**3) Meaning**\n"
                "Shows which brands drew the most quality attention.\n\n"
                "**4) Implications**\n"
                "High rank with low mention count = efficient engagement.\n\n"
                "**5) Actions**\n"
                "Study high-ranking brands' content and subreddit mix."
            ),
            (
                "**1) 이것은 무엇인가요?**\n"
                "기간 내 참여도 가중치 합계로 브랜드를 랭킹합니다.\n\n"
                "**2) 방법**\n"
                "기간 내 모든 주의 선택 지표를 합산합니다.\n\n"
                "**3) 의미**\n"
                "가장 높은 질적 주목을 받은 브랜드를 보여줍니다.\n\n"
                "**4) 시사점**\n"
                "언급이 적어도 랭킹이 높으면 효율적인 참여임을 의미합니다.\n\n"
                "**5) 활용 방안**\n"
                "상위 브랜드의 콘텐츠와 서브레딧 구성을 분석하세요."
            )
        )

# -----------------------------
# 3) Subreddit mix & diffusion
# -----------------------------
with st.expander(explain("Subreddit mix over time (stacked %)", "서브레딧 구성 변화(스택 %)")):
    if raw is None:
        st.info(explain("raw matches not found.", "raw 매치 데이터를 찾을 수 없습니다."))
    else:
        brand_pick = st.selectbox(
            explain("Brand", "브랜드"),
            options=selected if selected else all_brands, key="mix_brand"
        )
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        r = raw.loc[rmask].copy()
        r = r[r["keyword"] == brand_pick][["date", "subreddit"]]
        if r.empty:
            st.info(explain("No posts for this brand/date window.", "해당 브랜드/기간에 대한 포스트가 없습니다."))
        else:
            r["count"] = 1
            weekly = (
                r.groupby(["subreddit", pd.Grouper(key="date", freq="W")], as_index=False)["count"].sum()
            )
            pivot = weekly.pivot(index="date", columns="subreddit", values="count").fillna(0)
            shares = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100
            long_mix = shares.reset_index().melt("date", var_name="subreddit", value_name="share_%")
            fig_mix = px.area(long_mix, x="date", y="share_%", color="subreddit",
                              title=explain(f"{brand_pick} — subreddit mix (%)", f"{brand_pick} — 서브레딧 구성 (%)"))
            st.plotly_chart(fig_mix, use_container_width=True)
            # Expander after subreddit mix plot
            expander_details(
                "Subreddit mix — details",
                "서브레딧 구성 — 설명",
                (
                    "**1) What is this?**\n"
                    "Shows the brand's subreddit composition over time (stacked % by week).\n\n"
                    "**2) How**\n"
                    "Counts posts per subreddit per week, normalizes to 100% per week.\n\n"
                    "**3) Meaning**\n"
                    "Tracks diffusion (many subreddits) vs concentration (few dominate).\n\n"
                    "**4) Implications**\n"
                    "Sudden shifts may signal new audience or viral spread.\n\n"
                    "**5) Actions**\n"
                    "Identify rising subreddits for targeted outreach."
                ),
                (
                    "**1) 이것은 무엇인가요?**\n"
                    "브랜드별 주간 서브레딧 구성 비율(스택 %)의 변화를 보여줍니다.\n\n"
                    "**2) 방법**\n"
                    "주간 단위로 서브레딧별 포스트 수를 집계하여 100%로 정규화합니다.\n\n"
                    "**3) 의미**\n"
                    "확산(다양한 서브레딧)과 집중(소수 서브레딧 집중)을 파악할 수 있습니다.\n\n"
                    "**4) 시사점**\n"
                    "급격한 변화는 새로운 유입 또는 바이럴 확산을 의미할 수 있습니다.\n\n"
                    "**5) 활용 방안**\n"
                    "부상하는 서브레딧을 찾아 타겟 마케팅에 활용하세요."
                )
            )

with st.expander(explain("Subreddit diffusion heatmap (within-brand % shares)", "서브레딧 확산 히트맵 (브랜드 내 %)")):
    if raw is None:
        st.info(explain("raw matches not found.", "raw 매치 데이터를 찾을 수 없습니다."))
    else:
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        r = raw.loc[rmask].copy()
        r["brand"] = r["keyword"]
        if r.empty:
            st.info(explain("No raw matches in this window.", "이 기간에 해당하는 raw 매치가 없습니다."))
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
                               labels=dict(color="Share %"),
                               title=explain("Brand × Subreddit (% within brand)", "브랜드 × 서브레딧 (브랜드 내 %)"))
            st.plotly_chart(fig_hm, use_container_width=True)
            # Expander after diffusion heatmap
            expander_details(
                "Subreddit diffusion heatmap — details",
                "서브레딧 확산 히트맵 — 설명",
                (
                    "**1) What is this?**\n"
                    "Heatmap of subreddit % shares within each brand (sorted by HHI concentration).\n\n"
                    "**2) How**\n"
                    "Rows: brands; columns: subreddits; cell = % of brand's posts in that subreddit. Sorted by concentration (HHI).\n\n"
                    "**3) Meaning**\n"
                    "High HHI = niche (few subreddits dominate); low HHI = broad diffusion.\n\n"
                    "**4) Implications**\n"
                    "Broad brands reach diverse communities; niche brands may have loyal but narrow bases.\n\n"
                    "**5) Actions**\n"
                    "For niche brands, test cross-posting; for broad, segment messaging."
                ),
                (
                    "**1) 이것은 무엇인가요?**\n"
                    "브랜드별 서브레딧 내 점유율(%)을 히트맵으로 시각화합니다 (집중도 HHI 순).\n\n"
                    "**2) 방법**\n"
                    "행: 브랜드, 열: 서브레딧, 셀: 브랜드 내 해당 서브레딧 점유율(%). HHI(집중도) 순으로 정렬.\n\n"
                    "**3) 의미**\n"
                    "높은 HHI는 틈새(소수 서브레딧 집중), 낮은 HHI는 광범위 확산을 의미합니다.\n\n"
                    "**4) 시사점**\n"
                    "확산 브랜드는 다양한 커뮤니티에 도달, 틈새 브랜드는 충성도는 높으나 폭이 좁을 수 있습니다.\n\n"
                    "**5) 활용 방안**\n"
                    "틈새 브랜드는 크로스포스팅을, 확산 브랜드는 메시지 세분화를 시도하세요."
                )
            )

# -----------------------------
# 4) Sentiment overlay (optional)
# -----------------------------
with st.expander(explain("Sentiment overlay (optional)", "감성 분석 오버레이 (선택 사항)")):
    if sent_cache is None:
        st.info(explain(
            "Sentiment cache not found. Precompute and save as 'sentiment_cache.csv' with columns like ['date','keyword','compound'] to enable this section.",
            "감성 분석 캐시를 찾을 수 없습니다. ['date','keyword','compound'] 컬럼의 'sentiment_cache.csv'를 미리 계산해 저장하면 이 섹션이 활성화됩니다."
        ))
    else:
        # Sentiment cache already normalized to ['date','brand','compound']
        sc = sent_cache.copy()
        # Filter brand and date window
        sc = sc[sc['date'].dt.date.between(start_date, end_date)]
        if selected:
            sc = sc[sc['brand'].isin(selected)]

        if sc.empty:
            st.info(explain("No sentiment rows for the selected window/brands.", "선택한 기간/브랜드에 해당하는 감성 데이터가 없습니다."))
        else:
            # Weekly average sentiment per brand (tidy format)
            sc_w = (
                sc.groupby(['brand', pd.Grouper(key='date', freq='W')], as_index=False)['compound'].mean()
            )
            fig_sent = px.line(
                sc_w, x='date', y='compound', color='brand',
                title=explain('Weekly avg sentiment (compound)', '주간 평균 감성 점수 (compound)')
            )
            st.plotly_chart(fig_sent, use_container_width=True)
            # Expander after sentiment overlay plot
            expander_details(
                "Sentiment overlay — details",
                "감성 오버레이 — 설명",
                (
                    "**1) What is this?**\n"
                    "Shows average sentiment score per brand per week (compound, [-1,1]).\n\n"
                    "**2) How**\n"
                    "Aggregates sentiment of posts/comments per week by brand.\n\n"
                    "**3) Meaning**\n"
                    "Tracks positive/negative tone trends alongside mentions.\n\n"
                    "**4) Implications**\n"
                    "Sudden drops may indicate controversy or negative events.\n\n"
                    "**5) Actions**\n"
                    "Investigate dips, cross-check with spikes and top posts. Sentiment is noisy: interpret with caution."
                ),
                (
                    "**1) 이것은 무엇인가요?**\n"
                    "브랜드별 주간 평균 감성 점수(compound, -1~1)를 보여줍니다.\n\n"
                    "**2) 방법**\n"
                    "포스트/댓글의 감성 점수를 브랜드별·주간으로 집계합니다.\n\n"
                    "**3) 의미**\n"
                    "언급량과 함께 긍·부정 분위기 변화를 추적할 수 있습니다.\n\n"
                    "**4) 시사점**\n"
                    "급락은 논란·이슈 가능성을 시사합니다.\n\n"
                    "**5) 활용 방안**\n"
                    "하락 구간의 원인을 파악하고, 스파이크/상위 포스트와 교차 확인하세요. 감성은 노이즈가 크므로 해석에 주의하세요."
                )
            )

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
                    fig_ms = px.scatter(
                        join, x='mentions', y='compound', text='brand',
                        title=explain(
                            f'Mentions vs Avg sentiment — week of {latest_week.date()}',
                            f'언급량 vs 평균 감성 — {latest_week.date()} 주간'
                        ),
                        labels={'mentions': explain('Weekly mentions', '주간 언급수'),
                                'compound': explain('Avg sentiment', '평균 감성')}
                    )
                    fig_ms.update_traces(textposition='top center')
                    st.plotly_chart(fig_ms, use_container_width=True)
                    # Expander after mentions-vs-sentiment scatter
                    expander_details(
                        "Mentions vs Sentiment — details",
                        "언급량 vs 감성 — 설명",
                        (
                            "**1) What is this?**\n"
                            "Scatterplot of each brand's mentions vs average sentiment for the latest week.\n\n"
                            "**2) How**\n"
                            "X: total mentions; Y: average compound sentiment.\n\n"
                            "**3) Meaning**\n"
                            "Connects attention to tone: brands with high mentions & positive sentiment may have viral positive buzz; high mentions & negative sentiment may indicate controversy.\n\n"
                            "**4) Implications**\n"
                            "Helps distinguish between positive and negative surges.\n\n"
                            "**5) Actions**\n"
                            "Prioritize further analysis for outliers and big-move brands."
                        ),
                        (
                            "**1) 이것은 무엇인가요?**\n"
                            "각 브랜드의 주간 언급수와 평균 감성 점수를 산점도로 나타냅니다.\n\n"
                            "**2) 방법**\n"
                            "X: 언급량, Y: 평균 감성 점수.\n\n"
                            "**3) 의미**\n"
                            "주목도와 분위기 연결: 언급이 많고 감성이 높으면 긍정적 바이럴, 언급은 많지만 감성이 낮으면 논란 가능성.\n\n"
                            "**4) 시사점**\n"
                            "긍·부정 급등을 구분하는 데 도움.\n\n"
                            "**5) 활용 방안**\n"
                            "이상치와 급변 브랜드를 추가 분석하세요."
                        )
                    )

# -----------------------------
# 6) Momentum vs Attention quadrant
# -----------------------------
with st.expander(explain("Momentum vs Attention (weekly)", "모멘텀 vs 주목도 (주간)")):
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
        x_thr = st.number_input(
            explain("Attention threshold (x-axis line)", "주목도 기준선 (x축)"),
            value=float(quad["attention"].median()))
        y_thr = st.number_input(
            explain("Momentum threshold (y-axis line)", "모멘텀 기준선 (y축)"),
            value=float(quad["momentum"].median()))
        fig_q = px.scatter(
            quad, x="attention", y="momentum", text="brand",
            labels={
                "attention": explain("This week mentions", "이번 주 언급수"),
                "momentum": explain("Δ vs last week", "전주 대비 변화량")
            }
        )
        fig_q.add_vline(x=x_thr, line_dash="dash"); fig_q.add_hline(y=y_thr, line_dash="dash")
        fig_q.update_traces(textposition="top center")
        st.plotly_chart(fig_q, use_container_width=True)
        # Expander after quadrant plot and table
        st.dataframe(quad.sort_values(["momentum","attention"], ascending=False), use_container_width=True)
        expander_details(
            "Momentum vs Attention quadrants — details",
            "모멘텀 vs 주목도 사분면 — 설명",
            (
                "**1) What is this?**\n"
                "Quadrant chart mapping brands by current attention (x) and momentum (y) vs last week.\n\n"
                "**2) How**\n"
                "X: this week's mentions; Y: change vs last week. Thresholds = medians.\n\n"
                "**3) Meaning**\n"
                "Top-right = rising stars; bottom-right = big but cooling; top-left = small but surging; bottom-left = laggards.\n\n"
                "**4) Implications**\n"
                "Identifies breakout and fading brands at a glance.\n\n"
                "**5) Actions**\n"
                "Focus on brands in the top-right quadrant for further analysis or investment."
            ),
            (
                "**1) 이것은 무엇인가요?**\n"
                "브랜드의 주간 주목도(언급수)와 모멘텀(전주 대비 변화)을 사분면으로 시각화합니다.\n\n"
                "**2) 방법**\n"
                "X축: 이번 주 언급수, Y축: 전주 대비 변화량. 기준선은 중앙값.\n\n"
                "**3) 의미**\n"
                "오른쪽 위: 급등 대형주, 오른쪽 아래: 대형이나 둔화, 왼쪽 위: 소형이지만 급상승, 왼쪽 아래: 부진.\n\n"
                "**4) 시사점**\n"
                "급등/둔화 브랜드를 한눈에 파악할 수 있습니다.\n\n"
                "**5) 활용 방안**\n"
                "오른쪽 위(급등 대형)에 집중해 추가 분석·투자 검토하세요."
            )
        )
    else:
        st.info(explain("Need at least 2 weeks of data to compute momentum.", "모멘텀 계산을 위해 최소 2주 데이터가 필요합니다."))

# -----------------------------
# Top subreddits + Top posts (reuse from your original)
# -----------------------------
with st.expander(explain("Top subreddits for selected brands", "선택 브랜드별 인기 서브레딧")):
    if raw is None:
        st.info(explain(f"{RAW_MATCHES} not found.", f"{RAW_MATCHES} 파일을 찾을 수 없습니다."))
    else:
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        rsub = raw.loc[rmask].copy()
        if "keyword" in rsub.columns:
            rsub = rsub[rsub["keyword"].isin(selected)]
        if rsub.empty:
            st.info(explain("No raw matches for the selected brands/date range.", "선택한 브랜드/기간에 대한 데이터가 없습니다."))
        else:
            top_subs = (rsub.groupby(["keyword", "subreddit"]).size().reset_index(name="posts"))
            for b in selected:
                sub_b = top_subs[top_subs["keyword"] == b].sort_values("posts", ascending=False).head(10)
                if not sub_b.empty:
                    st.markdown(f"**{b}**")
                    st.dataframe(sub_b[["subreddit", "posts"]], use_container_width=True)
            # Markdown description after all brand tables
            st.markdown(explain(
                "**1) What is this?**\n"
                "Top 10 subreddits by post count for each selected brand in the date window.\n\n"
                "**2) How**\n"
                "Counts posts per subreddit, per brand, sorts by volume.\n\n"
                "**3) Meaning**\n"
                "Shows where each brand is most discussed.\n\n"
                "**4) Implications**\n"
                "Concentration may suggest strong niche; broad spread signals mass appeal.\n\n"
                "**5) Actions**\n"
                "Target high-volume subreddits for engagement or marketing.",
                "**1) 이것은 무엇인가요?**\n"
                "선택 브랜드별 기간 내 상위 10개 서브레딧(포스트 수 기준)입니다.\n\n"
                "**2) 방법**\n"
                "브랜드별 서브레딧 포스트 수를 집계해 정렬합니다.\n\n"
                "**3) 의미**\n"
                "각 브랜드가 어디서 가장 많이 논의되는지 파악할 수 있습니다.\n\n"
                "**4) 시사점**\n"
                "집중은 강한 틈새, 확산은 대중성을 시사합니다.\n\n"
                "**5) 활용 방안**\n"
                "상위 서브레딧을 타겟으로 참여·마케팅을 진행하세요."
            ))

with st.expander(explain("Top posts (by score) in window", "기간 내 최고 점수 포스트")):
    if raw is None:
        st.info(explain(f"{RAW_MATCHES} not found.", f"{RAW_MATCHES} 파일을 찾을 수 없습니다."))
    else:
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        rsub = raw.loc[rmask].copy()
        if "keyword" in rsub.columns and selected:
            rsub = rsub[rsub["keyword"].isin(selected)]
        cols = [c for c in ["date", "keyword", "alias", "subreddit", "title", "score", "num_comments"] if c in rsub.columns]
        top_posts = rsub.sort_values("score", ascending=False)[cols].head(50)
        st.dataframe(top_posts, use_container_width=True)
        # Markdown description after top posts table
        st.markdown(explain(
            "**1) What is this?**\n"
            "Top 50 posts by score for selected brands in the current window.\n\n"
            "**2) How**\n"
            "Filters by brand and date, sorts by score descending.\n\n"
            "**3) Meaning**\n"
            "Highlights the most impactful content.\n\n"
            "**4) Implications**\n"
            "Big spikes may be driven by a handful of high-scoring posts.\n\n"
            "**5) Actions**\n"
            "Analyze top posts for themes, formats, and subreddit context.",
            "**1) 이것은 무엇인가요?**\n"
            "선택 브랜드의 기간 내 최고 점수(상위 50개) 포스트입니다.\n\n"
            "**2) 방법**\n"
            "브랜드·기간으로 필터링 후 점수 내림차순 정렬.\n\n"
            "**3) 의미**\n"
            "가장 영향력 있는 콘텐츠를 확인할 수 있습니다.\n\n"
            "**4) 시사점**\n"
            "스파이크는 소수의 고득점 포스트가 주도할 수 있습니다.\n\n"
            "**5) 활용 방안**\n"
            "상위 포스트의 주제, 포맷, 서브레딧 맥락을 분석하세요."
        ))