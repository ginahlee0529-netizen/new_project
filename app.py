"""
app.py

Streamlit dashboard for analyzing negative user reviews in the Generative AI app ecosystem.
Project focus: What do users complain about in GenAI apps?
"""

import streamlit as st

from data_utils import (
    extract_tfidf_keywords,
    filter_data,
    get_negative_reviews,
    load_data,
    preprocess_data,
)
from visualizations import (
    app_negative_chart,
    complaint_theme_chart,
    complaint_trend_chart,
    keyword_chart,
    sentiment_by_app_chart,
)


st.set_page_config(
    page_title="GenAI User Complaint Dashboard",
    page_icon="🤖",
    layout="wide",
)


@st.cache_data
def cached_load_and_preprocess():
    """
    Load and preprocess data once, then cache it.
    This keeps the dashboard fast when users change filters.
    """
    raw_df = load_data()
    clean_df = preprocess_data(raw_df)
    return clean_df


# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
st.title("GenAI User Complaint Dashboard")
st.caption("A Streamlit dashboard analyzing negative user reviews from GenAI apps.")

df = cached_load_and_preprocess()


# -----------------------------
# 2. Sidebar interactive filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

app_options = sorted(df["App"].dropna().unique())
selected_apps = st.sidebar.multiselect(
    "Select App(s)",
    options=app_options,
    default=app_options,
)

theme_options = sorted(df["Review_Theme"].dropna().unique())
selected_themes = st.sidebar.multiselect(
    "Select Review Theme(s)",
    options=theme_options,
    default=theme_options,
)

min_date = df["Review_Date"].dt.date.min()
max_date = df["Review_Date"].dt.date.max()
date_range = st.sidebar.date_input(
    "Select Review Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

min_thumbs_up = st.sidebar.slider(
    "Minimum Thumbs Up Count",
    min_value=0,
    max_value=int(df["Thumbs_Up_Count"].max()),
    value=0,
)

top_n_keywords = st.sidebar.slider(
    "Number of TF-IDF Keywords",
    min_value=5,
    max_value=30,
    value=15,
)


# -----------------------------
# 3. Apply filters
# -----------------------------
filtered_df = filter_data(
    df=df,
    selected_apps=selected_apps,
    selected_themes=selected_themes,
    date_range=date_range,
    min_thumbs_up=min_thumbs_up,
)

negative_df = get_negative_reviews(filtered_df)


# -----------------------------
# 4. Dashboard summary metrics
# -----------------------------
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Reviews", f"{len(filtered_df):,}")

with col2:
    st.metric("Negative Reviews", f"{len(negative_df):,}")

with col3:
    negative_rate = 0 if len(filtered_df) == 0 else len(negative_df) / len(filtered_df) * 100
    st.metric("Negative Review Rate", f"{negative_rate:.1f}%")

with col4:
    avg_rating = 0 if filtered_df.empty else filtered_df["Star_Rating"].mean()
    st.metric("Average Star Rating", f"{avg_rating:.2f}")


if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()


# -----------------------------
# 5. Visual storytelling section
# -----------------------------
st.subheader("Complaint Analysis")

left_col, right_col = st.columns(2)

with left_col:
    st.plotly_chart(complaint_theme_chart(negative_df), use_container_width=True)

with right_col:
    st.plotly_chart(app_negative_chart(negative_df), use_container_width=True)

st.plotly_chart(complaint_trend_chart(filtered_df), use_container_width=True)

st.subheader("Sentiment and Text Modeling")

left_col, right_col = st.columns(2)

with left_col:
    st.plotly_chart(sentiment_by_app_chart(filtered_df), use_container_width=True)

with right_col:
    # Modeling step: TF-IDF keyword extraction from negative review text.
    keyword_df = extract_tfidf_keywords(negative_df, top_n=top_n_keywords)
    st.plotly_chart(keyword_chart(keyword_df), use_container_width=True)


# -----------------------------
# 6. Show examples of actual complaints
# -----------------------------
st.subheader("Sample Negative Reviews")

sample_columns = [
    "App",
    "Review_Date",
    "Star_Rating",
    "Sentiment_Polarity",
    "Review_Theme",
    "Thumbs_Up_Count",
    "Review_Text",
]

st.dataframe(
    negative_df[sample_columns]
    .sort_values(["Thumbs_Up_Count", "Review_Date"], ascending=False)
    .head(20),
    use_container_width=True,
)
