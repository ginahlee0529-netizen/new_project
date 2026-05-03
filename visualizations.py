"""
visualizations.py

This file contains Plotly chart functions used by the Streamlit app.
Each function returns a figure object that can be displayed with st.plotly_chart().
"""

import pandas as pd
import plotly.express as px


def complaint_theme_chart(negative_df: pd.DataFrame):
    """Bar chart: most common complaint themes among negative reviews."""
    theme_counts = (
        negative_df["Review_Theme"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )
    theme_counts.columns = ["Review Theme", "Number of Negative Reviews"]

    fig = px.bar(
        theme_counts.head(15),
        x="Number of Negative Reviews",
        y="Review Theme",
        orientation="h",
        title="Top Complaint Themes in Negative Reviews",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def app_negative_chart(negative_df: pd.DataFrame):
    """Bar chart: which apps have the most negative reviews."""
    app_counts = negative_df["App"].value_counts().reset_index()
    app_counts.columns = ["App", "Number of Negative Reviews"]

    fig = px.bar(
        app_counts,
        x="App",
        y="Number of Negative Reviews",
        title="Apps with the Most Negative Reviews",
    )
    return fig


def complaint_trend_chart(filtered_df: pd.DataFrame):
    """Line chart: monthly negative-review rate over time."""
    trend = (
        filtered_df.groupby("Review_Month")
        .agg(
            Total_Reviews=("Review_Text", "count"),
            Negative_Reviews=("Is_Negative", "sum"),
        )
        .reset_index()
    )

    trend["Negative Review Rate"] = (
        trend["Negative_Reviews"] / trend["Total_Reviews"] * 100
    )

    fig = px.line(
        trend,
        x="Review_Month",
        y="Negative Review Rate",
        markers=True,
        title="Negative Review Rate Over Time",
        labels={"Review_Month": "Month", "Negative Review Rate": "Negative Review Rate (%)"},
    )
    return fig


def sentiment_by_app_chart(filtered_df: pd.DataFrame):
    """Box plot: sentiment polarity distribution by app."""
    fig = px.box(
        filtered_df,
        x="App",
        y="Sentiment_Polarity",
        title="Sentiment Polarity Distribution by App",
    )
    return fig


def keyword_chart(keyword_df: pd.DataFrame):
    """Bar chart: TF-IDF keywords extracted from negative review text."""
    fig = px.bar(
        keyword_df.sort_values("TF_IDF_Score"),
        x="TF_IDF_Score",
        y="Keyword",
        orientation="h",
        title="Most Important Complaint Keywords Using TF-IDF",
    )
    return fig
