"""
data_utils.py

This file contains data loading, preprocessing, and simple NLP modeling utilities
for the GenAI User Complaint Dashboard.
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path("data/genai_reviews.csv")


@pd.api.extensions.register_dataframe_accessor("clean_check")
class _CleanCheckAccessor:
    """Small helper accessor used only for readable validation checks."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def has_columns(self, required_columns):
        return all(col in self._obj.columns for col in required_columns)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the Kaggle CSV from the local data folder.

    The app does not use file upload. The dataset must be saved as:
    data/genai_reviews.csv
    """
    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. Please place the CSV at data/genai_reviews.csv"
        )

    df = pd.read_csv(path)
    return df


def normalize_text(text: str) -> str:
    """
    Clean review text for keyword modeling.

    Steps:
    1. Convert text to lowercase.
    2. Remove URLs.
    3. Remove punctuation and special characters.
    4. Compress extra spaces.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the raw dataset for dashboard analysis.

    Main preprocessing work:
    - Convert review dates to datetime.
    - Remove rows with missing review text.
    - Create cleaned text for NLP modeling.
    - Create negative-review labels for complaint analysis.
    - Create monthly date values for trend charts.
    """
    required_columns = [
        "App",
        "Review_Date",
        "Star_Rating",
        "Review_Text",
        "Sentiment_Polarity",
        "Review_Theme",
    ]

    if not df.clean_check.has_columns(required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Convert dates safely. Invalid dates become NaT and are removed later.
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")

    # Make numeric columns safe for calculations.
    df["Star_Rating"] = pd.to_numeric(df["Star_Rating"], errors="coerce")
    df["Sentiment_Polarity"] = pd.to_numeric(df["Sentiment_Polarity"], errors="coerce")
    df["Thumbs_Up_Count"] = pd.to_numeric(
        df.get("Thumbs_Up_Count", 0), errors="coerce"
    ).fillna(0)

    # Remove rows that cannot support the analysis.
    df = df.dropna(subset=["App", "Review_Date", "Star_Rating", "Review_Text"])

    # Clean review text for TF-IDF keyword extraction.
    df["Clean_Text"] = df["Review_Text"].apply(normalize_text)

    # Define negative reviews. This combines star rating and sentiment polarity.
    # A review is treated as a complaint if it has a low rating or negative sentiment.
    df["Is_Negative"] = (df["Star_Rating"] <= 2) | (df["Sentiment_Polarity"] < 0)

    # Month column supports time-series complaint trend visualization.
    df["Review_Month"] = df["Review_Date"].dt.to_period("M").dt.to_timestamp()

    return df


def filter_data(
    df: pd.DataFrame,
    selected_apps,
    selected_themes,
    date_range,
    min_thumbs_up: int,
) -> pd.DataFrame:
    """Apply Streamlit sidebar filters to the cleaned dataframe."""
    filtered = df.copy()

    if selected_apps:
        filtered = filtered[filtered["App"].isin(selected_apps)]

    if selected_themes:
        filtered = filtered[filtered["Review_Theme"].isin(selected_themes)]

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["Review_Date"].dt.date >= start_date)
            & (filtered["Review_Date"].dt.date <= end_date)
        ]

    filtered = filtered[filtered["Thumbs_Up_Count"] >= min_thumbs_up]

    return filtered


def get_negative_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Return only negative reviews used for complaint-focused analysis."""
    return df[df["Is_Negative"]].copy()


def extract_tfidf_keywords(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    NLP modeling step: extract important complaint keywords using TF-IDF.

    TF-IDF gives higher weight to words that are frequent in complaint reviews
    but not common everywhere. This helps identify what users complain about.
    """
    if df.empty:
        return pd.DataFrame(columns=["Keyword", "TF_IDF_Score"])

    texts = df["Clean_Text"].dropna().astype(str)
    texts = texts[texts.str.len() > 0]

    if texts.empty:
        return pd.DataFrame(columns=["Keyword", "TF_IDF_Score"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Average TF-IDF score across negative reviews.
    scores = tfidf_matrix.mean(axis=0).A1

    keyword_df = pd.DataFrame(
        {"Keyword": feature_names, "TF_IDF_Score": scores}
    ).sort_values("TF_IDF_Score", ascending=False)

    return keyword_df.head(top_n)
