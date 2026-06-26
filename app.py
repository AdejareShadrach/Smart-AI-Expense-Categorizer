import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "data" / "expenses.csv"

CATEGORIES = ["Travel", "Meals", "Office Supplies", "Software", "Utilities", "Marketing", "Equipment"]


def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_categorizer(df: pd.DataFrame) -> Pipeline:
    """Build a text classification model to categorize expenses."""
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=100)
    classifier = MultinomialNB()

    model = Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )
    model.fit(df["description"], df["category"])
    return model


def categorize_expenses(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Predict categories and confidence scores for expenses."""
    result_df = df.copy()

    predictions = model.predict(df["description"])
    probabilities = model.predict_proba(df["description"])
    max_confidence = probabilities.max(axis=1)

    result_df["predicted_category"] = predictions
    result_df["confidence"] = max_confidence
    result_df["is_correct"] = result_df["predicted_category"] == result_df["category"]

    result_df["confidence_level"] = pd.cut(
        result_df["confidence"],
        bins=[0, 0.5, 0.7, 0.85, 1.0],
        labels=["Low", "Medium", "High", "Very High"],
        include_lowest=True,
    )

    return result_df


def main() -> None:
    st.set_page_config(page_title="Smart Expense Categorizer", layout="wide")
    st.title("Smart Expense Categorizer")
    st.write(
        "This AI-powered tool automatically categorizes business expenses for accounting and audit purposes."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file with columns: expense_id, description, amount, date, category",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_data(DATA_PATH)

    if df.empty:
        st.error("No expense data is available. Please run the data generator first.")
        st.stop()

    st.sidebar.header("Model Settings")
    use_ai = st.sidebar.checkbox("Use AI categorization", value=True)

    model = build_categorizer(df)
    categorized_df = categorize_expenses(model, df)

    correct_count = categorized_df["is_correct"].sum()
    accuracy = (correct_count / len(categorized_df)) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Expenses", f"{len(categorized_df):,}")
    col2.metric("AI Accuracy", f"{accuracy:.1f}%")
    col3.metric("Total Amount", f"${categorized_df['amount'].sum():,.2f}")
    col4.metric("Avg Confidence", f"{categorized_df['confidence'].mean():.1%}")

    st.subheader("Expense Review")
    review_tab, category_tab, export_tab = st.tabs(
        ["Low Confidence Expenses", "By Category", "Export"]
    )

    with review_tab:
        low_confidence = categorized_df[categorized_df["confidence"] < 0.7].sort_values(
            "confidence"
        )
        if low_confidence.empty:
            st.success("All expenses have high confidence predictions!")
        else:
            st.warning(f"Found {len(low_confidence)} expenses with low confidence (< 70%)")
            display_df = low_confidence[
                ["expense_id", "description", "amount", "predicted_category", "confidence"]
            ].copy()
            st.dataframe(display_df, use_container_width=True)

    with category_tab:
        category_summary = categorized_df.groupby("predicted_category").agg(
            {"amount": ["sum", "count"], "confidence": "mean"}
        ).round(2)
        st.bar_chart(categorized_df.groupby("predicted_category")["amount"].sum())

        st.write("### Category Breakdown")
        for category in CATEGORIES:
            cat_data = categorized_df[categorized_df["predicted_category"] == category]
            if len(cat_data) > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric(f"{category} Count", len(cat_data))
                col2.metric(f"{category} Total", f"${cat_data['amount'].sum():,.2f}")
                col3.metric(f"{category} Avg Confidence", f"{cat_data['confidence'].mean():.1%}")

    with export_tab:
        csv_data = categorized_df[
            ["expense_id", "description", "amount", "date", "predicted_category", "confidence"]
        ].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download categorized expenses",
            csv_data,
            "categorized_expenses.csv",
            "text/csv",
        )

    st.caption("Built for accounting, audit, and finance analysis projects.")


if __name__ == "__main__":
    main()
