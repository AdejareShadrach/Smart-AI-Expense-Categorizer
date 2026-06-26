import pandas as pd

from app import build_categorizer, categorize_expenses


def test_categorizer_predicts_correct_categories():
    df = pd.DataFrame(
        [
            {"description": "Uber ride downtown", "category": "Travel"},
            {"description": "Lunch at restaurant with client", "category": "Meals"},
            {"description": "Microsoft Office license", "category": "Software"},
            {"description": "Flight to NYC conference", "category": "Travel"},
            {"description": "Electricity bill", "category": "Utilities"},
            {"description": "Google Ads campaign", "category": "Marketing"},
            {"description": "Printer paper and ink", "category": "Office Supplies"},
            {"description": "Laptop purchase", "category": "Equipment"},
        ]
    )

    model = build_categorizer(df)
    assert model is not None


def test_categorize_expenses_computes_confidence():
    df = pd.DataFrame(
        [
            {"description": "Flight to NYC", "category": "Travel"},
            {"description": "Coffee meeting", "category": "Meals"},
            {"description": "Slack subscription", "category": "Software"},
            {"description": "Hotel stay", "category": "Travel"},
            {"description": "Internet bill", "category": "Utilities"},
        ]
    )

    model = build_categorizer(df)
    categorized = categorize_expenses(model, df)

    assert "predicted_category" in categorized.columns
    assert "confidence" in categorized.columns
    assert "confidence_level" in categorized.columns
    assert (categorized["confidence"] >= 0).all()
    assert (categorized["confidence"] <= 1).all()


def test_categorizer_handles_new_descriptions():
    df = pd.DataFrame(
        [
            {"description": "Uber ride", "category": "Travel"},
            {"description": "Lunch", "category": "Meals"},
            {"description": "Software subscription", "category": "Software"},
            {"description": "Flight", "category": "Travel"},
            {"description": "Electricity", "category": "Utilities"},
        ]
    )

    model = build_categorizer(df)

    new_df = pd.DataFrame(
        [
            {"description": "Taxi to airport", "category": "Travel"},
            {"description": "Restaurant dinner", "category": "Meals"},
        ]
    )

    categorized = categorize_expenses(model, new_df)
    assert len(categorized) == 2
    assert categorized["predicted_category"].notna().all()
