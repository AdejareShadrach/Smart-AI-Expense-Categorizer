# Smart Expense Categorizer

An AI tool that automatically classifies business expenses into categories (Travel, Meals, Software, etc.) using NLP — built to speed up bookkeeping, audits, and expense reporting.

![Dashboard Screenshot](assets/dashboard.png)
<!-- Replace with your actual screenshot/GIF -->

## Why this exists

Categorizing expenses manually is repetitive and error-prone, especially at scale. This tool learns from historical labeled expenses and predicts categories for new ones — while being honest about its own uncertainty, so low-confidence predictions get routed to a human instead of silently guessed.

## How it works

1. Load expense descriptions + historical categories from CSV
2. Train a **TF-IDF + Naive Bayes** text classification model
3. Predict categories for new expenses, with a confidence score
4. Flag predictions under 70% confidence for manual review
5. Show category totals, averages, and spending breakdown in a dashboard
6. Export categorized results to CSV

## Categories supported

Travel · Meals · Office Supplies · Software · Utilities · Marketing · Equipment

## Example output

| Description              | Predicted Category | Confidence | Review Needed |
|---------------------------|--------------------|------------|---------------|
| "Uber to airport"         | Travel             | 96%        | No             |
| "Zoom monthly plan"        | Software           | 89%        | No             |
| "Office party catering"    | Meals              | 61%        | ⚠️ Yes         |

*(sample rows — replace with real output once you have it)*

## Run locally

```bash
pip install -r requirements.txt
python generate_sample_data.py
streamlit run app.py
```

## Upload your own data

CSV columns expected: `expense_id`, `description`, `amount`, `date`, `category`

## Tech stack

- Python, pandas, scikit-learn
- TF-IDF vectorization + Naive Bayes classifier
- Streamlit dashboard

## What I'd improve next

- Try a stronger model (e.g. fine-tuned transformer) and compare accuracy
- Let users correct mis-categorized expenses and retrain on the fly
- Add multi-currency support

## About me

Built by an accountant learning Python and applied ML for real finance use cases.
