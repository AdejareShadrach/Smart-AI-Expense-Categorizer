from pathlib import Path
import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent / "data" / "expenses.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

categories = ["Travel", "Meals", "Office Supplies", "Software", "Utilities", "Marketing", "Equipment"]

expense_templates = {
    "Travel": [
        "Uber ride downtown",
        "Flight to NYC conference",
        "Hotel stay for business trip",
        "Taxi to airport",
        "Car rental for client visit",
        "Train ticket",
    ],
    "Meals": [
        "Lunch at restaurant with client",
        "Coffee meeting",
        "Team lunch",
        "Dinner for business discussion",
        "Catering for office event",
        "Breakfast meeting",
    ],
    "Office Supplies": [
        "Printer paper and ink",
        "Desk chair",
        "Filing cabinet",
        "Notepads and pens",
        "Folders and envelopes",
        "Stapler and tape",
    ],
    "Software": [
        "Monthly Slack subscription",
        "Microsoft Office license",
        "Adobe Creative Suite",
        "Project management tool",
        "Accounting software",
        "Cloud storage subscription",
    ],
    "Utilities": [
        "Electricity bill",
        "Internet service",
        "Water bill",
        "Gas bill",
        "Phone service",
        "Cleaning service",
    ],
    "Marketing": [
        "Google Ads campaign",
        "Social media marketing",
        "Email campaign platform",
        "Design work for brochure",
        "Website hosting",
        "LinkedIn ads",
    ],
    "Equipment": [
        "Laptop purchase",
        "Monitor",
        "Printer",
        "Camera equipment",
        "Furniture for office",
        "Desk lamp",
    ],
}

rows = []
for i in range(200):
    category = rng.choice(categories)
    description = rng.choice(expense_templates[category])
    amount = rng.normal(loc=150, scale=100)
    if rng.random() < 0.05:
        amount = rng.choice([2000, 5000, 8000])

    date = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 180), unit="D")

    rows.append(
        {
            "expense_id": f"EXP{i+1:04d}",
            "description": description,
            "amount": round(float(amount), 2),
            "date": date.strftime("%Y-%m-%d"),
            "category": category,
        }
    )

frame = pd.DataFrame(rows)
frame.to_csv(OUT_PATH, index=False)
print(f"Created {len(frame)} sample expenses at {OUT_PATH}")
