# Smart Expense Categorizer

An AI-powered tool that automatically categorizes business expenses for accounting and audit purposes using machine learning text classification.

## Features
- Automatically categorizes expenses using TF-IDF + Naive Bayes
- Provides confidence scores for each prediction
- Identifies low-confidence categorizations for manual review
- Shows category breakdown with totals and averages
- Exports categorized expenses to CSV
- Interactive dashboard with Streamlit

## How It Works
1. Loads expense descriptions from CSV file
2. Builds a text classification model trained on historical data
3. Predicts expense categories with confidence scores
4. Flags expenses with low confidence (< 70%) for review
5. Provides insights by category and spending patterns

## Expense Categories
- Travel (flights, hotels, taxis)
- Meals (lunches, dinners, coffee meetings)
- Office Supplies (paper, pens, furniture)
- Software (subscriptions, licenses)
- Utilities (electricity, internet, phone)
- Marketing (ads, campaigns)
- Equipment (laptops, monitors, cameras)

## Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate sample data:
   ```bash
   python generate_sample_data.py
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Upload Your Own CSV
The app accepts CSV files with these columns:
- `expense_id`: Unique identifier
- `description`: Expense description (text to categorize)
- `amount`: Expense amount in USD
- `date`: Transaction date
- `category`: Actual category (for training)

## Use Cases
- Automated accounting workflow
- Expense audit and validation
- Financial analysis and reporting
- Cost center allocation
- Compliance and tax preparation

## Portfolio Value
This project demonstrates:
- Natural Language Processing (NLP) / text classification
- Machine learning with scikit-learn
- Data analysis with pandas
- Practical accounting domain knowledge
- Interactive data visualization with Streamlit
- Production-ready code with testing
