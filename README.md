# Stock Portfolio Analyser

A beginner-friendly Streamlit application for uploading a portfolio transaction history, reviewing current holdings, exploring transaction activity, understanding portfolio performance and asking a grounded AI analyst questions about the calculated results.

## What the app does

The app keeps one validated upload available across four tabs in the same browser session:

1. **Data Upload** validates a CSV, reports its row and column counts and shows a preview.
2. **Consolidated Portfolio View** calculates shares held, FIFO cost basis and average purchase price; retrieves current prices through yfinance; and shows current value, unrealised gain or loss, allocation, filters, tables and charts.
3. **Historical Performance** explains purchase outflows, sale proceeds, net cash invested, current holdings value, total outcome, absolute return, simple return and annualised money-weighted return (XIRR).
4. **AI Analyst (Chat)** answers educational questions using verified Python-calculated portfolio facts and relevant validated transaction rows. It also compares adjusted closing-price performance over six user-selected periods.

## CSV format

Upload a comma-separated file with these exact headings:

```text
ticker,date,transaction_type,quantity,price
```

Dates must use `YYYY-MM-DD`. Transaction types must be `BUY` or `SELL`, and quantity and price must be numeric. The repository includes a clean demonstration file and a missing-values file for checking the friendly validation guidance.

## Calculations and data sources

- Remaining shares and cost basis use FIFO: sold shares are matched to the earliest available purchases first.
- Current values use the latest price returned by yfinance and show the price source, status and as-of time.
- Allocation and complete current-value totals are withheld if a required current price is unavailable, preventing misleading partial totals.
- Historical market comparisons use yfinance adjusted daily closing prices for 1 month, 6 months, 1 year, 5 years, 10 years and all time.
- XIRR is an annualised money-weighted return based on the dates and values of recorded cash flows. It is not a time-weighted return.

The uploaded CSV does not identify a currency. Amounts therefore have no currency symbol, no foreign-exchange conversion is performed, and fees, taxes, dividends, interest and unrecorded cash movements are excluded.

## AI safety and fallback behaviour

The AI Analyst uses Groq model `openai/gpt-oss-120b`. Python remains authoritative for portfolio calculations, and model tool use is disabled. The analyst:

- answers only from verified portfolio facts and supplied validated rows;
- clearly says when the available data is insufficient;
- does not invent reasons for market-price movements;
- refuses personalised instructions or quantities to buy, sell, hold or trade; and
- provides deterministic answers from verified Python facts if Groq is missing, unavailable or rate-limited.

This application provides educational analysis, not financial advice, and cannot execute trades.

## Run locally

1. Create and activate a Python environment.
2. Install the dependencies:

   ```text
   pip install -r requirements.txt
   ```

3. Store the Groq key only in a private local `.streamlit/secrets.toml` file. Never commit or share that file.
4. Start the application from the repository root:

   ```text
   streamlit run app.py
   ```

5. Upload `data/portfolio_transactions_clean.csv` to follow the complete demonstration journey.

Outbound HTTPS access is required for yfinance and Groq. If either provider is unavailable, the app keeps the remaining tabs usable and shows friendly, non-technical guidance.

## Streamlit Community Cloud

Deploy `app.py` from the repository root and use `requirements.txt` for the tested package versions. Add the Groq key privately through the app's Streamlit Community Cloud secrets settings; do not place it in GitHub, the README, screenshots, recordings or application code.
