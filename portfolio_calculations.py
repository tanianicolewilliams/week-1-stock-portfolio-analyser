"""Verified Step 5 portfolio calculations and market-price retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import yfinance as yf


QUANTITY_TOLERANCE = 1e-9
HISTORICAL_PRICE_PERIODS = {
    "1mo": "1 month",
    "6mo": "6 months",
    "1y": "1 year",
    "5y": "5 years",
    "10y": "10 years",
    "max": "All time",
}
YFINANCE_CACHE_DIRECTORY = (
    Path(gettempdir()) / "week1_stock_portfolio_yfinance_cache"
)
YFINANCE_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIRECTORY))


class PortfolioCalculationError(ValueError):
    """A portfolio transaction sequence cannot be calculated consistently."""


def calculate_transaction_cash_flows(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return date-ordered cash activity derived from validated transactions."""
    prepared = transactions.copy()
    prepared["normalized_transaction_type"] = (
        prepared["transaction_type"].astype(str).str.strip().str.upper()
    )

    unsupported_types = sorted(
        set(prepared["normalized_transaction_type"]) - {"BUY", "SELL"}
    )
    if unsupported_types:
        raise PortfolioCalculationError(
            "The performance analysis supports only BUY and SELL transactions."
        )

    transaction_amount = (
        pd.to_numeric(prepared["quantity"], errors="coerce")
        * pd.to_numeric(prepared["price"], errors="coerce")
    )
    if transaction_amount.isna().any():
        raise PortfolioCalculationError(
            "Every transaction needs a usable quantity and price."
        )

    transaction_dates = pd.to_datetime(prepared["date"], errors="coerce")
    if transaction_dates.isna().any():
        raise PortfolioCalculationError(
            "Every transaction needs a usable date for performance analysis."
        )

    is_buy = prepared["normalized_transaction_type"].eq("BUY")
    prepared["date"] = transaction_dates.dt.normalize()
    prepared["purchase_outflow"] = transaction_amount.where(is_buy, 0.0)
    prepared["sale_proceeds"] = transaction_amount.where(~is_buy, 0.0)
    prepared["cash_flow"] = (
        prepared["sale_proceeds"] - prepared["purchase_outflow"]
    )

    daily_cash_activity = (
        prepared.groupby("date", as_index=False, sort=True)[
            ["purchase_outflow", "sale_proceeds", "cash_flow"]
        ]
        .sum()
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    daily_cash_activity["cumulative_purchase_outflows"] = daily_cash_activity[
        "purchase_outflow"
    ].cumsum()
    daily_cash_activity["cumulative_sale_proceeds"] = daily_cash_activity[
        "sale_proceeds"
    ].cumsum()
    daily_cash_activity["cumulative_net_cash_invested"] = (
        daily_cash_activity["cumulative_purchase_outflows"]
        - daily_cash_activity["cumulative_sale_proceeds"]
    )
    return daily_cash_activity


def calculate_performance_metrics(
    cash_activity: pd.DataFrame,
    current_market_value: float | None,
) -> dict[str, float | None]:
    """Summarise transaction totals and current outcome when prices are complete."""
    total_purchase_outflows = float(cash_activity["purchase_outflow"].sum())
    total_sale_proceeds = float(cash_activity["sale_proceeds"].sum())
    net_cash_invested = total_purchase_outflows - total_sale_proceeds

    if current_market_value is None:
        return {
            "total_purchase_outflows": total_purchase_outflows,
            "total_sale_proceeds": total_sale_proceeds,
            "net_cash_invested": net_cash_invested,
            "current_market_value": None,
            "total_current_outcome": None,
            "absolute_return": None,
            "percentage_return": None,
        }

    current_market_value = float(current_market_value)
    total_current_outcome = total_sale_proceeds + current_market_value
    absolute_return = total_current_outcome - total_purchase_outflows
    percentage_return = None
    if total_purchase_outflows > QUANTITY_TOLERANCE:
        percentage_return = absolute_return / total_purchase_outflows * 100

    return {
        "total_purchase_outflows": total_purchase_outflows,
        "total_sale_proceeds": total_sale_proceeds,
        "net_cash_invested": net_cash_invested,
        "current_market_value": current_market_value,
        "total_current_outcome": total_current_outcome,
        "absolute_return": absolute_return,
        "percentage_return": percentage_return,
    }


def _xnpv(
    rate: float,
    cash_flows: list[float],
    year_fractions: list[float],
) -> float:
    """Calculate net present value for irregularly dated cash flows."""
    return sum(
        cash_flow / ((1.0 + rate) ** year_fraction)
        for cash_flow, year_fraction in zip(cash_flows, year_fractions)
    )


def _bisect_xirr_root(
    lower_rate: float,
    upper_rate: float,
    cash_flows: list[float],
    year_fractions: list[float],
    tolerance: float,
) -> float:
    """Find one XIRR root inside a bracket whose endpoints have opposite signs."""
    lower_value = _xnpv(lower_rate, cash_flows, year_fractions)

    for _ in range(200):
        midpoint = (lower_rate + upper_rate) / 2.0
        midpoint_value = _xnpv(midpoint, cash_flows, year_fractions)
        if abs(midpoint_value) <= tolerance:
            return midpoint

        if lower_value * midpoint_value <= 0:
            upper_rate = midpoint
        else:
            lower_rate = midpoint
            lower_value = midpoint_value

    return (lower_rate + upper_rate) / 2.0


def calculate_xirr(
    cash_activity: pd.DataFrame,
    current_market_value: float,
    valuation_date: object,
) -> float:
    """Return a single reliable annualised return for irregular cash flows."""
    dated_cash_flows = cash_activity[["date", "cash_flow"]].copy()
    dated_cash_flows["date"] = pd.to_datetime(
        dated_cash_flows["date"], errors="coerce"
    ).dt.normalize()
    if dated_cash_flows["date"].isna().any():
        raise PortfolioCalculationError(
            "Annualised return is unavailable because a transaction date is invalid."
        )

    normalized_valuation_date = pd.Timestamp(valuation_date).normalize()
    if normalized_valuation_date < dated_cash_flows["date"].max():
        raise PortfolioCalculationError(
            "Annualised return is unavailable because a transaction is dated after the valuation date."
        )

    if current_market_value > QUANTITY_TOLERANCE:
        terminal_value = pd.DataFrame(
            {
                "date": [normalized_valuation_date],
                "cash_flow": [float(current_market_value)],
            }
        )
        dated_cash_flows = pd.concat(
            [dated_cash_flows, terminal_value], ignore_index=True
        )

    dated_cash_flows = (
        dated_cash_flows.groupby("date", as_index=False, sort=True)["cash_flow"]
        .sum()
        .loc[lambda rows: rows["cash_flow"].abs() > QUANTITY_TOLERANCE]
        .reset_index(drop=True)
    )
    cash_flows = dated_cash_flows["cash_flow"].astype(float).tolist()
    if not cash_flows or not any(value < 0 for value in cash_flows):
        raise PortfolioCalculationError(
            "Annualised return needs at least one purchase outflow."
        )
    if not any(value > 0 for value in cash_flows):
        raise PortfolioCalculationError(
            "Annualised return needs at least one positive sale or ending value."
        )

    first_date = dated_cash_flows["date"].min()
    year_fractions = (
        (dated_cash_flows["date"] - first_date).dt.days / 365.0
    ).astype(float).tolist()
    npv_tolerance = max(1e-7, sum(abs(value) for value in cash_flows) * 1e-10)

    # Search rates from almost -100% through 100,000%, then accept only one root.
    rate_grid = [
        math.exp(
            math.log(1e-6)
            + step * (math.log(1001.0) - math.log(1e-6)) / 600
        )
        - 1.0
        for step in range(601)
    ]
    roots: list[float] = []
    previous_rate = rate_grid[0]
    previous_value = _xnpv(previous_rate, cash_flows, year_fractions)

    for current_rate in rate_grid[1:]:
        current_value = _xnpv(current_rate, cash_flows, year_fractions)
        if abs(previous_value) <= npv_tolerance:
            roots.append(previous_rate)
        elif previous_value * current_value < 0:
            roots.append(
                _bisect_xirr_root(
                    previous_rate,
                    current_rate,
                    cash_flows,
                    year_fractions,
                    npv_tolerance,
                )
            )
        previous_rate = current_rate
        previous_value = current_value

    if abs(previous_value) <= npv_tolerance:
        roots.append(previous_rate)

    unique_roots: list[float] = []
    for root in roots:
        if not any(abs(root - existing_root) < 1e-7 for existing_root in unique_roots):
            unique_roots.append(root)

    if not unique_roots:
        raise PortfolioCalculationError(
            "A reliable annualised return could not be found for these cash flows."
        )
    if len(unique_roots) > 1:
        raise PortfolioCalculationError(
            "The annualised return is ambiguous because these cash flows produce more than one result."
        )
    return unique_roots[0]


def calculate_fifo_holdings(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return open holdings after matching sales to the earliest purchase lots."""
    ordered = transactions.copy()
    ordered["source_order"] = range(len(ordered))
    ordered["normalized_ticker"] = (
        ordered["ticker"].astype(str).str.strip().str.upper()
    )
    ordered["normalized_transaction_type"] = (
        ordered["transaction_type"].astype(str).str.strip().str.upper()
    )
    ordered = ordered.sort_values(
        ["normalized_ticker", "date", "source_order"], kind="stable"
    )

    holding_rows: list[dict[str, float | str]] = []

    for ticker, ticker_transactions in ordered.groupby("normalized_ticker", sort=True):
        purchase_lots: list[list[float]] = []

        for transaction in ticker_transactions.itertuples(index=False):
            transaction_type = transaction.normalized_transaction_type
            quantity = float(transaction.quantity)
            price = float(transaction.price)

            if transaction_type == "BUY":
                purchase_lots.append([quantity, price])
                continue

            if transaction_type != "SELL":
                raise PortfolioCalculationError(
                    f"{ticker} contains an unsupported transaction type."
                )

            quantity_to_match = quantity
            while quantity_to_match > QUANTITY_TOLERANCE and purchase_lots:
                earliest_lot = purchase_lots[0]
                matched_quantity = min(quantity_to_match, earliest_lot[0])
                earliest_lot[0] -= matched_quantity
                quantity_to_match -= matched_quantity

                if earliest_lot[0] <= QUANTITY_TOLERANCE:
                    purchase_lots.pop(0)

            if quantity_to_match > QUANTITY_TOLERANCE:
                raise PortfolioCalculationError(
                    f"{ticker} sells more shares than the earlier purchases provide."
                )

        shares_held = sum(lot_quantity for lot_quantity, _ in purchase_lots)
        if shares_held <= QUANTITY_TOLERANCE:
            continue

        remaining_cost_basis = sum(
            lot_quantity * lot_price for lot_quantity, lot_price in purchase_lots
        )
        holding_rows.append(
            {
                "ticker": ticker,
                "shares_held": shares_held,
                "remaining_cost_basis": remaining_cost_basis,
                "average_purchase_price": remaining_cost_basis / shares_held,
            }
        )

    return pd.DataFrame(
        holding_rows,
        columns=[
            "ticker",
            "shares_held",
            "remaining_cost_basis",
            "average_purchase_price",
        ],
    )


def _usable_closing_prices(history: pd.DataFrame) -> pd.Series:
    """Return numeric non-missing closing prices from a yfinance response."""
    if history.empty or "Close" not in history.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(history["Close"], errors="coerce").dropna()


def _format_market_timestamp(timestamp: object, intraday: bool) -> str:
    """Format the market timestamp without inventing precision for daily data."""
    market_timestamp = pd.Timestamp(timestamp)
    if not intraday:
        return market_timestamp.strftime("%Y-%m-%d")
    return market_timestamp.strftime("%Y-%m-%d %H:%M %Z").strip()


def _fetch_latest_price(ticker: str) -> tuple[float, str, str] | None:
    """Fetch the latest intraday price, falling back to the latest daily close."""
    price_history = yf.Ticker(ticker)
    attempts = [
        {
            "period": "1d",
            "interval": "1m",
            "prepost": True,
            "intraday": True,
            "status": "Latest available intraday price",
        },
        {
            "period": "1mo",
            "interval": "1d",
            "prepost": False,
            "intraday": False,
            "status": "Latest available market close",
        },
    ]

    for attempt in attempts:
        try:
            history = price_history.history(
                period=attempt["period"],
                interval=attempt["interval"],
                prepost=attempt["prepost"],
                auto_adjust=False,
                actions=False,
                timeout=10,
            )
        except Exception:
            continue

        closing_prices = _usable_closing_prices(history)
        if closing_prices.empty:
            continue

        latest_timestamp = closing_prices.index[-1]
        return (
            float(closing_prices.iloc[-1]),
            _format_market_timestamp(latest_timestamp, attempt["intraday"]),
            str(attempt["status"]),
        )

    return None


def fetch_latest_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Return the latest available price and a friendly status for each ticker."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    price_rows: list[dict[str, float | str | None]] = []

    for ticker in tickers:
        try:
            latest_price = _fetch_latest_price(ticker)
        except Exception:
            latest_price = None

        if latest_price is None:
            price_rows.append(
                {
                    "ticker": ticker,
                    "current_price": None,
                    "price_as_of": "Not available",
                    "price_status": (
                        "Unavailable - yfinance did not return a usable price."
                    ),
                    "retrieved_at": retrieved_at,
                }
            )
            continue

        current_price, price_as_of, price_status = latest_price
        price_rows.append(
            {
                "ticker": ticker,
                "current_price": current_price,
                "price_as_of": price_as_of,
                "price_status": price_status,
                "retrieved_at": retrieved_at,
            }
        )

    return pd.DataFrame(price_rows)


def fetch_historical_price_performance(
    tickers: tuple[str, ...],
    period_key: str,
) -> pd.DataFrame:
    """Return validated adjusted closing-price returns for a selected period."""
    if period_key not in HISTORICAL_PRICE_PERIODS:
        raise ValueError(f"Unsupported historical price period: {period_key}")

    retrieved_at = datetime.now(timezone.utc).isoformat()
    result_rows: list[dict[str, float | str | None]] = []

    for ticker in tickers:
        start_price: float | None = None
        end_price: float | None = None
        start_date = "Not available"
        end_date = "Not available"
        return_percent: float | None = None
        status = "Unavailable - yfinance did not return usable historical prices."

        try:
            history = yf.Ticker(ticker).history(
                period=period_key,
                interval="1d",
                prepost=False,
                auto_adjust=True,
                actions=False,
                repair=False,
                timeout=15,
            )
            closing_prices = _usable_closing_prices(history)
            if len(closing_prices) >= 2:
                possible_start_price = float(closing_prices.iloc[0])
                possible_end_price = float(closing_prices.iloc[-1])
                if (
                    math.isfinite(possible_start_price)
                    and math.isfinite(possible_end_price)
                    and possible_start_price > 0
                    and possible_end_price > 0
                ):
                    start_price = possible_start_price
                    end_price = possible_end_price
                    start_date = _format_market_timestamp(
                        closing_prices.index[0],
                        intraday=False,
                    )
                    end_date = _format_market_timestamp(
                        closing_prices.index[-1],
                        intraday=False,
                    )
                    return_percent = (
                        (end_price / start_price) - 1
                    ) * 100
                    status = "Available - yfinance adjusted daily closing prices."
        except Exception:
            pass

        result_rows.append(
            {
                "ticker": ticker,
                "period_key": period_key,
                "period_label": HISTORICAL_PRICE_PERIODS[period_key],
                "start_price": start_price,
                "start_date": start_date,
                "end_price": end_price,
                "end_date": end_date,
                "return_percent": return_percent,
                "price_status": status,
                "retrieved_at": retrieved_at,
            }
        )

    return pd.DataFrame(result_rows)
