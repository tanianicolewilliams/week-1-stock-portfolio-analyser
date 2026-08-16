"""Week 1 Stock Portfolio Analyser - Step 8: grounded AI Analyst chat."""

from io import BytesIO

import pandas as pd
import streamlit as st

from ai_analyst import (
    build_historical_price_facts,
    build_portfolio_facts,
    format_historical_price_response,
    local_safety_response,
    portfolio_signature,
    request_groq_analysis,
    select_relevant_transactions,
)
from portfolio_calculations import (
    HISTORICAL_PRICE_PERIODS,
    PortfolioCalculationError,
    calculate_fifo_holdings,
    calculate_performance_metrics,
    calculate_transaction_cash_flows,
    calculate_xirr,
    fetch_latest_prices,
    fetch_historical_price_performance,
)


REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "transaction_type",
    "quantity",
    "price",
]
NUMERIC_COLUMNS = ["quantity", "price"]
TEXT_COLUMNS = ["ticker", "transaction_type"]
CSV_TEMPLATE = ",".join(REQUIRED_COLUMNS) + "\n"


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV file into a pandas dataframe."""
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(ttl=900, show_spinner=False)
def load_market_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch and cache the latest available market prices for 15 minutes."""
    return fetch_latest_prices(tickers)


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_price_performance(
    tickers: tuple[str, ...],
    period_key: str,
) -> pd.DataFrame:
    """Fetch and cache a selected historical market comparison for one hour."""
    return fetch_historical_price_performance(tickers, period_key)


def describe_transaction_rows(problem_mask: pd.Series) -> str:
    """Return friendly one-based transaction-row references for a problem mask."""
    row_numbers = [
        str(row_number)
        for row_number, has_problem in enumerate(problem_mask.tolist(), start=1)
        if has_problem
    ]
    if len(row_numbers) == 1:
        return f"transaction row {row_numbers[0]}"
    return "transaction rows " + ", ".join(row_numbers[:-1]) + " and " + row_numbers[-1]


def validate_and_prepare(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate required fields and safely convert dates and numbers."""
    prepared = dataframe.copy()
    errors: list[str] = []

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in prepared.columns
    ]
    if missing_columns:
        errors.append(
            "The CSV is missing the required column(s): "
            + ", ".join(missing_columns)
            + ". Add the exact headings: "
            + ", ".join(REQUIRED_COLUMNS)
            + "."
        )
        return prepared, errors

    if prepared.empty:
        errors.append(
            "The CSV has the required headings but no transaction rows. "
            "Add at least one BUY or SELL transaction, then upload it again."
        )
        return prepared, errors

    for column in TEXT_COLUMNS:
        field_label = column.replace("_", " ")
        missing_text_mask = prepared[column].isna() | (
            prepared[column].astype(str).str.strip() == ""
        )
        missing_text_count = int(missing_text_mask.sum())
        if missing_text_count:
            value_word = "value" if missing_text_count == 1 else "values"
            errors.append(
                f"The {field_label} column is missing {missing_text_count} required "
                f"{value_word} in {describe_transaction_rows(missing_text_mask)}. "
                f"Add a {field_label} to every transaction, then upload the CSV again."
            )

    converted_dates = pd.to_datetime(prepared["date"], errors="coerce")
    invalid_date_mask = converted_dates.isna()
    invalid_date_count = int(invalid_date_mask.sum())
    if invalid_date_count:
        value_word = "value" if invalid_date_count == 1 else "values"
        errors.append(
            f"The date column has {invalid_date_count} missing or invalid {value_word} "
            f"in {describe_transaction_rows(invalid_date_mask)}. "
            "Use a valid date in YYYY-MM-DD format for every transaction."
        )
    prepared["date"] = converted_dates

    for column in NUMERIC_COLUMNS:
        original_values = prepared[column]
        missing_number_mask = original_values.isna() | (
            original_values.astype(str).str.strip() == ""
        )
        converted_numbers = pd.to_numeric(original_values, errors="coerce")
        nonnumeric_number_mask = converted_numbers.isna() & ~missing_number_mask

        missing_number_count = int(missing_number_mask.sum())
        if missing_number_count:
            value_word = "value" if missing_number_count == 1 else "values"
            errors.append(
                f"The {column} column is missing {missing_number_count} required "
                f"{value_word} in {describe_transaction_rows(missing_number_mask)}. "
                f"Add a {column} to every transaction, then upload the CSV again."
            )

        nonnumeric_number_count = int(nonnumeric_number_mask.sum())
        if nonnumeric_number_count:
            value_word = "value" if nonnumeric_number_count == 1 else "values"
            errors.append(
                f"The {column} column has {nonnumeric_number_count} non-numeric "
                f"{value_word} in {describe_transaction_rows(nonnumeric_number_mask)}. "
                f"Enter numbers only in the {column} column, then upload the CSV again."
            )
        prepared[column] = converted_numbers

    if errors:
        return prepared, errors

    try:
        calculate_fifo_holdings(prepared)
    except PortfolioCalculationError as calculation_error:
        calculation_message = str(calculation_error)
        if "sells more shares" in calculation_message:
            errors.append(
                f"The SELL quantities are not possible as entered. {calculation_message} "
                "Reduce the SELL quantity or add the missing earlier BUY transaction, "
                "then upload the CSV again."
            )
        else:
            errors.append(
                f"The transaction sequence cannot be calculated. {calculation_message} "
                "Check the transaction types and quantities, then upload the CSV again."
            )

    return prepared, errors


st.set_page_config(
    page_title="Stock Portfolio Analyser",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] .block-container {
            padding-top: 1.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stock Portfolio Analyser")
st.write(
    "Upload your transaction history to see what you own, evaluate buying and "
    "selling activity, and understand how your portfolio has performed."
)

if "portfolio_data" not in st.session_state:
    st.session_state.portfolio_data = None
if "portfolio_file_name" not in st.session_state:
    st.session_state.portfolio_file_name = None
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []
if "ai_portfolio_signature" not in st.session_state:
    st.session_state.ai_portfolio_signature = None
if "ai_historical_period_key" not in st.session_state:
    st.session_state.ai_historical_period_key = None
if "ai_historical_price_results" not in st.session_state:
    st.session_state.ai_historical_price_results = None


def show_upload_prompt_if_needed() -> None:
    """Guide the user to Data Upload only when no validated data is available."""
    if st.session_state.portfolio_data is None:
        st.info("Upload and validate a CSV in the Data Upload tab to use this section.")


(
    data_upload_tab,
    consolidated_portfolio_tab,
    historical_performance_tab,
    ai_analyst_tab,
) = st.tabs(
    [
        "Data Upload",
        "Consolidated Portfolio View",
        "Historical Performance",
        "AI Analyst (Chat)",
    ]
)

with data_upload_tab:
    st.subheader("Upload and validate")
    st.write(
        "Upload a correctly formatted CSV so your data shows correctly across "
        "all four tabs."
    )

    st.download_button(
        "Download blank CSV template",
        data=CSV_TEMPLATE,
        file_name="portfolio_transactions_template.csv",
        mime="text/csv",
    )

    with st.expander("Required CSV format"):
        st.write("Your CSV must include these five column headings:")
        displayed_columns = [
            "transaction_type*" if column == "transaction_type" else column
            for column in REQUIRED_COLUMNS
        ]
        st.code(",".join(displayed_columns), language=None)
        st.caption(r"\* transaction_type - Buy or Sell (enter BUY or SELL).")
        st.caption(
            "The asterisk is an explanation only. In your CSV, keep the actual "
            "column heading as transaction_type without the asterisk."
        )

    uploaded_file = st.file_uploader(
        "Upload your portfolio CSV file",
        type=["csv"],
        help="The file is used only during this app session.",
    )

    if uploaded_file is None:
        if st.session_state.portfolio_data is None:
            st.info("No CSV file has been uploaded yet. Choose a file above to begin.")
        else:
            st.info(
                f"{st.session_state.portfolio_file_name} remains available "
                "during this app session."
            )
    else:
        st.write("**Selected file:**", uploaded_file.name)

        try:
            with st.spinner("Checking the CSV format and transaction values..."):
                uploaded_data = load_csv(uploaded_file.getvalue())
                validated_data, validation_errors = validate_and_prepare(uploaded_data)
        except pd.errors.EmptyDataError:
            st.error(
                "The selected CSV is empty. Add the five required column headings "
                "and at least one transaction row, then upload it again."
            )
        except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError):
            st.error(
                "The selected file could not be read as a standard CSV. Save it as "
                "a comma-separated CSV with the required headings, then upload it again."
            )
        else:
            if validation_errors:
                st.error(
                    "The CSV did not pass validation. Fix the items below and upload it again."
                )
                st.markdown(
                    "\n".join(
                        f"- {validation_error}"
                        for validation_error in validation_errors
                    )
                )
                if st.session_state.portfolio_data is not None:
                    st.info(
                        "The last successfully validated CSV remains available "
                        "in this session."
                    )
            else:
                st.session_state.portfolio_data = validated_data
                st.session_state.portfolio_file_name = uploaded_file.name
                st.success("The CSV was loaded and validated successfully.")

                row_metric, column_metric = st.columns(2)
                row_metric.metric("Rows", len(validated_data))
                column_metric.metric("Columns", len(validated_data.columns))

                st.subheader("Data preview")
                preview_data = validated_data.copy()
                preview_data["date"] = preview_data["date"].dt.strftime("%Y-%m-%d")
                st.dataframe(preview_data, width="stretch", hide_index=True)

with consolidated_portfolio_tab:
    st.subheader("Consolidated Portfolio View")
    st.write(
        "See your current holdings and live estimated values, then filter and "
        "compare your BUY and SELL transactions."
    )

    portfolio_data = st.session_state.portfolio_data
    if portfolio_data is None:
        st.info(
            "Transaction filters, the filtered table and the activity chart "
            "will appear here after a CSV has been uploaded and validated."
        )
    else:
        st.subheader("Current holdings")

        try:
            fifo_holdings = calculate_fifo_holdings(portfolio_data)
        except PortfolioCalculationError as calculation_error:
            st.error(
                "The current holdings could not be calculated consistently. "
                f"{calculation_error}"
            )
            fifo_holdings = pd.DataFrame()

        if fifo_holdings.empty:
            st.info("The uploaded transactions do not leave any shares currently held.")
        else:
            held_tickers = tuple(fifo_holdings["ticker"].tolist())
            with st.spinner("Fetching the latest available market prices..."):
                latest_prices = load_market_prices(held_tickers)

            current_holdings = fifo_holdings.merge(
                latest_prices, on="ticker", how="left", validate="one_to_one"
            )
            current_holdings["market_value"] = (
                current_holdings["shares_held"]
                * current_holdings["current_price"]
            )
            current_holdings["unrealised_gain_or_loss"] = (
                current_holdings["market_value"]
                - current_holdings["remaining_cost_basis"]
            )
            current_holdings["allocation_percent"] = pd.NA

            missing_price_mask = current_holdings["current_price"].isna()
            all_prices_available = not missing_price_mask.any()

            if all_prices_available:
                total_market_value = current_holdings["market_value"].sum()
                total_unrealised_gain_or_loss = current_holdings[
                    "unrealised_gain_or_loss"
                ].sum()
                if total_market_value > 0:
                    current_holdings["allocation_percent"] = (
                        current_holdings["market_value"]
                        / total_market_value
                        * 100
                    )

                if total_unrealised_gain_or_loss > 0:
                    gain_loss_colour = "#2EAD64"
                elif total_unrealised_gain_or_loss < 0:
                    gain_loss_colour = "#E74C3C"
                else:
                    gain_loss_colour = "inherit"

                st.markdown(
                    f"""
                    <style>
                    .st-key-total_unrealised_metric
                    [data-testid="stMetricValue"] {{
                        color: {gain_loss_colour} !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                market_value_metric, gain_loss_metric = st.columns(2)
                market_value_metric.metric(
                    "Total current market value",
                    f"{total_market_value:,.2f}",
                )
                with gain_loss_metric:
                    with st.container(key="total_unrealised_metric"):
                        st.metric(
                            "Total unrealised gain or loss",
                            f"{total_unrealised_gain_or_loss:,.2f}",
                        )
            else:
                unavailable_tickers = ", ".join(
                    current_holdings.loc[missing_price_mask, "ticker"].tolist()
                )
                st.warning(
                    "Current price data is unavailable for "
                    f"{unavailable_tickers}. Shares and purchase costs are still "
                    "shown, but complete portfolio totals and allocation are "
                    "withheld so that a partial result is not misleading."
                )

            holdings_display = current_holdings[
                [
                    "ticker",
                    "shares_held",
                    "remaining_cost_basis",
                    "average_purchase_price",
                    "current_price",
                    "market_value",
                    "unrealised_gain_or_loss",
                ]
            ].copy()

            numeric_display_columns = [
                "shares_held",
                "remaining_cost_basis",
                "average_purchase_price",
                "current_price",
                "market_value",
                "unrealised_gain_or_loss",
            ]
            for column in numeric_display_columns:
                holdings_display[column] = holdings_display[column].map(
                    lambda value: (
                        "Not available"
                        if pd.isna(value)
                        else f"{float(value):,.2f}"
                    )
                )

            holdings_display.columns = [
                "Ticker",
                "Shares held",
                "Cost basis",
                "Average price",
                "Latest price",
                "Market value",
                "Unrealised gain/loss",
            ]
            st.dataframe(
                holdings_display,
                width="stretch",
                hide_index=True,
            )

            price_details = " · ".join(
                (
                    f"{row.ticker}: {float(row.current_price):,.2f} "
                    f"({row.price_status.lower()}, as of {row.price_as_of})"
                    if not pd.isna(row.current_price)
                    else f"{row.ticker}: {row.price_status.lower()}"
                )
                for row in current_holdings.itertuples(index=False)
            )
            st.caption(f"Yfinance prices used - {price_details}.")

            if all_prices_available and total_market_value > 0:
                st.subheader("Portfolio allocation by current market value")
                allocation_chart = current_holdings[
                    ["ticker", "allocation_percent", "market_value"]
                ].copy()
                allocation_chart["allocation_label"] = allocation_chart[
                    "allocation_percent"
                ].map(lambda value: f"{float(value):.1f}%")
                st.vega_lite_chart(
                    allocation_chart,
                    {
                        "layer": [
                            {
                                "mark": {"type": "arc", "outerRadius": 125},
                                "encoding": {
                                    "theta": {
                                        "field": "allocation_percent",
                                        "type": "quantitative",
                                        "stack": True,
                                    },
                                    "color": {
                                        "field": "ticker",
                                        "type": "nominal",
                                        "scale": {
                                            "range": [
                                                "#1F4E8C",
                                                "#6FA8DC",
                                                "#2EAD64",
                                            ]
                                        },
                                        "legend": {"title": "Ticker"},
                                    },
                                    "order": {
                                        "field": "ticker",
                                        "type": "nominal",
                                        "sort": "ascending",
                                    },
                                    "tooltip": [
                                        {
                                            "field": "ticker",
                                            "type": "nominal",
                                            "title": "Ticker",
                                        },
                                        {
                                            "field": "allocation_percent",
                                            "type": "quantitative",
                                            "title": "Allocation (%)",
                                            "format": ".1f",
                                        },
                                        {
                                            "field": "market_value",
                                            "type": "quantitative",
                                            "title": "Market value",
                                            "format": ",.2f",
                                        },
                                    ],
                                },
                            },
                            {
                                "mark": {
                                    "type": "text",
                                    "radius": 82,
                                    "fontSize": 14,
                                    "fontWeight": "bold",
                                    "fill": "white",
                                },
                                "encoding": {
                                    "theta": {
                                        "field": "allocation_percent",
                                        "type": "quantitative",
                                        "stack": True,
                                    },
                                    "text": {
                                        "field": "allocation_label",
                                        "type": "nominal",
                                    },
                                    "order": {
                                        "field": "ticker",
                                        "type": "nominal",
                                        "sort": "ascending",
                                    },
                                },
                            },
                        ]
                    },
                    width="stretch",
                    height=360,
                )

        st.divider()
        st.subheader("Explore transaction activity")
        st.write(
            "Use the filters below to explore the validated transactions. "
            "The table and chart update together."
        )

        ticker_values = portfolio_data["ticker"].astype(str)
        transaction_type_values = portfolio_data["transaction_type"].astype(str)
        ticker_options = sorted(ticker_values.unique().tolist())
        transaction_type_options = sorted(
            transaction_type_values.unique().tolist()
        )

        st.markdown(
            """
            <style>
            .st-key-consolidated_ticker_filter [data-tag] {
                background-color: #1F4E8C !important;
                color: #FFFFFF !important;
            }
            .st-key-consolidated_transaction_type_filter [data-tag] {
                background-color: #6FA8DC !important;
                color: #0B1F33 !important;
            }
            .st-key-consolidated_ticker_filter [data-tag] button,
            .st-key-consolidated_transaction_type_filter [data-tag] button {
                color: inherit !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        ticker_filter_column, type_filter_column = st.columns(2)
        with ticker_filter_column:
            selected_tickers = st.multiselect(
                "Ticker",
                options=ticker_options,
                default=ticker_options,
                key="consolidated_ticker_filter",
            )
        with type_filter_column:
            selected_transaction_types = st.multiselect(
                "Transaction type",
                options=transaction_type_options,
                default=transaction_type_options,
                key="consolidated_transaction_type_filter",
            )

        filtered_transactions = portfolio_data.loc[
            ticker_values.isin(selected_tickers)
            & transaction_type_values.isin(selected_transaction_types)
        ].copy()
        filtered_transactions = filtered_transactions.sort_values(
            ["date", "ticker"], kind="stable"
        )

        st.caption(
            f"Showing {len(filtered_transactions)} of "
            f"{len(portfolio_data)} validated transactions."
        )

        if filtered_transactions.empty:
            st.warning(
                "No transactions match these filters. Select at least one "
                "ticker and transaction type to show the table and chart."
            )
        else:
            st.subheader("Filtered transactions")
            display_transactions = filtered_transactions[
                REQUIRED_COLUMNS
            ].copy()
            display_transactions["date"] = display_transactions[
                "date"
            ].dt.strftime("%Y-%m-%d")
            display_transactions["quantity"] = display_transactions[
                "quantity"
            ].map(lambda value: f"{value:,.2f}")
            display_transactions["price"] = display_transactions["price"].map(
                lambda value: f"{value:,.2f}"
            )
            display_transactions.columns = [
                "Ticker",
                "Date",
                "Transaction type",
                "Quantity",
                "Price (uploaded currency)",
            ]
            st.dataframe(
                display_transactions,
                width="stretch",
                hide_index=True,
            )

            st.subheader("BUY versus SELL quantities")
            chart_data = (
                filtered_transactions.groupby(
                    ["ticker", "transaction_type"], sort=True
                )["quantity"]
                .sum()
                .unstack(fill_value=0)
            )
            chart_colors = [
                "#2EAD64" if transaction_type == "BUY" else "#E74C3C"
                for transaction_type in chart_data.columns
            ]
            st.bar_chart(
                chart_data,
                x_label="Ticker",
                y_label="Total quantity",
                color=chart_colors,
                stack=False,
                width="stretch",
                height=400,
            )
            st.caption(
                "This chart totals the transaction quantities currently "
                "included by the filters."
            )

with historical_performance_tab:
    st.subheader("Historical Performance")
    st.write(
        "Review cash invested, sale proceeds, current value, overall return, and "
        "your annualised money-weighted return (XIRR) in one place."
    )
    show_upload_prompt_if_needed()

    portfolio_data = st.session_state.portfolio_data
    if portfolio_data is not None:
        try:
            cash_activity = calculate_transaction_cash_flows(portfolio_data)
            performance_fifo_holdings = calculate_fifo_holdings(portfolio_data)
        except PortfolioCalculationError as calculation_error:
            st.error(
                "Historical performance could not be calculated consistently. "
                f"{calculation_error}"
            )
        else:
            current_market_value: float | None
            unavailable_price_tickers: list[str] = []
            performance_price_details: list[str] = []

            if performance_fifo_holdings.empty:
                current_market_value = 0.0
            else:
                performance_tickers = tuple(
                    performance_fifo_holdings["ticker"].tolist()
                )
                with st.spinner(
                    "Fetching the latest available prices for the performance summary..."
                ):
                    performance_prices = load_market_prices(performance_tickers)

                performance_holdings = performance_fifo_holdings.merge(
                    performance_prices,
                    on="ticker",
                    how="left",
                    validate="one_to_one",
                )
                missing_performance_price_mask = performance_holdings[
                    "current_price"
                ].isna()
                unavailable_price_tickers = performance_holdings.loc[
                    missing_performance_price_mask, "ticker"
                ].tolist()
                performance_price_details = [
                    (
                        f"{row.ticker}: {float(row.current_price):,.2f} "
                        f"({row.price_status.lower()}, as of {row.price_as_of})"
                        if not pd.isna(row.current_price)
                        else f"{row.ticker}: {row.price_status.lower()}"
                    )
                    for row in performance_holdings.itertuples(index=False)
                ]

                if unavailable_price_tickers:
                    current_market_value = None
                else:
                    current_market_value = float(
                        (
                            performance_holdings["shares_held"]
                            * performance_holdings["current_price"]
                        ).sum()
                    )

            performance_metrics = calculate_performance_metrics(
                cash_activity,
                current_market_value,
            )
            valuation_date = pd.Timestamp.today().normalize()

            xirr_value: float | None = None
            xirr_fallback_message: str | None = None
            if current_market_value is None:
                xirr_fallback_message = (
                    "Annualised return is not available because a complete ending market "
                    "value cannot be calculated without all current prices."
                )
            else:
                try:
                    xirr_value = calculate_xirr(
                        cash_activity,
                        current_market_value,
                        valuation_date,
                    )
                except PortfolioCalculationError as xirr_error:
                    xirr_fallback_message = str(xirr_error)

            def format_performance_amount(value: float | None) -> str:
                """Format a performance amount without implying a currency symbol."""
                if value is None or pd.isna(value):
                    return "Not available"
                return f"{float(value):,.2f}"

            st.subheader("Portfolio outcome summary")
            percentage_return = performance_metrics["percentage_return"]
            performance_summary_items = [
                [
                    {
                        "metric": "(A) Total purchase outflows",
                        "value": format_performance_amount(
                            performance_metrics["total_purchase_outflows"]
                        ),
                        "meaning": "Money you spent buying shares.",
                        "calculation": None,
                    },
                    {
                        "metric": "(B) Total sale proceeds",
                        "value": format_performance_amount(
                            performance_metrics["total_sale_proceeds"]
                        ),
                        "meaning": (
                            "Money you received from selling shares; this is not the same as profit."
                        ),
                        "calculation": None,
                    },
                    {
                        "metric": "(C) Net cash invested",
                        "value": format_performance_amount(
                            performance_metrics["net_cash_invested"]
                        ),
                        "meaning": (
                            "Your own money still invested after deducting money received from sales."
                        ),
                        "calculation": "A - B = C",
                    },
                ],
                [
                    {
                        "metric": "(D) Current holdings market value",
                        "value": format_performance_amount(
                            performance_metrics["current_market_value"]
                        ),
                        "meaning": (
                            "What the shares you still own are worth at the latest available prices."
                        ),
                        "calculation": None,
                    },
                    {
                        "metric": "(E) Estimated total current outcome",
                        "value": format_performance_amount(
                            performance_metrics["total_current_outcome"]
                        ),
                        "meaning": (
                            "Money received from sales plus the current value of shares you still own."
                        ),
                        "calculation": "B + D = E",
                    },
                    {
                        "metric": "(F) Absolute return",
                        "value": format_performance_amount(
                            performance_metrics["absolute_return"]
                        ),
                        "meaning": (
                            "How much your portfolio has gained or lost overall."
                        ),
                        "calculation": "E - A = F",
                    },
                ],
                [
                    {
                        "metric": "(G) Simple return",
                        "value": (
                            "Not available"
                            if percentage_return is None
                            else f"{float(percentage_return):,.2f}%"
                        ),
                        "meaning": (
                            "Your overall gain or loss as a percentage of the money spent buying shares."
                        ),
                        "calculation": "F / A x 100 = G",
                    },
                    {
                        "metric": "(H) Annualised return",
                        "value": (
                            "Not available"
                            if xirr_value is None
                            else f"{xirr_value * 100:,.2f}%"
                        ),
                        "meaning": (
                            "Your estimated yearly return after allowing for when you bought and sold shares."
                        ),
                        "calculation": None,
                    },
                    {
                        "metric": "(I) Valuation date",
                        "value": valuation_date.strftime("%Y-%m-%d"),
                        "meaning": (
                            "The date used to value the shares you still own and calculate the annualised return."
                        ),
                        "calculation": None,
                    },
                ],
            ]
            st.caption(
                "The letters show how the calculations connect to one another."
            )
            for performance_row_number, performance_row in enumerate(
                performance_summary_items
            ):
                performance_columns = st.columns(3)
                for performance_column, performance_item in zip(
                    performance_columns,
                    performance_row,
                ):
                    with performance_column:
                        card_height = 340 if performance_row_number == 2 else 280
                        with st.container(border=True, height=card_height):
                            st.metric(
                                performance_item["metric"],
                                performance_item["value"],
                            )
                            st.caption(performance_item["meaning"])
                            if performance_item["calculation"] is not None:
                                st.markdown(
                                    f"**Calculation:** {performance_item['calculation']}"
                                )

            if xirr_value is not None:
                with st.container(border=True):
                    st.markdown("**(H) Annualised return — How it was calculated**")
                    st.markdown(
                        "- **"
                        + format_performance_amount(
                            performance_metrics["total_purchase_outflows"]
                        )
                        + " went out in total**, spread across the individual BUY dates.\n"
                        + "- **"
                        + format_performance_amount(
                            performance_metrics["total_sale_proceeds"]
                        )
                        + " came back in total**, spread across the individual SELL dates.\n"
                        + "- The shares still owned were worth **"
                        + format_performance_amount(
                            performance_metrics["current_market_value"]
                        )
                        + "** on **"
                        + f"{valuation_date.day} "
                        + valuation_date.strftime("%B %Y")
                        + "**.\n"
                        + "- The app found the yearly growth rate that makes "
                        + "those dated amounts balance: **"
                        + f"{xirr_value * 100:,.2f}% per year**."
                    )

            st.caption(
                "Amounts are shown without a currency symbol because the uploaded "
                "CSV does not identify a currency. The app assumes current prices "
                "are directly comparable and performs no currency conversion."
            )
            if performance_price_details:
                st.caption(
                    "Yfinance ending prices used - "
                    + " · ".join(performance_price_details)
                    + "."
                )

            if unavailable_price_tickers:
                st.warning(
                    "Current prices are unavailable for "
                    f"{', '.join(unavailable_price_tickers)}. Purchase outflows, "
                    "sale proceeds and net cash invested are still shown, but "
                    "current outcome and return metrics are withheld so that a "
                    "partial result is not misleading."
                )
            if xirr_fallback_message is not None:
                st.info(xirr_fallback_message)

            st.subheader("Cumulative transaction cash activity")
            chart_columns = {
                "cumulative_purchase_outflows": "Purchase outflows",
                "cumulative_sale_proceeds": "Sale proceeds",
                "cumulative_net_cash_invested": "Net cash invested",
            }
            cash_activity_chart = cash_activity[
                ["date", *chart_columns.keys()]
            ].rename(columns=chart_columns)
            cash_activity_chart = cash_activity_chart.melt(
                id_vars="date",
                var_name="Cash activity",
                value_name="Amount",
            )
            st.vega_lite_chart(
                cash_activity_chart,
                {
                    "mark": {
                        "type": "line",
                        "point": True,
                        "strokeWidth": 3,
                    },
                    "encoding": {
                        "x": {
                            "field": "date",
                            "type": "temporal",
                            "title": "Transaction date",
                        },
                        "y": {
                            "field": "Amount",
                            "type": "quantitative",
                            "title": "Cumulative amount",
                        },
                        "color": {
                            "field": "Cash activity",
                            "type": "nominal",
                            "scale": {
                                "domain": [
                                    "Purchase outflows",
                                    "Sale proceeds",
                                    "Net cash invested",
                                ],
                                "range": ["#1F4E8C", "#2EAD64", "#6FA8DC"],
                            },
                            "legend": {"title": None},
                        },
                        "tooltip": [
                            {
                                "field": "date",
                                "type": "temporal",
                                "title": "Date",
                                "format": "%Y-%m-%d",
                            },
                            {
                                "field": "Cash activity",
                                "type": "nominal",
                                "title": "Series",
                            },
                            {
                                "field": "Amount",
                                "type": "quantitative",
                                "title": "Amount",
                                "format": ",.2f",
                            },
                        ],
                    },
                },
                width="stretch",
                height=400,
            )
            st.caption(
                "This chart uses only the dated BUY and SELL transactions in "
                "the uploaded CSV. It does not represent historical portfolio "
                "market values, because those values are not available in the data."
            )

            st.subheader("Assumptions")
            st.markdown(
                """
- All transaction amounts and current prices are assumed to be comparable in one currency; no foreign-exchange conversion is performed.
- Brokerage fees, taxes, dividends, interest and cash movements not present in the uploaded CSV are excluded.
- Sale proceeds are counted as value returned by the portfolio. If they were later reinvested, the later BUY is included as another purchase outflow.
- Current holdings use the latest available yfinance prices and the valuation date shown above. Annualised return is withheld when the ending value or a reliable result is unavailable.
                """
            )

with ai_analyst_tab:
    st.subheader("AI Analyst (Chat)")
    st.write(
        "Ask our friendly AI analyst plain-language questions about your "
        "portfolio data."
    )
    show_upload_prompt_if_needed()
    st.caption(
        "The AI Analyst uses verified portfolio facts for educational analysis, "
        "not financial advice. It cannot execute trades or tell you personally "
        "what to buy or sell."
    )
    st.markdown(
        """
        <style>
        [data-testid="stChatMessageAvatarUser"] {
            background-color: #D8B4E2 !important;
            color: #3F2450 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    portfolio_data = st.session_state.portfolio_data
    if portfolio_data is not None:
        current_signature = portfolio_signature(portfolio_data)
        if st.session_state.ai_portfolio_signature != current_signature:
            st.session_state.ai_chat_history = []
            st.session_state.ai_historical_period_key = None
            st.session_state.ai_historical_price_results = None
            st.session_state.ai_portfolio_signature = current_signature

        try:
            ai_fifo_holdings = calculate_fifo_holdings(portfolio_data)
            ai_tickers: tuple[str, ...] = tuple()
            if ai_fifo_holdings.empty:
                ai_latest_prices = pd.DataFrame(
                    columns=[
                        "ticker",
                        "current_price",
                        "price_as_of",
                        "price_status",
                        "retrieved_at",
                    ]
                )
            else:
                ai_tickers = tuple(ai_fifo_holdings["ticker"].tolist())
                with st.spinner(
                    "Preparing the latest verified portfolio facts for the AI Analyst..."
                ):
                    ai_latest_prices = load_market_prices(ai_tickers)
            ai_portfolio_facts = build_portfolio_facts(
                portfolio_data,
                ai_latest_prices,
                pd.Timestamp.today().normalize(),
            )
        except (PortfolioCalculationError, KeyError, TypeError, ValueError):
            st.error(
                "The AI Analyst could not prepare a reliable portfolio summary. "
                "The other tabs remain available, and no AI request was made."
            )
        else:
            st.markdown("**Suggested portfolio questions**")
            st.markdown(
                "- Which stock makes up the largest part of my current portfolio?\n"
                "- Am I heavily concentrated in one holding?\n"
                "- How far is each holding from an equal one-third allocation?\n"
                "- Which current holding has the largest unrealised gain or loss?\n"
                "- What do my portfolio figures show about overall performance?\n"
                "- After selecting a period below, which stock performed best?"
            )

            st.markdown("**Historical market comparison**")
            st.caption(
                "Choose a period to fetch verified yfinance adjusted daily closing "
                "prices and compare the current holdings. This measures market "
                "performance, not your transaction-timed portfolio return."
            )
            highlighted_historical_period = (
                st.session_state.ai_historical_period_key
            )
            if highlighted_historical_period in HISTORICAL_PRICE_PERIODS:
                st.markdown(
                    f"""
                    <style>
                    .st-key-ai_historical_period_{highlighted_historical_period} button {{
                        background-color: #8B5CF6 !important;
                        border-color: #C4B5FD !important;
                        color: #FFFFFF !important;
                        font-weight: 700 !important;
                        box-shadow: 0 0 0 2px rgba(196, 181, 253, 0.35) !important;
                    }}
                    .st-key-ai_historical_period_{highlighted_historical_period} button:hover {{
                        background-color: #7C3AED !important;
                        border-color: #DDD6FE !important;
                        color: #FFFFFF !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
            historical_button_columns = st.columns(
                len(HISTORICAL_PRICE_PERIODS)
            )
            selected_historical_period_key = None
            for historical_button_column, (
                period_key,
                period_label,
            ) in zip(
                historical_button_columns,
                HISTORICAL_PRICE_PERIODS.items(),
            ):
                with historical_button_column:
                    button_label = (
                        f"✓ {period_label}"
                        if period_key == highlighted_historical_period
                        else period_label
                    )
                    if st.button(
                        button_label,
                        key=f"ai_historical_period_{period_key}",
                        disabled=not ai_tickers,
                        use_container_width=True,
                    ):
                        selected_historical_period_key = period_key

            if selected_historical_period_key is not None:
                with st.spinner(
                    "Fetching and validating the selected yfinance history..."
                ):
                    st.session_state.ai_historical_price_results = (
                        load_historical_price_performance(
                            ai_tickers,
                            selected_historical_period_key,
                        )
                    )
                st.session_state.ai_historical_period_key = (
                    selected_historical_period_key
                )
                st.rerun()

            historical_price_results = (
                st.session_state.ai_historical_price_results
            )
            if isinstance(historical_price_results, pd.DataFrame):
                historical_price_facts = build_historical_price_facts(
                    historical_price_results
                )
                ai_portfolio_facts["historical_market_comparison"] = (
                    historical_price_facts
                )
                st.markdown(
                    format_historical_price_response(historical_price_facts)
                )

            current_portfolio_facts = ai_portfolio_facts["current_portfolio"]
            performance_facts = ai_portfolio_facts["performance"]
            data_facts = ai_portfolio_facts["data_summary"]

            def show_verified_ai_fallback() -> None:
                """Show useful Python-calculated facts without relying on Groq."""
                st.markdown("**Verified portfolio facts available without AI**")
                transaction_metric, holdings_metric, tickers_metric = st.columns(3)
                transaction_metric.metric(
                    "Validated transactions",
                    f"{data_facts['validated_transaction_count']}",
                )
                holdings_metric.metric(
                    "Current holdings",
                    f"{current_portfolio_facts['holding_count']}",
                )
                tickers_metric.metric(
                    "Tickers in the CSV",
                    f"{data_facts['ticker_count']}",
                )

                largest_holding = current_portfolio_facts[
                    "largest_holding_by_current_market_value"
                ]
                if current_portfolio_facts["all_current_prices_available"]:
                    market_value_metric, gain_loss_metric, largest_metric = st.columns(3)
                    market_value_metric.metric(
                        "Current market value",
                        f"{current_portfolio_facts['total_current_market_value']:,.2f}",
                    )
                    gain_loss_metric.metric(
                        "Unrealised gain or loss",
                        f"{current_portfolio_facts['total_unrealised_gain_or_loss']:,.2f}",
                    )
                    if largest_holding is not None:
                        largest_metric.metric(
                            "Largest holding",
                            str(largest_holding["ticker"]),
                            f"{largest_holding['allocation_percent']:,.2f}% of current value",
                        )
                else:
                    st.info(
                        "Complete current-value and allocation facts are withheld "
                        "because one or more current prices are unavailable."
                    )

                with st.expander("See the verified current holdings used by the AI"):
                    holdings_rows = current_portfolio_facts["holdings"]
                    if holdings_rows:
                        holdings_fallback_display = pd.DataFrame(holdings_rows)[
                            [
                                "ticker",
                                "shares_held",
                                "remaining_fifo_cost_basis",
                                "average_purchase_price",
                                "current_price",
                                "current_market_value",
                                "unrealised_gain_or_loss",
                                "allocation_percent",
                                "price_source_status",
                                "price_as_of",
                            ]
                        ].rename(
                            columns={
                                "ticker": "Ticker",
                                "shares_held": "Shares held",
                                "remaining_fifo_cost_basis": "Remaining FIFO cost basis",
                                "average_purchase_price": "Average purchase price",
                                "current_price": "Current price",
                                "current_market_value": "Current market value",
                                "unrealised_gain_or_loss": "Unrealised gain or loss",
                                "allocation_percent": "Allocation (%)",
                                "price_source_status": "Price source",
                                "price_as_of": "Price as of",
                            }
                        )
                        st.dataframe(
                            holdings_fallback_display,
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "Shares held": st.column_config.NumberColumn(format="%.4f"),
                                "Remaining FIFO cost basis": st.column_config.NumberColumn(format="%.2f"),
                                "Average purchase price": st.column_config.NumberColumn(format="%.2f"),
                                "Current price": st.column_config.NumberColumn(format="%.2f"),
                                "Current market value": st.column_config.NumberColumn(format="%.2f"),
                                "Unrealised gain or loss": st.column_config.NumberColumn(format="%.2f"),
                                "Allocation (%)": st.column_config.NumberColumn(format="%.2f%%"),
                            },
                        )
                    else:
                        st.info(
                            "The uploaded transactions do not leave any shares currently held."
                        )
                st.caption(
                    "Amounts have no currency symbol because the uploaded CSV does "
                    "not identify a currency."
                )

            secret_read_error = False
            try:
                groq_api_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
            except Exception:
                groq_api_key = ""
                secret_read_error = True

            clear_chat_column, chat_status_column = st.columns([1, 4])
            with clear_chat_column:
                if st.button(
                    "Clear chat",
                    disabled=not st.session_state.ai_chat_history,
                    use_container_width=True,
                ):
                    st.session_state.ai_chat_history = []
                    st.rerun()
            with chat_status_column:
                if groq_api_key:
                    st.success("Our friendly AI is ready for you! Ask away!")

            for chat_message in st.session_state.ai_chat_history:
                with st.chat_message(chat_message["role"]):
                    st.markdown(chat_message["content"])

            if not groq_api_key:
                if secret_read_error:
                    st.warning(
                        "The local Streamlit secrets file could not be read. The AI "
                        "chat is unavailable, but the verified portfolio facts below "
                        "and the other tabs still work."
                    )
                else:
                    st.info(
                        "The Groq API key is not configured yet. Add it privately "
                        "through the approved local Streamlit secrets file, then "
                        "restart the app. Never paste the key into this chat."
                    )
                show_verified_ai_fallback()
                st.chat_input(
                    "Configure the private Groq secret to ask a portfolio question",
                    disabled=True,
                )
            else:
                question = st.chat_input("Ask a question about your portfolio")
                if question:
                    st.session_state.ai_chat_history.append(
                        {"role": "user", "content": question}
                    )
                    with st.chat_message("user"):
                        st.markdown(question)

                    existing_history = st.session_state.ai_chat_history[:-1]
                    relevant_transactions = select_relevant_transactions(
                        portfolio_data,
                        question,
                    )
                    deterministic_response = local_safety_response(
                        question,
                        ai_portfolio_facts,
                    )
                    with st.chat_message("assistant"):
                        if deterministic_response is not None:
                            answer = deterministic_response
                            provider_succeeded = False
                        else:
                            with st.spinner("Analysing the verified portfolio facts..."):
                                answer, provider_succeeded = request_groq_analysis(
                                    groq_api_key,
                                    question,
                                    ai_portfolio_facts,
                                    relevant_transactions,
                                    existing_history,
                                )
                        st.markdown(answer)
                        if provider_succeeded:
                            st.caption(
                                "Grounded in the verified Python-calculated portfolio "
                                "facts and relevant validated transaction rows."
                            )
                    st.session_state.ai_chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()

                with st.expander("Verified portfolio facts used by the AI"):
                    show_verified_ai_fallback()

