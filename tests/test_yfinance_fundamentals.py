"""Tests for the fundamentals-related yfinance functions in y_finance.py.

Only ``get_YFin_data_online`` (OHLCV) had test coverage; the five
fundamentals functions below (~40% of the file) had none — their parsing/
formatting logic was only reachable indirectly through mocked
``interface.route_to_vendor`` calls elsewhere, which never actually exercise
the real yfinance-shaped data.
"""

from unittest import mock

import pandas as pd
import pytest

import tradingagents.dataflows.y_finance as y_finance
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _patched_ticker(fake_ticker_cls):
    return mock.patch.object(y_finance.yf, "Ticker", fake_ticker_cls)


@pytest.mark.unit
class TestGetFundamentals:
    def test_happy_path_includes_only_non_none_fields(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.info = {
                    "longName": "Apple Inc.",
                    "sector": "Technology",
                    "trailingPE": 30.5,
                    "forwardPE": None,  # must be excluded
                }

        with _patched_ticker(FakeTicker):
            result = y_finance.get_fundamentals("AAPL")

        assert "Name: Apple Inc." in result
        assert "Sector: Technology" in result
        assert "PE Ratio (TTM): 30.5" in result
        assert "Forward PE" not in result

    def test_empty_info_raises_no_market_data_error(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.info = {}

        with _patched_ticker(FakeTicker), pytest.raises(NoMarketDataError):
            y_finance.get_fundamentals("FAKE")

    def test_truthy_but_all_none_stub_raises_no_market_data_error(self):
        # yfinance returns a stub dict for unknown symbols: truthy, but every
        # whitelisted field is None.
        class FakeTicker:
            def __init__(self, symbol):
                self.info = {"trailingPegRatio": None}

        with _patched_ticker(FakeTicker), pytest.raises(NoMarketDataError):
            y_finance.get_fundamentals("UNKNOWNXYZ")

    def test_generic_exception_returns_error_string(self):
        class FakeTicker:
            def __init__(self, symbol):
                raise RuntimeError("network exploded")

        with _patched_ticker(FakeTicker):
            result = y_finance.get_fundamentals("AAPL")

        assert "Error retrieving fundamentals for AAPL" in result


def _financial_frame(columns):
    return pd.DataFrame({col: [1.0, 2.0] for col in columns}, index=["Total Assets", "Total Debt"])


@pytest.mark.unit
class TestGetBalanceSheet:
    def test_quarterly_uses_quarterly_attribute(self):
        quarterly = _financial_frame(["2024-09-30"])
        annual = _financial_frame(["2024-12-31"])

        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_balance_sheet = quarterly
                self.balance_sheet = annual

        with _patched_ticker(FakeTicker):
            result = y_finance.get_balance_sheet("AAPL", freq="quarterly")

        assert "2024-09-30" in result
        assert "Balance Sheet data for AAPL (quarterly)" in result

    def test_annual_uses_annual_attribute(self):
        quarterly = _financial_frame(["2024-09-30"])
        annual = _financial_frame(["2024-12-31"])

        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_balance_sheet = quarterly
                self.balance_sheet = annual

        with _patched_ticker(FakeTicker):
            result = y_finance.get_balance_sheet("AAPL", freq="annual")

        assert "2024-12-31" in result
        assert "2024-09-30" not in result

    def test_empty_frame_raises_no_market_data_error(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_balance_sheet = pd.DataFrame()
                self.balance_sheet = pd.DataFrame()

        with _patched_ticker(FakeTicker), pytest.raises(NoMarketDataError):
            y_finance.get_balance_sheet("AAPL", freq="quarterly")

    def test_future_dated_column_filtered_by_curr_date(self):
        frame = _financial_frame(["2024-09-30", "2030-01-01"])

        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_balance_sheet = frame

        with _patched_ticker(FakeTicker):
            result = y_finance.get_balance_sheet(
                "AAPL", freq="quarterly", curr_date="2025-01-01"
            )

        assert "2024-09-30" in result
        assert "2030-01-01" not in result

    def test_generic_exception_returns_error_string(self):
        class FakeTicker:
            def __init__(self, symbol):
                raise RuntimeError("boom")

        with _patched_ticker(FakeTicker):
            result = y_finance.get_balance_sheet("AAPL")

        assert "Error retrieving balance sheet for AAPL" in result


@pytest.mark.unit
class TestGetCashflow:
    def test_happy_path(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_cashflow = _financial_frame(["2024-09-30"])

        with _patched_ticker(FakeTicker):
            result = y_finance.get_cashflow("AAPL", freq="quarterly")

        assert "Cash Flow data for AAPL (quarterly)" in result
        assert "2024-09-30" in result

    def test_empty_frame_raises_no_market_data_error(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_cashflow = pd.DataFrame()

        with _patched_ticker(FakeTicker), pytest.raises(NoMarketDataError):
            y_finance.get_cashflow("AAPL", freq="quarterly")

    def test_generic_exception_returns_error_string(self):
        class FakeTicker:
            def __init__(self, symbol):
                raise RuntimeError("boom")

        with _patched_ticker(FakeTicker):
            result = y_finance.get_cashflow("AAPL")

        assert "Error retrieving cash flow for AAPL" in result


@pytest.mark.unit
class TestGetIncomeStatement:
    def test_happy_path(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_income_stmt = _financial_frame(["2024-09-30"])

        with _patched_ticker(FakeTicker):
            result = y_finance.get_income_statement("AAPL", freq="quarterly")

        assert "Income Statement data for AAPL (quarterly)" in result
        assert "2024-09-30" in result

    def test_empty_frame_raises_no_market_data_error(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_income_stmt = pd.DataFrame()

        with _patched_ticker(FakeTicker), pytest.raises(NoMarketDataError):
            y_finance.get_income_statement("AAPL", freq="quarterly")

    def test_generic_exception_returns_error_string(self):
        class FakeTicker:
            def __init__(self, symbol):
                raise RuntimeError("boom")

        with _patched_ticker(FakeTicker):
            result = y_finance.get_income_statement("AAPL")

        assert "Error retrieving income statement for AAPL" in result


@pytest.mark.unit
class TestGetInsiderTransactions:
    def test_happy_path(self):
        frame = pd.DataFrame({"Shares": [100, 200]}, index=["Insider A", "Insider B"])

        class FakeTicker:
            def __init__(self, symbol):
                self.insider_transactions = frame

        with _patched_ticker(FakeTicker):
            result = y_finance.get_insider_transactions("AAPL")

        assert "Insider Transactions data for AAPL" in result
        assert "100" in result

    def test_none_returns_plain_message_not_exception(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.insider_transactions = None

        with _patched_ticker(FakeTicker):
            result = y_finance.get_insider_transactions("AAPL")

        assert result == "No insider transactions reported for symbol 'AAPL'"

    def test_empty_dataframe_returns_plain_message_not_exception(self):
        class FakeTicker:
            def __init__(self, symbol):
                self.insider_transactions = pd.DataFrame()

        with _patched_ticker(FakeTicker):
            result = y_finance.get_insider_transactions("AAPL")

        assert result == "No insider transactions reported for symbol 'AAPL'"

    def test_generic_exception_returns_error_string(self):
        class FakeTicker:
            def __init__(self, symbol):
                raise RuntimeError("boom")

        with _patched_ticker(FakeTicker):
            result = y_finance.get_insider_transactions("AAPL")

        assert "Error retrieving insider transactions for AAPL" in result
