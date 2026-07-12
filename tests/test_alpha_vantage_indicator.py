"""Tests for alpha_vantage_indicator.get_indicator's CSV parsing.

Alpha Vantage is the fallback technical-indicator vendor when yfinance
fails; existing Alpha Vantage tests (test_alpha_vantage_hardening.py) only
cover request transport/rate-limiting, not this file's response-parsing
logic. A bug in the column-name mapping or date filtering here would only
surface in production, after yfinance has already failed.
"""

from unittest.mock import patch

import pytest

from tradingagents.dataflows.alpha_vantage_common import AlphaVantageNotConfiguredError
from tradingagents.dataflows.alpha_vantage_indicator import get_indicator

TARGET = "tradingagents.dataflows.alpha_vantage_indicator._make_api_request"


@pytest.mark.unit
class TestSupportedIndicators:
    def test_sma_extracts_correct_column_and_dates_in_range(self):
        csv = (
            "time,SMA\n"
            "2024-10-15,150.0\n"
            "2024-10-20,151.5\n"
            "2024-09-01,140.0\n"  # outside look_back window, must be excluded
        )
        with patch(TARGET, return_value=csv) as mock_request:
            result = get_indicator("AAPL", "close_50_sma", "2024-10-25", look_back_days=30)

        args, kwargs = mock_request.call_args
        assert args[0] == "SMA"
        assert args[1]["time_period"] == "50"

        assert "2024-10-15: 150.0" in result
        assert "2024-10-20: 151.5" in result
        assert "2024-09-01" not in result

    def test_rsi_extracts_rsi_column(self):
        csv = "time,RSI\n2024-10-15,65.2\n"
        with patch(TARGET, return_value=csv) as mock_request:
            result = get_indicator("AAPL", "rsi", "2024-10-25", look_back_days=30)

        args, kwargs = mock_request.call_args
        assert args[0] == "RSI"
        assert "2024-10-15: 65.2" in result

    def test_macd_line_extracts_macd_column(self):
        csv = "time,MACD,MACD_Signal,MACD_Hist\n2024-10-15,1.2,1.0,0.2\n"
        with patch(TARGET, return_value=csv):
            result = get_indicator("AAPL", "macd", "2024-10-25", look_back_days=30)

        assert "2024-10-15: 1.2" in result

    def test_macd_signal_extracts_signal_column_not_macd_line(self):
        csv = "time,MACD,MACD_Signal,MACD_Hist\n2024-10-15,1.2,1.0,0.2\n"
        with patch(TARGET, return_value=csv):
            result = get_indicator("AAPL", "macds", "2024-10-25", look_back_days=30)

        assert "2024-10-15: 1.0" in result
        assert "2024-10-15: 1.2" not in result

    def test_macd_histogram_extracts_hist_column(self):
        csv = "time,MACD,MACD_Signal,MACD_Hist\n2024-10-15,1.2,1.0,0.2\n"
        with patch(TARGET, return_value=csv):
            result = get_indicator("AAPL", "macdh", "2024-10-25", look_back_days=30)

        assert "2024-10-15: 0.2" in result

    def test_bollinger_bands_map_to_correct_columns(self):
        csv = (
            "time,Real Upper Band,Real Middle Band,Real Lower Band\n"
            "2024-10-15,110.0,100.0,90.0\n"
        )
        with patch(TARGET, return_value=csv):
            assert "2024-10-15: 100.0" in get_indicator("AAPL", "boll", "2024-10-25", 30)
        with patch(TARGET, return_value=csv):
            assert "2024-10-15: 110.0" in get_indicator("AAPL", "boll_ub", "2024-10-25", 30)
        with patch(TARGET, return_value=csv):
            assert "2024-10-15: 90.0" in get_indicator("AAPL", "boll_lb", "2024-10-25", 30)

    def test_vwma_returns_informative_message_without_api_call(self):
        with patch(TARGET) as mock_request:
            result = get_indicator("AAPL", "vwma", "2024-10-25", look_back_days=30)

        mock_request.assert_not_called()
        assert "VWMA" in result


@pytest.mark.unit
class TestErrorAndEdgeCases:
    def test_unsupported_indicator_raises_value_error(self):
        with pytest.raises(ValueError, match="not supported"):
            get_indicator("AAPL", "not_a_real_indicator", "2024-10-25", look_back_days=30)

    def test_empty_response_returns_error_string(self):
        with patch(TARGET, return_value=""):
            result = get_indicator("AAPL", "rsi", "2024-10-25", look_back_days=30)
        assert "Error: No data returned" in result

    def test_no_rows_within_date_range_returns_placeholder(self):
        csv = "time,RSI\n2020-01-01,50.0\n"  # far outside the look-back window
        with patch(TARGET, return_value=csv):
            result = get_indicator("AAPL", "rsi", "2024-10-25", look_back_days=30)
        assert "No data available for the specified date range." in result

    def test_missing_time_column_returns_error_string(self):
        csv = "date,RSI\n2024-10-15,65.2\n"
        with patch(TARGET, return_value=csv):
            result = get_indicator("AAPL", "rsi", "2024-10-25", look_back_days=30)
        assert "'time' column not found" in result

    def test_not_configured_error_propagates(self):
        with (
            patch(TARGET, side_effect=AlphaVantageNotConfiguredError("no key")),
            pytest.raises(AlphaVantageNotConfiguredError),
        ):
            get_indicator("AAPL", "rsi", "2024-10-25", look_back_days=30)

    def test_unexpected_exception_returns_error_string_not_raised(self):
        with patch(TARGET, side_effect=RuntimeError("network exploded")):
            result = get_indicator("AAPL", "rsi", "2024-10-25", look_back_days=30)
        assert "Error retrieving rsi data" in result
