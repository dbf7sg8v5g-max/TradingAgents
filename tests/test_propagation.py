"""Tests for Propagator — initial state construction and graph invocation args.

``get_graph_args`` (recursion limit, callback wiring) had no tests; only
``create_initial_state``'s ``asset_type`` field was spot-checked elsewhere.
"""

import pytest

from tradingagents.graph.propagation import Propagator


@pytest.mark.unit
class TestGetGraphArgs:
    def test_default_recursion_limit_and_stream_mode(self):
        propagator = Propagator(max_recur_limit=100)
        args = propagator.get_graph_args()

        assert args["stream_mode"] == "values"
        assert args["config"]["recursion_limit"] == 100
        assert "callbacks" not in args["config"]

    def test_custom_recursion_limit_is_used(self):
        propagator = Propagator(max_recur_limit=42)
        args = propagator.get_graph_args()

        assert args["config"]["recursion_limit"] == 42

    def test_callbacks_added_when_provided(self):
        propagator = Propagator(max_recur_limit=100)
        callbacks = [object()]

        args = propagator.get_graph_args(callbacks=callbacks)

        assert args["config"]["callbacks"] == callbacks

    def test_callbacks_omitted_when_none(self):
        propagator = Propagator(max_recur_limit=100)

        args = propagator.get_graph_args(callbacks=None)

        assert "callbacks" not in args["config"]

    def test_callbacks_omitted_when_empty_list(self):
        propagator = Propagator(max_recur_limit=100)

        args = propagator.get_graph_args(callbacks=[])

        assert "callbacks" not in args["config"]


@pytest.mark.unit
class TestCreateInitialState:
    def test_returns_full_expected_key_set(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2024-10-25")

        expected_keys = {
            "messages",
            "company_of_interest",
            "asset_type",
            "instrument_context",
            "trade_date",
            "past_context",
            "investment_debate_state",
            "risk_debate_state",
            "market_report",
            "fundamentals_report",
            "sentiment_report",
            "news_report",
        }
        assert expected_keys.issubset(state.keys())

    def test_defaults_and_basic_fields(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2024-10-25")

        assert state["company_of_interest"] == "AAPL"
        assert state["trade_date"] == "2024-10-25"
        assert state["asset_type"] == "stock"
        assert state["instrument_context"] == ""
        assert state["past_context"] == ""
        assert state["messages"] == [("human", "AAPL")]

    def test_trade_date_is_coerced_to_string(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", 20241025)

        assert state["trade_date"] == "20241025"

    def test_non_default_asset_type_and_contexts_are_threaded_through(self):
        propagator = Propagator()
        state = propagator.create_initial_state(
            "BTC-USD",
            "2024-10-25",
            asset_type="crypto",
            past_context="prior lesson",
            instrument_context="BTC-USD is a crypto asset.",
        )

        assert state["asset_type"] == "crypto"
        assert state["past_context"] == "prior lesson"
        assert state["instrument_context"] == "BTC-USD is a crypto asset."

    def test_investment_debate_state_is_zeroed(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2024-10-25")
        debate_state = state["investment_debate_state"]

        assert debate_state["count"] == 0
        assert debate_state["history"] == ""
        assert debate_state["bull_history"] == ""
        assert debate_state["bear_history"] == ""
        assert debate_state["current_response"] == ""
        assert debate_state["judge_decision"] == ""

    def test_risk_debate_state_is_zeroed(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2024-10-25")
        risk_state = state["risk_debate_state"]

        assert risk_state["count"] == 0
        assert risk_state["history"] == ""
        assert risk_state["latest_speaker"] == ""
        assert risk_state["aggressive_history"] == ""
        assert risk_state["conservative_history"] == ""
        assert risk_state["neutral_history"] == ""
        assert risk_state["current_aggressive_response"] == ""
        assert risk_state["current_conservative_response"] == ""
        assert risk_state["current_neutral_response"] == ""
        assert risk_state["judge_decision"] == ""
