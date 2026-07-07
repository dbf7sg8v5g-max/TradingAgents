"""Tests for the bull/bear researcher and risk-debator state mutations.

These five agent factories are the producing end of the same state contract
``conditional_logic.py`` reads to route the debate/risk-discussion loops
(history concatenation, ``count`` increment, ``current_response``/
``latest_speaker`` updates). A dropped sibling field or wrong key here would
silently reset debate context or desync routing, with nothing else in the
suite to catch it since no test previously exercised these node functions.
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator


def _mock_llm(reply_text):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=reply_text)
    return llm


def _base_state(**overrides):
    state = {
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "market_report": "market report",
        "sentiment_report": "sentiment report",
        "news_report": "news report",
        "fundamentals_report": "fundamentals report",
    }
    state.update(overrides)
    return state


def _invest_debate_state(**overrides):
    state = {
        "history": "prior history",
        "bull_history": "prior bull",
        "bear_history": "prior bear",
        "current_response": "",
        "judge_decision": "",
        "count": 2,
    }
    state.update(overrides)
    return state


def _risk_debate_state(**overrides):
    state = {
        "history": "prior history",
        "aggressive_history": "prior aggressive",
        "conservative_history": "prior conservative",
        "neutral_history": "prior neutral",
        "latest_speaker": "",
        "current_aggressive_response": "aggressive says X",
        "current_conservative_response": "conservative says Y",
        "current_neutral_response": "neutral says Z",
        "judge_decision": "",
        "count": 2,
    }
    state.update(overrides)
    return state


@pytest.mark.unit
class TestBullResearcher:
    def test_state_mutation(self):
        llm = _mock_llm("Strong growth ahead.")
        state = _base_state(
            investment_debate_state=_invest_debate_state(current_response="Bear Analyst: watch out"),
        )
        node = create_bull_researcher(llm)

        result = node(state)["investment_debate_state"]

        assert result["history"] == "prior history\nBull Analyst: Strong growth ahead."
        assert result["bull_history"] == "prior bull\nBull Analyst: Strong growth ahead."
        assert result["bear_history"] == "prior bear"  # sibling history preserved
        assert result["current_response"] == "Bull Analyst: Strong growth ahead."
        assert result["count"] == 3


@pytest.mark.unit
class TestBearResearcher:
    def test_state_mutation(self):
        llm = _mock_llm("Too much risk here.")
        state = _base_state(
            investment_debate_state=_invest_debate_state(current_response="Bull Analyst: buy now"),
        )
        node = create_bear_researcher(llm)

        result = node(state)["investment_debate_state"]

        assert result["history"] == "prior history\nBear Analyst: Too much risk here."
        assert result["bear_history"] == "prior bear\nBear Analyst: Too much risk here."
        assert result["bull_history"] == "prior bull"  # sibling history preserved
        assert result["current_response"] == "Bear Analyst: Too much risk here."
        assert result["count"] == 3


@pytest.mark.unit
class TestAggressiveDebator:
    def test_state_mutation(self):
        llm = _mock_llm("Go big.")
        state = _base_state(
            trader_investment_plan="Buy 100 shares",
            risk_debate_state=_risk_debate_state(),
        )
        node = create_aggressive_debator(llm)

        result = node(state)["risk_debate_state"]

        assert result["history"] == "prior history\nAggressive Analyst: Go big."
        assert result["aggressive_history"] == "prior aggressive\nAggressive Analyst: Go big."
        assert result["latest_speaker"] == "Aggressive"
        assert result["current_aggressive_response"] == "Aggressive Analyst: Go big."
        # sibling fields preserved unchanged
        assert result["conservative_history"] == "prior conservative"
        assert result["neutral_history"] == "prior neutral"
        assert result["current_conservative_response"] == "conservative says Y"
        assert result["current_neutral_response"] == "neutral says Z"
        assert result["count"] == 3


@pytest.mark.unit
class TestConservativeDebator:
    def test_state_mutation(self):
        llm = _mock_llm("Too risky, scale back.")
        state = _base_state(
            trader_investment_plan="Buy 100 shares",
            risk_debate_state=_risk_debate_state(),
        )
        node = create_conservative_debator(llm)

        result = node(state)["risk_debate_state"]

        assert result["history"] == "prior history\nConservative Analyst: Too risky, scale back."
        assert (
            result["conservative_history"]
            == "prior conservative\nConservative Analyst: Too risky, scale back."
        )
        assert result["latest_speaker"] == "Conservative"
        assert result["current_conservative_response"] == "Conservative Analyst: Too risky, scale back."
        # sibling fields preserved unchanged
        assert result["aggressive_history"] == "prior aggressive"
        assert result["neutral_history"] == "prior neutral"
        assert result["current_aggressive_response"] == "aggressive says X"
        assert result["current_neutral_response"] == "neutral says Z"
        assert result["count"] == 3


@pytest.mark.unit
class TestNeutralDebator:
    def test_state_mutation(self):
        llm = _mock_llm("Balanced approach is best.")
        state = _base_state(
            trader_investment_plan="Buy 100 shares",
            risk_debate_state=_risk_debate_state(),
        )
        node = create_neutral_debator(llm)

        result = node(state)["risk_debate_state"]

        assert result["history"] == "prior history\nNeutral Analyst: Balanced approach is best."
        assert result["neutral_history"] == "prior neutral\nNeutral Analyst: Balanced approach is best."
        assert result["latest_speaker"] == "Neutral"
        assert result["current_neutral_response"] == "Neutral Analyst: Balanced approach is best."
        # sibling fields preserved unchanged
        assert result["aggressive_history"] == "prior aggressive"
        assert result["conservative_history"] == "prior conservative"
        assert result["current_aggressive_response"] == "aggressive says X"
        assert result["current_conservative_response"] == "conservative says Y"
        assert result["count"] == 3
