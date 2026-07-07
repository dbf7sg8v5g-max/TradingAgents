"""Tests for ConditionalLogic — the graph's debate/tool-loop routing.

These routing methods control termination of the multi-agent debate and
risk-discussion loops. A regression here (an off-by-one in the round-count
threshold, or a typo'd ``.startswith`` prefix) would silently break
round-robin routing — infinite loop or premature exit — with nothing else
in the suite to catch it, since no other test imports ``ConditionalLogic``.
"""

from types import SimpleNamespace

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic


def _state_with_last_message(tool_calls):
    return {"messages": [SimpleNamespace(tool_calls=tool_calls)]}


def _debate_state(count, current_response=""):
    return {
        "investment_debate_state": {
            "count": count,
            "current_response": current_response,
        }
    }


def _risk_state(count, latest_speaker=""):
    return {
        "risk_debate_state": {
            "count": count,
            "latest_speaker": latest_speaker,
        }
    }


@pytest.mark.unit
class TestAnalystToolLoops:
    @pytest.mark.parametrize(
        "method,tools_label,clear_label",
        [
            ("should_continue_market", "tools_market", "Msg Clear Market"),
            ("should_continue_social", "tools_social", "Msg Clear Sentiment"),
            ("should_continue_news", "tools_news", "Msg Clear News"),
            ("should_continue_fundamentals", "tools_fundamentals", "Msg Clear Fundamentals"),
        ],
    )
    def test_routes_to_tools_when_tool_calls_present(self, method, tools_label, clear_label):
        logic = ConditionalLogic()
        state = _state_with_last_message(tool_calls=[{"name": "get_stock_data", "args": {}, "id": "1"}])
        assert getattr(logic, method)(state) == tools_label

    @pytest.mark.parametrize(
        "method,tools_label,clear_label",
        [
            ("should_continue_market", "tools_market", "Msg Clear Market"),
            ("should_continue_social", "tools_social", "Msg Clear Sentiment"),
            ("should_continue_news", "tools_news", "Msg Clear News"),
            ("should_continue_fundamentals", "tools_fundamentals", "Msg Clear Fundamentals"),
        ],
    )
    def test_routes_to_clear_when_no_tool_calls(self, method, tools_label, clear_label):
        logic = ConditionalLogic()
        state = _state_with_last_message(tool_calls=[])
        assert getattr(logic, method)(state) == clear_label


@pytest.mark.unit
class TestShouldContinueDebate:
    def test_below_threshold_routes_by_last_speaker_bull(self):
        logic = ConditionalLogic(max_debate_rounds=1)
        state = _debate_state(count=0, current_response="Bull Analyst: ...")
        assert logic.should_continue_debate(state) == "Bear Researcher"

    def test_below_threshold_routes_by_last_speaker_bear(self):
        logic = ConditionalLogic(max_debate_rounds=1)
        state = _debate_state(count=1, current_response="Bear Analyst: ...")
        assert logic.should_continue_debate(state) == "Bull Researcher"

    def test_empty_current_response_defaults_to_bull(self):
        logic = ConditionalLogic(max_debate_rounds=1)
        state = _debate_state(count=0, current_response="")
        assert logic.should_continue_debate(state) == "Bull Researcher"

    def test_at_threshold_routes_to_research_manager(self):
        # max_debate_rounds=1 -> threshold is 2*1=2 turns (one each).
        logic = ConditionalLogic(max_debate_rounds=1)
        state = _debate_state(count=2, current_response="Bear Analyst: ...")
        assert logic.should_continue_debate(state) == "Research Manager"

    def test_just_below_threshold_still_continues(self):
        logic = ConditionalLogic(max_debate_rounds=1)
        state = _debate_state(count=1, current_response="Bull Analyst: ...")
        assert logic.should_continue_debate(state) == "Bear Researcher"

    def test_above_threshold_routes_to_research_manager(self):
        logic = ConditionalLogic(max_debate_rounds=1)
        state = _debate_state(count=5, current_response="Bull Analyst: ...")
        assert logic.should_continue_debate(state) == "Research Manager"

    def test_nondefault_max_rounds_scales_threshold(self):
        # max_debate_rounds=3 -> threshold is 2*3=6; count=5 must still continue.
        logic = ConditionalLogic(max_debate_rounds=3)
        state = _debate_state(count=5, current_response="Bull Analyst: ...")
        assert logic.should_continue_debate(state) == "Bear Researcher"

        state_at_threshold = _debate_state(count=6, current_response="Bear Analyst: ...")
        assert logic.should_continue_debate(state_at_threshold) == "Research Manager"


@pytest.mark.unit
class TestShouldContinueRiskAnalysis:
    def test_aggressive_speaker_routes_to_conservative(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _risk_state(count=0, latest_speaker="Aggressive")
        assert logic.should_continue_risk_analysis(state) == "Conservative Analyst"

    def test_conservative_speaker_routes_to_neutral(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _risk_state(count=1, latest_speaker="Conservative")
        assert logic.should_continue_risk_analysis(state) == "Neutral Analyst"

    def test_neutral_speaker_routes_to_aggressive(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _risk_state(count=2, latest_speaker="Neutral")
        assert logic.should_continue_risk_analysis(state) == "Aggressive Analyst"

    def test_empty_latest_speaker_defaults_to_aggressive(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _risk_state(count=0, latest_speaker="")
        assert logic.should_continue_risk_analysis(state) == "Aggressive Analyst"

    def test_at_threshold_routes_to_portfolio_manager(self):
        # max_risk_discuss_rounds=1 -> threshold is 3*1=3 turns (one each).
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _risk_state(count=3, latest_speaker="Neutral")
        assert logic.should_continue_risk_analysis(state) == "Portfolio Manager"

    def test_just_below_threshold_still_continues(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _risk_state(count=2, latest_speaker="Neutral")
        assert logic.should_continue_risk_analysis(state) == "Aggressive Analyst"

    def test_above_threshold_routes_to_portfolio_manager(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = _risk_state(count=7, latest_speaker="Aggressive")
        assert logic.should_continue_risk_analysis(state) == "Portfolio Manager"

    def test_nondefault_max_rounds_scales_threshold(self):
        # max_risk_discuss_rounds=2 -> threshold is 3*2=6; count=5 must still continue.
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = _risk_state(count=5, latest_speaker="Neutral")
        assert logic.should_continue_risk_analysis(state) == "Aggressive Analyst"

        state_at_threshold = _risk_state(count=6, latest_speaker="Neutral")
        assert logic.should_continue_risk_analysis(state_at_threshold) == "Portfolio Manager"
