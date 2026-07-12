"""Tests for GraphSetup — the graph topology wiring (nodes/edges built per
analyst selection).

``setup_graph()`` returns an uncompiled ``StateGraph``, so its ``.nodes``
(node name -> node) and ``.edges``/``.branches`` (direct / conditional edge
maps) are inspectable directly without compiling or running a real LLM.
Untested previously; a wrong edge target or a node missing for a selected
analyst would only have surfaced via a full integration run.
"""

import pytest

from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS, build_analyst_execution_plan
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup

FIXED_NODES = {
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Neutral Analyst",
    "Conservative Analyst",
    "Portfolio Manager",
}


def _build_workflow(selected_analysts):
    tool_nodes = {key: (lambda state: state) for key in selected_analysts}
    setup = GraphSetup(
        quick_thinking_llm=None,
        deep_thinking_llm=None,
        tool_nodes=tool_nodes,
        conditional_logic=ConditionalLogic(),
    )
    return setup.setup_graph(selected_analysts=selected_analysts)


@pytest.mark.unit
class TestNodeSelection:
    def test_only_selected_analyst_nodes_are_added(self):
        workflow = _build_workflow(["market", "news"])
        node_names = set(workflow.nodes.keys())

        for key in ("market", "news"):
            spec = ANALYST_NODE_SPECS[key]
            assert spec.agent_node in node_names
            assert spec.clear_node in node_names
            assert spec.tool_node in node_names

        for key in ("social", "fundamentals"):
            spec = ANALYST_NODE_SPECS[key]
            assert spec.agent_node not in node_names
            assert spec.clear_node not in node_names
            assert spec.tool_node not in node_names

    def test_fixed_nodes_always_present(self):
        workflow = _build_workflow(["market"])
        assert FIXED_NODES.issubset(workflow.nodes.keys())

    def test_all_four_analysts_selected(self):
        workflow = _build_workflow(["market", "social", "news", "fundamentals"])
        node_names = set(workflow.nodes.keys())
        for spec in ANALYST_NODE_SPECS.values():
            assert spec.agent_node in node_names
            assert spec.clear_node in node_names
            assert spec.tool_node in node_names


@pytest.mark.unit
class TestEdgeSequencing:
    def test_start_edges_to_first_selected_analyst(self):
        workflow = _build_workflow(["news", "market"])
        assert ("__start__", "News Analyst") in workflow.edges

    def test_selection_order_drives_clear_to_next_analyst_edge(self):
        workflow_news_first = _build_workflow(["news", "market"])
        assert ("Msg Clear News", "Market Analyst") in workflow_news_first.edges
        assert ("Msg Clear Market", "News Analyst") not in workflow_news_first.edges

        workflow_market_first = _build_workflow(["market", "news"])
        assert ("Msg Clear Market", "News Analyst") in workflow_market_first.edges
        assert ("Msg Clear News", "Market Analyst") not in workflow_market_first.edges

    def test_tool_node_edges_back_to_agent(self):
        workflow = _build_workflow(["market"])
        assert ("tools_market", "Market Analyst") in workflow.edges

    def test_last_analyst_clear_node_routes_to_bull_researcher(self):
        workflow = _build_workflow(["fundamentals"])
        assert ("Msg Clear Fundamentals", "Bull Researcher") in workflow.edges

    def test_single_analyst_clear_node_routes_to_bull_researcher(self):
        workflow = _build_workflow(["news", "market", "fundamentals"])
        # last spec in the plan (fundamentals) routes onward to Bull Researcher
        assert ("Msg Clear Fundamentals", "Bull Researcher") in workflow.edges
        # non-last analysts route to the next analyst, not Bull Researcher
        assert ("Msg Clear News", "Bull Researcher") not in workflow.edges
        assert ("Msg Clear Market", "Bull Researcher") not in workflow.edges

    def test_conditional_edge_registered_per_selected_analyst(self):
        workflow = _build_workflow(["market", "news"])
        assert "Market Analyst" in workflow.branches
        assert "News Analyst" in workflow.branches

    def test_fixed_debate_and_risk_conditional_edges_present(self):
        workflow = _build_workflow(["market"])
        for source in (
            "Bull Researcher",
            "Bear Researcher",
            "Aggressive Analyst",
            "Conservative Analyst",
            "Neutral Analyst",
        ):
            assert source in workflow.branches

        assert ("Research Manager", "Trader") in workflow.edges
        assert ("Trader", "Aggressive Analyst") in workflow.edges
        assert ("Portfolio Manager", "__end__") in workflow.edges


@pytest.mark.unit
class TestErrorHandling:
    def test_missing_tool_node_for_selected_analyst_raises_key_error(self):
        setup = GraphSetup(
            quick_thinking_llm=None,
            deep_thinking_llm=None,
            tool_nodes={},  # no entry for "market"
            conditional_logic=ConditionalLogic(),
        )
        with pytest.raises(KeyError):
            setup.setup_graph(selected_analysts=["market"])

    def test_unknown_analyst_key_raises_value_error(self):
        setup = GraphSetup(
            quick_thinking_llm=None,
            deep_thinking_llm=None,
            tool_nodes={"market": (lambda state: state)},
            conditional_logic=ConditionalLogic(),
        )
        with pytest.raises(ValueError):
            setup.setup_graph(selected_analysts=["market", "macro"])


@pytest.mark.unit
def test_build_analyst_execution_plan_reused_for_expected_nodes():
    # Sanity: this test file's expectations are derived from the same plan
    # builder setup.py itself consumes, so a spec rename updates both.
    plan = build_analyst_execution_plan(["market", "news"])
    workflow = _build_workflow(["market", "news"])
    for spec in plan.specs:
        assert spec.agent_node in workflow.nodes
        assert spec.clear_node in workflow.nodes
        assert spec.tool_node in workflow.nodes
