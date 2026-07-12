"""Tests for the pure-logic message/report handling pieces of cli/main.py.

These are the only realistically unit-testable parts of a large,
mostly-interactive file: everything else calls ``questionary``/``rich``
directly with no injectable pure core. Follows the direct-import,
parametrized style of tests/test_cli_symbol_handling.py.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import cli.main as m


@pytest.mark.unit
class TestExtractContentString:
    @pytest.mark.parametrize(
        "content,expected",
        [
            ("hello world", "hello world"),
            ("  padded  ", "padded"),
            (None, None),
            ("", None),
            ("   ", None),
            (42, "42"),
        ],
    )
    def test_scalar_and_none_inputs(self, content, expected):
        assert m.extract_content_string(content) == expected

    @pytest.mark.parametrize(
        "content",
        ["[]", "None", "0"],
    )
    def test_literal_eval_falsy_strings_treated_as_empty(self, content):
        # A string that parses (via ast.literal_eval) to a falsy Python value
        # is treated as no content, not literal text.
        assert m.extract_content_string(content) is None

    def test_unparseable_string_kept_as_real_text(self):
        assert m.extract_content_string("not literal syntax!") == "not literal syntax!"

    def test_dict_with_text_key(self):
        assert m.extract_content_string({"text": "hi there"}) == "hi there"

    def test_dict_with_empty_text_key_is_none(self):
        assert m.extract_content_string({"text": ""}) is None

    def test_dict_without_text_key_is_none(self):
        assert m.extract_content_string({"foo": "bar"}) is None

    def test_list_of_typed_blocks_joins_text_parts(self):
        content = [
            {"type": "text", "text": "part1"},
            "part2",
            {"type": "other", "text": "ignored"},
        ]
        assert m.extract_content_string(content) == "part1 part2"

    def test_empty_list_is_none(self):
        assert m.extract_content_string([]) is None


@pytest.mark.unit
class TestClassifyMessageType:
    def test_human_message_is_user(self):
        assert m.classify_message_type(HumanMessage(content="hi")) == ("User", "hi")

    def test_human_message_continue_is_control(self):
        assert m.classify_message_type(HumanMessage(content="Continue")) == ("Control", "Continue")

    def test_tool_message_is_data(self):
        result = m.classify_message_type(ToolMessage(content="data", tool_call_id="1"))
        assert result == ("Data", "data")

    def test_ai_message_is_agent(self):
        assert m.classify_message_type(AIMessage(content="response")) == ("Agent", "response")

    def test_unknown_type_falls_back_to_system(self):
        class Foo:
            content = "plain"

        assert m.classify_message_type(Foo()) == ("System", "plain")


@pytest.mark.unit
class TestFormatTokens:
    @pytest.mark.parametrize(
        "n,expected",
        [(0, "0"), (999, "999"), (1000, "1.0k"), (1500, "1.5k"), (25000, "25.0k")],
    )
    def test_formats_by_magnitude(self, n, expected):
        assert m.format_tokens(n) == expected


@pytest.mark.unit
class TestFormatToolArgs:
    def test_short_args_unchanged(self):
        assert m.format_tool_args("short", max_length=80) == "short"

    def test_long_args_truncated_with_ellipsis(self):
        result = m.format_tool_args("x" * 100, max_length=80)
        assert len(result) == 80
        assert result.endswith("...")


@pytest.mark.unit
class TestMessageBufferInit:
    def test_init_for_analysis_includes_selected_and_fixed_agents(self):
        buf = m.MessageBuffer()
        buf.init_for_analysis(["market", "news"])

        assert "Market Analyst" in buf.agent_status
        assert "News Analyst" in buf.agent_status
        assert "Sentiment Analyst" not in buf.agent_status
        assert "Fundamentals Analyst" not in buf.agent_status
        for fixed_agent in ("Bull Researcher", "Bear Researcher", "Research Manager", "Trader",
                             "Aggressive Analyst", "Neutral Analyst", "Conservative Analyst",
                             "Portfolio Manager"):
            assert buf.agent_status[fixed_agent] == "pending"

    def test_init_for_analysis_report_sections_match_selection(self):
        buf = m.MessageBuffer()
        buf.init_for_analysis(["market", "news"])

        assert "market_report" in buf.report_sections
        assert "news_report" in buf.report_sections
        assert "sentiment_report" not in buf.report_sections
        assert "fundamentals_report" not in buf.report_sections
        # sections with no analyst_key are always included
        assert "investment_plan" in buf.report_sections
        assert "trader_investment_plan" in buf.report_sections
        assert "final_trade_decision" in buf.report_sections


@pytest.mark.unit
class TestMessageBufferUpdates:
    def _buffer(self):
        buf = m.MessageBuffer()
        buf.init_for_analysis(["market", "news"])
        return buf

    def test_update_agent_status_noop_for_unknown_agent(self):
        buf = self._buffer()
        buf.update_agent_status("Not A Real Agent", "completed")
        assert "Not A Real Agent" not in buf.agent_status

    def test_update_agent_status_sets_current_agent(self):
        buf = self._buffer()
        buf.update_agent_status("Market Analyst", "in_progress")
        assert buf.agent_status["Market Analyst"] == "in_progress"
        assert buf.current_agent == "Market Analyst"

    def test_update_report_section_noop_for_unselected_section(self):
        buf = self._buffer()
        buf.update_report_section("sentiment_report", "should be ignored")
        assert "sentiment_report" not in buf.report_sections

    def test_update_report_section_stores_content(self):
        buf = self._buffer()
        buf.update_report_section("market_report", "market content")
        assert buf.report_sections["market_report"] == "market content"


@pytest.mark.unit
class TestGetCompletedReportsCount:
    def _buffer(self):
        buf = m.MessageBuffer()
        buf.init_for_analysis(["market"])
        return buf

    def test_zero_when_nothing_set(self):
        buf = self._buffer()
        assert buf.get_completed_reports_count() == 0

    def test_content_alone_does_not_count(self):
        # Interim debate-round content must not count as a completed report.
        buf = self._buffer()
        buf.update_report_section("market_report", "draft content")
        assert buf.get_completed_reports_count() == 0

    def test_agent_completed_alone_does_not_count(self):
        buf = self._buffer()
        buf.update_agent_status("Market Analyst", "completed")
        assert buf.get_completed_reports_count() == 0

    def test_content_and_completed_agent_counts(self):
        buf = self._buffer()
        buf.update_report_section("market_report", "final content")
        buf.update_agent_status("Market Analyst", "completed")
        assert buf.get_completed_reports_count() == 1


@pytest.mark.unit
class TestFinalReportAssembly:
    def test_sections_assembled_in_fixed_order_with_headers(self):
        buf = m.MessageBuffer()
        buf.init_for_analysis(["market"])

        buf.update_report_section("market_report", "market body")
        buf.update_report_section("trader_investment_plan", "trader body")

        assert "## Analyst Team Reports" in buf.final_report
        assert "### Market Analysis\nmarket body" in buf.final_report
        assert "## Trading Team Plan" in buf.final_report
        assert "trader body" in buf.final_report
        # Sections never set are omitted entirely.
        assert "Portfolio Management Decision" not in buf.final_report

    def test_final_report_none_when_nothing_set(self):
        buf = m.MessageBuffer()
        buf.init_for_analysis(["market"])
        assert buf.final_report is None
