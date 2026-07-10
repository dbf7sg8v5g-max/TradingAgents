"""Tests for the untested remainder of tradingagents/dataflows/utils.py.

``safe_ticker_component`` already has dedicated coverage in
tests/test_safe_ticker_component.py and is not duplicated here.
"""

from datetime import datetime

import pandas as pd
import pytest

from tradingagents.dataflows.utils import (
    decorate_all_methods,
    get_current_date,
    get_next_weekday,
    save_output,
)


@pytest.mark.unit
class TestGetNextWeekday:
    def test_saturday_rolls_to_monday(self):
        # 2024-10-19 is a Saturday.
        result = get_next_weekday("2024-10-19")
        assert result == datetime(2024, 10, 21)

    def test_sunday_rolls_to_monday(self):
        # 2024-10-20 is a Sunday.
        result = get_next_weekday("2024-10-20")
        assert result == datetime(2024, 10, 21)

    def test_weekday_is_unchanged(self):
        # 2024-10-16 is a Wednesday.
        result = get_next_weekday("2024-10-16")
        assert result == datetime(2024, 10, 16)

    def test_string_input_matches_datetime_input(self):
        as_string = get_next_weekday("2024-10-19")
        as_datetime = get_next_weekday(datetime(2024, 10, 19))
        assert as_string == as_datetime

    def test_does_not_account_for_market_holidays(self):
        # A weekday that happens to be a market holiday is still returned
        # unchanged — this function is weekend-aware only, not holiday-aware.
        result = get_next_weekday("2024-01-01")  # New Year's Day, a Monday
        assert result == datetime(2024, 1, 1)


@pytest.mark.unit
class TestDecorateAllMethods:
    def test_wraps_callables_and_records_calls(self):
        calls = []

        def counting_decorator(func):
            def wrapper(*args, **kwargs):
                calls.append(func.__name__)
                return func(*args, **kwargs)
            return wrapper

        @decorate_all_methods(counting_decorator)
        class Toy:
            CONST = "unchanged"

            def greet(self):
                return "hello"

            def add(self, a, b):
                return a + b

        toy = Toy()
        assert toy.greet() == "hello"
        assert toy.add(2, 3) == 5
        assert calls == ["greet", "add"]

    def test_non_callable_attributes_untouched(self):
        def identity_decorator(func):
            return func

        @decorate_all_methods(identity_decorator)
        class Toy:
            CONST = 42

        assert Toy.CONST == 42


@pytest.mark.unit
class TestSaveOutput:
    def test_writes_csv_when_save_path_given(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out_file = tmp_path / "out.csv"

        save_output(df, "tag", save_path=str(out_file))

        assert out_file.exists()
        read_back = pd.read_csv(out_file, index_col=0)
        assert list(read_back["a"]) == [1, 2]

    def test_noop_when_save_path_none(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        save_output(df, "tag", save_path=None)
        assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
class TestGetCurrentDate:
    def test_returns_iso_format(self):
        result = get_current_date()
        assert len(result) == 10
        year, month, day = result.split("-")
        assert year.isdigit() and len(year) == 4
        assert month.isdigit() and len(month) == 2
        assert day.isdigit() and len(day) == 2
