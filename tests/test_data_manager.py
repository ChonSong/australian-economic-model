"""
Tests for the data manager module (data_manager.py).

Covers:
  - Date parsing (RBA and ABS formats)
  - CSV parser helpers
  - DataManager initialization and cache paths
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime

from aus_econ_model.models.data_manager import (
    parse_date,
    _parse_abs_period,
    _find_header_row,
    _parse_rba_csv,
    _parse_abs_csv,
    DataManager,
    RBA_SOURCES,
    ABS_SOURCES,
)


class TestParseDate:
    """Test the flexible date parser."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("30/09/1976", pd.Timestamp("1976-09-30")),
            ("03-Jul-2013", pd.Timestamp("2013-07-03")),
            ("2013-07-03", pd.Timestamp("2013-07-03")),
            ("2013-07", pd.Timestamp("2013-07-01")),
            ("201307", pd.Timestamp("2013-07-01")),
            ("Jul-2013", pd.Timestamp("2013-07-01")),
            ("2024", pd.Timestamp("2024-01-01")),
        ],
    )
    def test_valid_dates(self, input_str, expected):
        """Common date formats should parse successfully."""
        result = parse_date(input_str)
        assert result == expected, f"Failed to parse '{input_str}'"

    def test_none_date(self):
        """None should return NaT."""
        result = parse_date(None)
        assert pd.isna(result)

    def test_nan_date(self):
        """NaN float should return NaT."""
        result = parse_date(np.nan)
        assert pd.isna(result)

    def test_empty_string(self):
        """Empty string should return NaT."""
        result = parse_date("")
        assert pd.isna(result)

    @pytest.mark.parametrize("bad_input", ["...", "NA", "N/A", "-"])
    def test_missing_indicators(self, bad_input):
        """Common missing-value indicators should return NaT."""
        result = parse_date(bad_input)
        assert pd.isna(result)

    def test_quoted_date(self):
        """Dates wrapped in quotes should still parse."""
        result = parse_date('"30/06/2020"')
        assert pd.notna(result)

    def test_year_only_extraction(self):
        """Year-only strings should extract as Jan 1."""
        result = parse_date("2024")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_abs_period_quarterly(self):
        """ABS quarterly format YYYY-QN should parse."""
        assert _parse_abs_period("2024-Q3") == pd.Timestamp("2024-07-01")
        assert _parse_abs_period("1997-Q1") == pd.Timestamp("1997-01-01")

    def test_abs_period_monthly(self):
        """ABS monthly format YYYY-MM should parse."""
        assert _parse_abs_period("2018-09") == pd.Timestamp("2018-09-01")

    def test_abs_period_annual(self):
        """ABS annual format YYYY should parse."""
        assert _parse_abs_period("1945") == pd.Timestamp("1945-01-01")

    def test_abs_period_half_yearly(self):
        """ABS half-yearly format YYYY-SN should parse."""
        assert _parse_abs_period("2024-S1") == pd.Timestamp("2024-01-01")
        assert _parse_abs_period("2024-S2") == pd.Timestamp("2024-07-01")


class TestRbaCsvParser:
    """Test the RBA CSV parser helper."""

    def test_find_header_row_with_series_id(self):
        """Should find header row labelled 'Series ID'."""
        lines = [
            "Preamble line 1",
            "Preamble line 2",
            "Series ID, A123, B456",
            "01/01/2020, 100, 200",
        ]
        header, data = _find_header_row(lines)
        assert header == 2
        assert data == 3

    def test_find_header_row_with_date_pattern(self):
        """Should fall back to detecting date pattern in first column."""
        lines = [
            "Preamble",
            "Some header",
            "01/01/2020, 100, 200",
            "02/01/2020, 101, 201",
        ]
        header, data = _find_header_row(lines)
        assert header == 1
        assert data == 2

    def test_empty_lines_handling(self):
        """Parser should handle empty input."""
        result = _parse_rba_csv("")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_simple_csv_parse(self):
        """Parser should handle a minimal valid RBA CSV."""
        csv_content = "Series ID,VAL1,VAL2\n01/01/2020,100,200\n01/02/2020,101,201\n"
        result = _parse_rba_csv(csv_content)
        assert not result.empty
        assert "date" in result.columns
        assert len(result) == 2


class TestAbsCsvParser:
    """Test the ABS CSV parser."""

    def test_empty_content(self):
        """Empty content should return empty DataFrame."""
        result = _parse_abs_csv("")
        assert isinstance(result, pd.DataFrame)

    def test_simple_abs_csv(self):
        """Minimal ABS CSV with standard columns."""
        csv_content = (
            "DATAFLOW,MEASURE,TIME_PERIOD,OBS_VALUE\n"
            "CPI,A1,2024-Q3,135.2\n"
            "CPI,A1,2024-Q4,136.5\n"
        )
        result = _parse_abs_csv(csv_content)
        assert not result.empty
        assert (
            "date" in result.columns
            or "OBS_VALUE" in result.columns
            or "value" in result.columns
        )


class TestDataManagerInit:
    """Test DataManager initialization."""

    def test_default_initialization(self):
        """DataManager should initialise with default paths."""
        dm = DataManager()
        assert dm.cache_max_age.days == 1
        assert dm.rba_dir is not None
        assert dm.abs_dir is not None

    def test_custom_cache_dir(self, tmp_path):
        """DataManager should accept a custom cache directory."""
        dm = DataManager(cache_dir=tmp_path)
        assert dm.data_dir == tmp_path

    def test_cache_path_construction(self, tmp_path):
        """Cache paths should follow expected pattern."""
        dm = DataManager(cache_dir=tmp_path)
        rba_path = dm._cache_path("rba", "D2")
        assert rba_path.name == "D2.parquet"
        assert "rba" in str(rba_path)

    def test_cache_freshness_missing(self, tmp_path):
        """_is_fresh should return False for missing files."""
        dm = DataManager(cache_dir=tmp_path)
        missing = tmp_path / "nonexistent.parquet"
        assert dm._is_fresh(missing) is False

    def test_source_registry_contents(self):
        """RBA and ABS source registries should contain expected entries."""
        assert "D2" in RBA_SOURCES  # Lending and Credit
        assert "CPI" in ABS_SOURCES
        assert "LF" in ABS_SOURCES
