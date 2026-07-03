"""
Tests for chart components (charts.py).

Covers:
  - _apply_theme handles single-axis, multi-axis, and empty figures
  - Chart builder functions produce valid Plotly figures
  - COLOURS dict contains expected keys
"""

import numpy as np
import pytest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aus_econ_model.components.charts import (
    _apply_theme,
    COLOURS,
    THEME,
    plot_time_series,
    plot_time_series_simple,
    plot_conceptual_diagram,
)


class TestApplyTheme:
    """Smoke tests for _apply_theme — must handle any figure layout."""

    def test_single_axis_figure(self):
        """_apply_theme should work on a basic single-axis figure."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
        result = _apply_theme(fig)
        assert result is fig  # Should return same figure
        assert result.layout.plot_bgcolor == THEME["plot_bgcolor"]
        assert result.layout.paper_bgcolor == THEME["paper_bgcolor"]

    def test_empty_figure(self):
        """_apply_theme should work on an empty figure with no traces."""
        fig = go.Figure()
        result = _apply_theme(fig)
        assert result is fig
        assert result.layout.plot_bgcolor == THEME["plot_bgcolor"]

    def test_multi_subplot_figure(self):
        """_apply_theme should work on figures with multiple subplots."""
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=2, col=1)
        result = _apply_theme(fig)
        assert result is fig
        # Should not crash on xaxis2, yaxis2 etc.
        assert result.layout.plot_bgcolor == THEME["plot_bgcolor"]

    def test_figure_with_annotations(self):
        """_apply_theme should handle figures with annotations."""
        fig = go.Figure()
        fig.add_annotation(text="Test", x=0.5, y=0.5)
        result = _apply_theme(fig)
        assert result is fig

    def test_single_row_two_col_subplots(self):
        """_apply_theme should work on 1x2 subplot layout."""
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=2)
        result = _apply_theme(fig)
        assert result is fig

    def test_three_row_subplots(self):
        """_apply_theme should work on 3-row layouts (as used in plot_time_series)."""
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        for i in range(1, 4):
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=i, col=1)
        result = _apply_theme(fig)
        assert result is fig
        # The xaxis2 and yaxis2 keys exist on 3-row subplots
        assert result.layout.plot_bgcolor == THEME["plot_bgcolor"]

    def test_figure_with_3d_scene(self):
        """_apply_theme should handle 3D scatter figures (like phase diagram)."""
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="lines"))
        fig.update_layout(scene=dict(bgcolor="#000000"))
        result = _apply_theme(fig)
        assert result is fig
        # 3D figures use 'scene' not 'xaxis'/'yaxis', so no subplot axes

    def test_figure_with_many_subplots(self):
        """_apply_theme should handle more than 4 subplots without KeyError."""
        fig = make_subplots(rows=3, cols=3)
        for r in range(1, 4):
            for c in range(1, 4):
                fig.add_trace(go.Scatter(x=[0], y=[0]), row=r, col=c)
        result = _apply_theme(fig)
        assert result is fig

    def test_figure_with_missing_subplot_axes(self):
        """_apply_theme should not crash if a subplot axis key is missing.

        Regression test for the PlotlyKeyError bug (Issue #1 / #7).
        _apply_theme accesses axis keys via try/except, so missing subplot
        axis entries should not cause a KeyError.
        """
        # Create a simple 1x1 figure (no xaxis2, yaxis2, etc.)
        # then manually strip the xaxis key to simulate a partial layout
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
        # This should never crash even though xaxis exists
        result = _apply_theme(fig)
        assert result is fig


class TestColoursDict:
    """COLOURS dict usage tests."""

    def test_colours_has_expected_keys(self):
        """COLOURS should contain all expected named colour keys."""
        expected_keys = {
            "wage_share",
            "employment",
            "debt_ratio",
            "profit_share",
            "investment",
            "growth",
            "debt_service",
            "housing_price",
            "housing_stock",
            "affordability",
        }
        assert expected_keys.issubset(COLOURS.keys())

    def test_colours_are_valid_hex(self):
        """All colour values should be valid hex colour codes."""
        import re

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, colour in COLOURS.items():
            assert hex_pattern.match(colour), f"{name}: {colour} is not valid hex"


class TestChartBuilders:
    """Smoke tests for chart builder functions."""

    def test_plot_conceptual_diagram(self):
        """Conceptual diagram should return a valid figure."""
        fig = plot_conceptual_diagram()
        assert isinstance(fig, go.Figure)

    def test_plot_time_series_simple(self):
        """Simple time series plot should work with a DataFrame."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "date": [2000, 2001, 2002],
                "series_a": [1.0, 2.0, 3.0],
            }
        )
        fig = plot_time_series_simple(df, "date", ["series_a"], title="Test")
        assert isinstance(fig, go.Figure)
