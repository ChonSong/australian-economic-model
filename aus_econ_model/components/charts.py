"""
Reusable visualisation components for the Streamlit app.

Provides Plotly chart builders that directly consume model output
and data frames, formatted with a consistent dark theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour palette ──────────────────────────────────────────────────────────

COLOURS = {
    "wage_share": "#E74C3C",
    "employment": "#3498DB",
    "debt_ratio": "#F39C12",
    "profit_share": "#2ECC71",
    "investment": "#9B59B6",
    "growth": "#1ABC9C",
    "debt_service": "#E67E22",
    "housing_price": "#FF6B6B",
    "housing_stock": "#4ECDC4",
    "affordability": "#FFE66D",
}

THEME = {
    "plot_bgcolor": "#0e1117",
    "paper_bgcolor": "#0e1117",
    "font_color": "#FAFAFA",
    "grid_color": "rgba(255,255,255,0.08)",
    "zeroline_color": "rgba(255,255,255,0.15)",
}


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=THEME["plot_bgcolor"],
        paper_bgcolor=THEME["paper_bgcolor"],
        font=dict(color=THEME["font_color"], size=12),
        title=dict(font=dict(color=THEME["font_color"])),
        legend=dict(font=dict(color=THEME["font_color"]),
                    bordercolor=THEME["grid_color"]),
        hoverlabel=dict(font=dict(color=THEME["font_color"]),
                        bgcolor="#1e1e1e"),
        xaxis=dict(gridcolor=THEME["grid_color"],
                   zerolinecolor=THEME["zeroline_color"],
                   title=dict(font=dict(color=THEME["font_color"])),
                   tickfont=dict(color=THEME["font_color"])),
        yaxis=dict(gridcolor=THEME["grid_color"],
                   zerolinecolor=THEME["zeroline_color"],
                   title=dict(font=dict(color=THEME["font_color"])),
                   tickfont=dict(color=THEME["font_color"])),
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
    )
    # Also theme secondary axes if they exist
    for axis_key in ["xaxis2", "xaxis3", "xaxis4",
                     "yaxis2", "yaxis3", "yaxis4",
                     "xaxis5", "xaxis6", "yaxis5", "yaxis6"]:
        try:
            axis = fig.layout[axis_key]
        except (KeyError, AttributeError):
            axis = None
        if axis is not None:
            axis.gridcolor = THEME["grid_color"]
            axis.zerolinecolor = THEME["zeroline_color"]
            if hasattr(axis, "title"):
                axis.title.font.color = THEME["font_color"]
            if hasattr(axis, "tickfont"):
                axis.tickfont.color = THEME["font_color"]
    # Theme annotations
    for ann in fig.layout.annotations:
        if ann is not None:
            ann.font.color = ann.font.color if ann.font and ann.font.color else THEME["font_color"]
    return fig


# ── Model Output Charts ─────────────────────────────────────────────────────


def plot_time_series(sol, width=None, height=400):
    """
    Three-panel time series: wage share, employment rate, debt ratio.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Wage Share (ω)", "Employment Rate (λ)", "Private Debt / GDP (d)"),
        vertical_spacing=0.06,
    )

    fig.add_trace(go.Scatter(
        x=sol.t, y=sol.omega, mode="lines", name="Wage Share",
        line=dict(color=COLOURS["wage_share"], width=2),
        hovertemplate="Year %{x:.0f}<br>ω = %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sol.t, y=sol.lam, mode="lines", name="Employment",
        line=dict(color=COLOURS["employment"], width=2),
        hovertemplate="Year %{x:.0f}<br>λ = %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sol.t, y=sol.d, mode="lines", name="Debt Ratio",
        line=dict(color=COLOURS["debt_ratio"], width=2),
        hovertemplate="Year %{x:.0f}<br>d = %{y:.2f}<extra></extra>",
    ), row=3, col=1)

    # Add zero lines and reference levels
    for row in [1, 2, 3]:
        fig.add_hline(y=0, line_color=THEME["zeroline_color"], line_width=1, row=row, col=1)

    fig.update_layout(
        height=height,
        showlegend=False,
        **({"width": width} if width else {}),
    )
    fig.update_xaxes(title_text="Years", row=3, col=1)
    return _apply_theme(fig)


def plot_phase_diagram(sol, width=None, height=400):
    """
    3D phase diagram: debt ratio vs employment vs wage share.
    Shows the trajectory path through state space.
    """
    fig = go.Figure()

    # Colour the trajectory by time
    t_norm = (sol.t - sol.t.min()) / (sol.t.max() - sol.t.min() + 1e-10)

    fig.add_trace(go.Scatter3d(
        x=sol.d,
        y=sol.omega,
        z=sol.lam,
        mode="lines",
        line=dict(
            color=t_norm,
            colorscale="Viridis",
            width=4,
            showscale=True,
            colorbar=dict(title="Time"),
        ),
        name="Trajectory",
        hovertemplate="Debt/GDP: %{x:.2f}<br>Wage Share: %{y:.3f}<br>Employment: %{z:.3f}<extra></extra>",
    ))

    # Mark start and end
    fig.add_trace(go.Scatter3d(
        x=[sol.d[0]], y=[sol.omega[0]], z=[sol.lam[0]],
        mode="markers",
        marker=dict(color="#2ECC71", size=8, symbol="circle"),
        name="Start",
    ))

    fig.add_trace(go.Scatter3d(
        x=[sol.d[-1]], y=[sol.omega[-1]], z=[sol.lam[-1]],
        mode="markers",
        marker=dict(color="#E74C3C", size=8, symbol="x"),
        name="End" if sol.t[-1] > sol.t[0] else "",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Debt / GDP (d)",
            yaxis_title="Wage Share (ω)",
            zaxis_title="Employment (λ)",
            xaxis=dict(gridcolor=THEME["grid_color"]),
            yaxis=dict(gridcolor=THEME["grid_color"]),
            zaxis=dict(gridcolor=THEME["grid_color"]),
            bgcolor=THEME["plot_bgcolor"],
        ),
        height=height,
        **({"width": width} if width else {}),
    )
    return _apply_theme(fig)


def plot_debt_dynamics(sol, width=None, height=300):
    """Debt service ratio and investment share over time."""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Debt Service vs Investment"),
    )

    fig.add_trace(go.Scatter(
        x=sol.t, y=sol.debt_service_ratio, mode="lines",
        name="Debt Service (r × d)",
        line=dict(color=COLOURS["debt_service"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=sol.t, y=sol.investment_share, mode="lines",
        name="Investment Share κ(π)",
        line=dict(color=COLOURS["investment"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=sol.t, y=sol.profit_share, mode="lines",
        name="Profit Share (π)",
        line=dict(color=COLOURS["profit_share"], width=2, dash="dot"),
    ))

    fig.update_layout(height=height, showlegend=True)
    fig.update_xaxes(title_text="Years")
    fig.update_yaxes(title_text="Share of GDP")
    return _apply_theme(fig)


# ── Data Charts ─────────────────────────────────────────────────────────────


def plot_time_series_simple(df, x_col, y_cols, title="", height=400):
    """Generic multi-line time series plot from a DataFrame."""
    fig = go.Figure()
    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col] if x_col in df.columns else df.index,
                y=df[col],
                mode="lines",
                name=col,
            ))
    fig.update_layout(title=title, height=height)
    return _apply_theme(fig)


def plot_housing_submodel(core, housing, width=None, height=350):
    """Two-panel housing sub-model output."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Housing Price / Income", "Debt Ratio vs Housing"),
        shared_xaxes=False,
    )

    fig.add_trace(go.Scatter(
        x=core.t, y=housing["price_to_income"], mode="lines",
        name="Price/Income",
        line=dict(color=COLOURS["housing_price"], width=2),
    ), row=1, col=1)

    # Debt ratio vs housing price (scatter coloured by time)
    t_norm = (core.t - core.t.min()) / (core.t.max() - core.t.min() + 1e-10)
    fig.add_trace(go.Scatter(
        x=core.d, y=housing["price_to_income"],
        mode="markers",
        marker=dict(color=t_norm, colorscale="Viridis", size=4, showscale=True),
        name="Debt → House Prices",
        hovertemplate="Debt/GDP: %{x:.2f}<br>Price/Income: %{y:.2f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(height=height)
    fig.update_xaxes(title_text="Years", row=1, col=1)
    fig.update_xaxes(title_text="Debt / GDP", row=1, col=2)
    fig.update_yaxes(title_text="Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Price/Income", row=1, col=2)
    return _apply_theme(fig)


# ── Annotated Diagrams ──────────────────────────────────────────────────────


def plot_conceptual_diagram():
    """
    Causal-loop diagram of the Keen model — showing the feedback mechanisms.
    Rendered as a simple network with Plotly scatter for visual explanation.
    """
    # Node positions
    nodes = {
        "Credit\nDemand": (0, 1),
        "Private\nDebt": (1, 1),
        "Aggregate\nDemand": (2, 0.5),
        "Employment": (3, 0),
        "Wage\nShare": (4, 0.5),
        "Profit\nShare": (3, 1),
        "Debt\nService": (2, 1.5),
    }

    edges = [
        ("Credit\nDemand", "Private\nDebt", "+", "Borrowing creates debt"),
        ("Private\nDebt", "Aggregate\nDemand", "+", "ΔDebt adds to AD"),
        ("Aggregate\nDemand", "Employment", "+", "More demand → more hiring"),
        ("Employment", "Wage\nShare", "+", "Tighter labour → higher wages"),
        ("Wage\nShare", "Profit\nShare", "−", "Higher wages squeeze profits"),
        ("Private\nDebt", "Debt\nService", "+", "More debt → more interest"),
        ("Debt\nService", "Profit\nShare", "−", "Interest drains profits"),
        ("Profit\nShare", "Aggregate\nDemand", "+", "Profits fund investment"),
        ("Profit\nShare", "Credit\nDemand", "+", "Expectations drive borrowing"),
        ("Wage\nShare", "Aggregate\nDemand", "+", "Wages fund consumption"),
    ]

    fig = go.Figure()

    # Nodes
    nx = [v[0] for v in nodes.values()]
    ny = [v[1] for v in nodes.values()]
    labels = list(nodes.keys())

    fig.add_trace(go.Scatter(
        x=nx, y=ny,
        mode="markers+text",
        marker=dict(size=30, color="#2C3E50", line=dict(color="#3498DB", width=2)),
        text=labels,
        textposition="middle center",
        textfont=dict(size=10, color="#FAFAFA"),
        hoverinfo="text",
        hovertext=labels,
    ))

    # Edges
    for src, dst, sign, tooltip in edges:
        sx, sy = nodes[src]
        dx, dy = nodes[dst]
        # Midpoint for label
        mx, my = (sx + dx) / 2, (sy + dy) / 2

        # Arrow (simplified)
        fig.add_annotation(
            x=dx, y=dy, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#7F8C8D",
        )
        fig.add_annotation(
            x=mx, y=my + 0.1,
            text=f"<b>{sign}</b>",
            showarrow=False,
            font=dict(size=18, color="#E74C3C" if sign == "−" else "#2ECC71"),
        )

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.5, 5]),
        yaxis=dict(visible=False, range=[-0.5, 2]),
        height=500,
        title="Keen Model — Causal Structure",
        showlegend=False,
    )
    return _apply_theme(fig)


def plot_parameter_sensitivity(sol_base, params_list, sols_list, width=None, height=400):
    """
    Overlay trajectories for different parameter values.
    Useful for sensitivity analysis.
    """
    fig = go.Figure()

    for i, (p, sol) in enumerate(zip(params_list, sols_list)):
        if sol is not None and sol.success:
            name = getattr(p, "_label", f"Run {i+1}")
            fig.add_trace(go.Scatter(
                x=sol.t, y=sol.d, mode="lines",
                name=f"d: {name}",
                line=dict(dash="dash" if i > 0 else "solid", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=sol.t, y=sol.omega, mode="lines",
                name=f"ω: {name}",
                line=dict(dash="dash" if i > 0 else "solid", width=1.5),
                visible="legendonly",  # Hidden by default
            ))

    fig.update_layout(
        title="Sensitivity Analysis — Debt Ratio",
        height=height,
        hovermode="x unified",
        **({"width": width} if width else {}),
    )
    fig.update_xaxes(title_text="Years")
    fig.update_yaxes(title_text="Debt / GDP")
    return _apply_theme(fig)
