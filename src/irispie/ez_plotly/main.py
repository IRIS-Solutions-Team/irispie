"""
"""


#[

from __future__ import annotations

import math as _ma
from typing import Literal
from numbers import Real
from collections.abc import Iterable, Sequence
import plotly.graph_objects as _pg
import plotly.subplots as _ps
import plotly.io as _pi
import documark as _dm

from ..dates import Period, Frequency

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
#]


__all__ = (
    "add_line",
    "add_bar",
    "add_histogram",
    "customize_tick_labels",
    "add_trace",
    "add_highlight",
    "add_vline",
    "auto_tiles",
    "freeze_span",
    "highlight",
    "make_subplots",
    "get_date_axis_mode",
    "resolve_date_format",
    "resolve_subplot_reference",
    "set_default_plotly_renderer",
    "update_subplot_title",
    "vline",
    "PlotlyDateAxisModeType",
)


PlotlyDateAxisModeType = Literal["instant", "period", ]


_DATE_FORMAT_STYLES = {
    "sdmx": {
        Frequency.YEARLY: "%Y",
        Frequency.HALFYEARLY: "%Y-%m",
        Frequency.QUARTERLY: "%Y-Q%q",
        Frequency.MONTHLY: "%Y-%m",
        Frequency.WEEKLY: "%Y-W%W",
        Frequency.DAILY: "%Y-%m-%d",
        Frequency.INTEGER: None,
    },
    "compact": {
        Frequency.YEARLY: "%yY",
        Frequency.HALFYEARLY: "%yM%m",
        Frequency.QUARTERLY: "%yQ%q",
        Frequency.MONTHLY: "%yM%m",
        Frequency.WEEKLY: "%yW%W",
        Frequency.DAILY: "%y%b%d",
        Frequency.INTEGER: None,
    },
}


def resolve_date_format(
    date_format_style: str | dict[int, str] | None,
    frequency: Frequency,
) -> str:
    """
    """
    if not date_format_style:
        date_format_style = "sdmx"
    if isinstance(date_format_style, str):
        date_format_style = _DATE_FORMAT_STYLES[date_format_style]
    return date_format_style[frequency]


def set_default_plotly_renderer(renderer: str, ) -> None:
    """
    """
    _pi.renderers.default = renderer


_EMPTY_SUBPLOT_TITLE = " "
_DEFAULT_VERTICAL_SPACING = 0.1
_DEFAULT_HORIZONTAL_SPACING = 0.05


@_dm.reference(category="arrange", )
def make_subplots(
    rows_columns: Iterable[int] | None = None,
    subplot_titles: Sequence[str] | bool | None = True,
    vertical_spacing: float | None = None,
    horizontal_spacing: float | None = None,
    figure_height: int | None = None,
    figure_title: str | None = None,
    show_legend: bool = False,
    **kwargs,
) -> _pg.Figure:
    """
················································································

==Create a figure with multiple subplots==

················································································
    """
    if vertical_spacing is None:
        vertical_spacing = _DEFAULT_VERTICAL_SPACING
    if horizontal_spacing is None:
        horizontal_spacing = _DEFAULT_HORIZONTAL_SPACING
    num_rows = rows_columns[0] if rows_columns else 1
    num_columns = rows_columns[1] if rows_columns else 1
    total_num_subplots = num_rows * num_columns
    if subplot_titles is True:
        subplot_titles = (_EMPTY_SUBPLOT_TITLE, ) * total_num_subplots
    figure = _ps.make_subplots(
        rows=num_rows,
        cols=num_columns,
        subplot_titles=subplot_titles or None,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        **kwargs,
    )
    #
    if figure_title is not None:
        figure.update_layout(title=figure_title, )
    #
    if figure_height is not None:
        figure.update_layout(height=figure_height, )
    #
    figure.update_layout(showlegend=show_legend, )
    #
    return figure

@_dm.reference(category="custom", )
def add_highlight(
    figure: _pg.Figure,
    span: Iterable[Period] | None = None,
    subplot: tuple[int, int] | int | None = None,
    color: str = "rgba(0, 0, 0, 0.15)",
) -> _pg.Figure:
    """
················································································

==Highlight a certain date span in charts==

Highlight a certain time span in a figure. The span can be specified either as a
time `Span`, or a tuple of two `Periods`.

    irispie.plotly.highlight(span)

················································································
    """
    highlight_start, highlight_end = _resolve_highlight_span(figure, span, )
    _, index = resolve_subplot_reference(figure, subplot, )
    xref = f"x{index+1}" if index else "x"
    yref = f"y{index+1} domain" if index else "y domain"
    shape = {
        "type": "rect",
        "xref": xref,
        "x0": highlight_start,
        "x1": highlight_end,
        "yref": yref,
        "y0": 0,
        "y1": 1,
        "fillcolor": color,
        "line": {"width": 0, },
    }
    figure.add_shape(shape, )


highlight = add_highlight


def add_vline(
    figure,
    position,
    subplot: tuple[int, int] | int | None = None,
    color: str = "rgba(0, 0, 0, 0.5)",
    width: int = None,
) -> None:
    r"""
    """
    date_axis_mode = get_date_axis_mode(figure, )
    plotly_row_col, _ = resolve_subplot_reference(figure, subplot, )
    if hasattr(position, "to_plotly_date"):
        position = position.to_plotly_date(mode=date_axis_mode, )
    figure.add_vline(position, **plotly_row_col, line={"color": color, "width": width, }, )


vline = add_vline


def freeze_span(
    figure: _pg.Figure,
    span: Iterable[Period],
    subplot: tuple[int, int] | int | None = None,
) -> None:
    """
    """
    start, end = _resolve_figure_span(figure, span, )
    xaxis_update = {
        "range": (start, end),
        "autorange": False,
    }
    plotly_row_col, *_ = resolve_subplot_reference(figure, subplot, )
    figure.update_xaxes(xaxis_update, **plotly_row_col, )


def _resolve_figure_span(
    figure: _pg.Figure,
    span: Iterable[Period] | None = None,
) -> tuple[str, str]:
    """
    """
    #[
    if hasattr(span, "start", ) and hasattr(span, "end", ):
        start = span.start if span.start else None
        end = span.end if span.end else None
    elif span:
        span = tuple(span, )
        start, end = span[0], span[-1]
    else:
        start, end = None, None
    figure_span = figure.layout.xaxis.range
    mode = get_date_axis_mode(figure, ) or "period"
    start = start.to_plotly_edge_before(mode=mode, ) if start is not None else figure_span[0]
    end = end.to_plotly_edge_after(mode=mode, ) if end is not None else figure_span[-1]
    return start, end,
    #]


def _resolve_highlight_span(
    figure: _pg.Figure,
    span: Iterable[Period] | None = None,
) -> tuple[str, str]:
    """
    """
    #[
    if hasattr(span, "start", ) and hasattr(span, "end", ):
        start = span.start if span.start else None
        end = span.end if span.end else None
    elif span:
        span = tuple(span, )
        start, end = span[0], span[-1]
    else:
        start, end = None, None
    figure_span = figure.layout.xaxis.range
    mode = get_date_axis_mode(figure, )
    highlight_start = start.to_plotly_edge_before(mode=mode, ) if start is not None else figure_span[0]
    highlight_end = end.to_plotly_edge_after(mode=mode, ) if end is not None else figure_span[-1]
    return highlight_start, highlight_end,
    #]


def resolve_subplot_reference(
    figure: _pg.Figure,
    subplot: tuple[int, int] | int | None,
) -> tuple[dict[str, int], int]:
    r"""
    """
    if subplot is None:
        return {}, None
    rows, columns = figure._get_subplot_rows_columns()
    num_rows = len(rows)
    num_columns = len(columns)
    num_tiles = num_rows * num_columns
    if isinstance(subplot, Sequence, ):
        row, column, = subplot
        index = row * num_columns + column
        plotly_row_col = {"row": row+1, "col": column+1, }
        return plotly_row_col, index,
    if isinstance(subplot, Real, ):
        index = int(subplot, ) if subplot >= 0 else num_tiles + int(subplot)
        row = index // num_columns
        column = index % num_columns
        plotly_row_col = {"row": row+1, "col": column+1, }
        return plotly_row_col, index,
    raise TypeError(f"Invalid subplot reference: {subplot}")


def add_trace(
    figure: _pg.Figure,
    trace,
    subplot: tuple[int, int] | int | None = None,
) -> None:
    plotly_row_col, _ = resolve_subplot_reference(figure, subplot, )
    figure.add_trace(trace, **plotly_row_col, )


def customize_tick_labels(
    figure: _pg.Figure,
    tick_labels: Iterable[str],
    tick_values: Iterable[Any],
    subplot: tuple[int, int] | int | None = None,
) -> None:
    plotly_row_col, _ = resolve_subplot_reference(figure, subplot, )
    figure.update_xaxes(
        ticktext=list(tick_labels),
        tickvals=list(tick_values),
        **plotly_row_col,
    )

def update_subplot_title(
    figure: _pg.Figure,
    subplot: tuple[int, int] | int,
    title: str,
) -> None:
    r"""
    """
    _, index = resolve_subplot_reference(figure, subplot, )
    annotation = next(figure.select_annotations(index, ), None, )
    if annotation:
        annotation.text = title


def get_date_axis_mode(figure: _pg.Figure, ) -> PlotlyDateAxisModeType | None:
    r"""
    """
    #[
    xaxis = figure.layout.xaxis
    return xaxis.ticklabelmode if xaxis.type == "date" else None
    #]


def add_histogram(figure, *args, subplot, **kwargs, ) -> _pg.Histogram:
    r"""
    """
    #[
    trace = _pg.Histogram(*args, **kwargs, )
    add_trace(figure, trace, subplot=subplot, )
    return trace
    #]


def add_bar(figure, *args, subplot, **kwargs, ) -> _pg.Histogram:
    r"""
    """
    #[
    trace = _pg.Bar(*args, **kwargs, )
    add_trace(figure, trace, subplot=subplot, )
    return trace
    #]


def add_line(figure, *args, subplot, **kwargs, ) -> _pg.Scatter:
    r"""
    """
    #[
    trace = _pg.Scatter(*args, **kwargs, mode="lines+markers", )
    add_trace(figure, trace, subplot=subplot, )
    return trace


def auto_tiles(num_charts, ) -> tuple[int, int]:
    r"""
    """
    #[
    n = _ma.ceil(_ma.sqrt(num_charts, ), )
    if n * (n-1) >= num_charts:
        return (n, n-1, )
    return (n, n, )
    #]


# @_dm.reference(
#     path=("visualization_reporting", "ez_plotlyper.md", ),
#     categories={
#         "arrange": "Arranging charts",
#         "custom": "Customizing charts",
#     },
# )
# class plotly:
#     """
# ················································································
# 
# Plotly wrapper
# ===============
# 
# ················································································
#     """
#     #[
# 
#     Figure = _pg.Figure
#     make_subplots = staticmethod(make_subplots)
#     update_subplot_title = staticmethod(update_subplot_title)
#     resolve_subplot_reference = staticmethod(resolve_subplot_reference)
#     highlight = staticmethod(highlight)
#     freeze_span = staticmethod(freeze_span)
#
#    #]

