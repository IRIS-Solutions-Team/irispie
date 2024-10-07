"""
"""


#[

from __future__ import absolute_import

from typing import (Sequence, Iterable, )
import plotly.graph_objects as _pg
import plotly.subplots as _ps
import documark as _dm

from .dates import (Period, )

#]


__all__ = (
    "plotly",
)


_EMPTY_SUBPLOT_TITLE = " "


@_dm.reference(category="arrange", )
def make_subplots(
    rows: int,
    columns: int | None = None,
    cols: int | None = None,
    subplot_titles: Sequence[str] | bool | None = True,
    vertical_spacing: float | None = 0.1,
    horizontal_spacing: float | None = 0.05,
    **kwargs,
) -> _pg.Figure:
    """
················································································

==Create a figure with multiple subplots==

················································································
    """
    if cols is not None and columns is None:
        columns = cols
    total_num_subplots = rows * columns
    if subplot_titles is True:
        subplot_titles = (_EMPTY_SUBPLOT_TITLE, )*total_num_subplots
    return _ps.make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=subplot_titles or None,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        **kwargs,
    )


@_dm.reference(category="custom", )
def highlight(
    figure: _pg.Figure,
    span: Iterable[Period] | None = None,
    /,
    subplot: tuple[int, int] | int | None = None,
    fill_color: str = "rgba(0, 0, 0, 0.15)",
) -> _pg.Figure:
    """
················································································

==Highlight a certain date span in charts==

Highlight a certain time span in a figure. The span can be specified either as a
time `Span`, or a tuple of two `Periods`.

    irispie.plotly.highlight(span)

················································································
    """
    start, end = _resolve_figure_span(figure, span, )
    _, index = resolve_subplot(figure, subplot, )
    xref = f"x{index+1}" if index else "x"
    yref = f"y{index+1} domain" if index else "y domain"
    shape = {
        "type": "rect",
        "xref": xref,
        "x0": start,
        "x1": end,
        "yref": yref,
        "y0": 0,
        "y1": 1,
        "fillcolor": fill_color,
        "line": {"width": 0, },
    }
    figure.add_shape(shape, )


def freeze_span(
    figure: _pg.Figure,
    span: Iterable[Period],
    /,
    subplot: tuple[int, int] | int | None = None,
) -> None:
    """
    """
    start, end = _resolve_figure_span(figure, span, )
    xaxis_update = {
        "range": (start, end),
        "autorange": False,
    }
    row_column, *_ = resolve_subplot(figure, subplot, )
    figure.update_xaxes(xaxis_update, **row_column, )


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
        start, end = span[0], span[-1]
    else:
        start, end = None, None
    figure_span = figure.layout.xaxis.range
    start = start.to_iso_string(position="start", ) if start is not None else figure_span[0]
    end = end.to_iso_string(position="end", ) if end is not None else figure_span[-1]
    return start, end,
    #]


def resolve_subplot(
    figure: _pg.Figure,
    subplot: tuple[int, int] | int | None,
    /,
) -> tuple[dict[str, int], int]:
    """
    """
    if subplot is None:
        return {}, None
    rows, columns = figure._get_subplot_rows_columns()
    num_rows = len(rows)
    num_columns = len(columns)
    if isinstance(subplot, Sequence):
        row = subplot[0]
        column = subplot[1]
        index = row * num_columns + column
        row_column = {"row": row+1, "col": column+1, }
        return row_column, index,
    if isinstance(subplot, int):
        index = subplot
        row = index // num_columns
        column = index % num_columns
        row_column = {"row": row+1, "col": column+1, }
        return row_column, index,
    raise TypeError(f"Invalid subplot type: {type(subplot)}")


@_dm.reference(
    path=("visualization_reporting", "plotly_wrapper.md", ),
    categories={
        "arrange": "Arranging charts",
        "custom": "Customizing charts",
    },
)
class plotly:
    """
················································································

Plotly wrapper
===============

················································································
    """
    #[

    Figure = _pg.Figure
    make_subplots = staticmethod(make_subplots)
    highlight = staticmethod(highlight)
    freeze_span = staticmethod(freeze_span)

    #]

