"""
"""


#[
from __future__ import absolute_import

from typing import (Sequence, Iterable, )
import plotly.graph_objects as _pg
import plotly.subplots as _ps

from . import dates as _dates
from . import pages as _pages
#]


__all__ = (
    "make_subplots",
    "highlight",
    "PlotlyWrapper",
)


_EMPTY_SUBPLOT_TITLE = " "


@_pages.reference(category="arrange", )
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


@_pages.reference(category="custom", )
def highlight(
    figure: _pg.Figure,
    span: Iterable[_dates.Dater],
    /,
    subplot: tuple[int, int] | int | None = None,
    fill_color: str = "rgba(0, 0, 0, 0.15)",
) -> _pg.Figure:
    """
················································································

==Highlight a certain date span in charts==

················································································
    """
    span = tuple(span)
    _, index = resolve_subplot(figure, subplot, )
    xref = f"x{index+1}" if index else "x"
    yref = f"y{index+1} domain" if index else "y domain"
    shape = {
        "type": "rect",
        "xref": xref,
        "x0": span[0].to_plotly_date(position="start", ),
        "x1": span[-1].to_plotly_date(position="end", ),
        "yref": yref,
        "y0": 0,
        "y1": 1,
        "fillcolor": fill_color,
        "line": {"width": 0, },
    }
    figure.add_shape(shape, )


def resolve_subplot(
    figure: _pg.Figure,
    subplot: tuple[int, int] | int | None,
    /,
) -> tuple[tuple[int, int], int]:
    """
    """
    if subplot is None:
        tile = None
        index = None
        return tile, index,
    rows, columns = figure._get_subplot_rows_columns()
    num_rows = len(rows)
    num_columns = len(columns)
    if isinstance(subplot, Sequence):
        row = subplot[0]
        column = subplot[1]
        index = row * num_columns + column
        tile = row, column,
        return tile, index,
    if isinstance(subplot, int):
        index = subplot
        row = index // num_columns
        column = index % num_columns
        tile = row, column,
        return tile, index,
    raise TypeError(f"Invalid subplot type: {type(subplot)}")


@_pages.reference(
    path=("visualization_reporting", "plotly_wrapper.md", ),
    categories={
        "arrange": "Arranging charts",
        "custom": "Customizing charts",
    },
)
class PlotlyWrapper:
    """
················································································

Plotly wrapper
===============

················································································
    """
    make_subplots = staticmethod(make_subplots)
    highlight = staticmethod(highlight)


