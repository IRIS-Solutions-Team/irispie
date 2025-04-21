"""
"""


#[

from __future__ import absolute_import

from typing import Literal, Iterable, Sequence
import plotly.graph_objects as _pg
import plotly.subplots as _ps
import plotly.io as _pi
import documark as _dm

from .dates import Period, Frequency

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
#]


__all__ = (
    "plotly", "set_default_plotly_renderer",
    "make_subplots"
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


_EMPTY_SUBPLOT_TITLE = " "


def set_default_plotly_renderer(renderer: str, ) -> None:
    """
    """
    _pi.renderers.default = renderer


_DEFAULT_VERTICAL_SPACING = 0.1
_DEFAULT_HORIZONTAL_SPACING = 0.05

@_dm.reference(category="arrange", )
def make_subplots(
    rows_columns: Iterable[int] | None = None,
    subplot_titles: Sequence[str] | bool | None = True,
    vertical_spacing: float | None = None,
    horizontal_spacing: float | None = None,
    figure_height: int | None = None,
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
        subplot_titles = (_EMPTY_SUBPLOT_TITLE, )*total_num_subplots
    figure = _ps.make_subplots(
        rows=num_rows,
        cols=num_columns,
        subplot_titles=subplot_titles or None,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        **kwargs,
    )
    if figure_height is not None:
        figure.update_layout(height=figure_height, )
    return figure

@_dm.reference(category="custom", )
def highlight(
    figure: _pg.Figure,
    span: Iterable[Period] | None = None,
    subplot_ref: tuple[int, int] | int | None = None,
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
    _, index = resolve_subplot_reference(figure, subplot_ref, )
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


def vline(
    figure,
    vline_period,
    subplot_ref: tuple[int, int] | int | None = None,
    color: str = "rgba(0, 0, 0, 0.5)",
    width: int = None,
) -> None:
    r"""
    """
    date_axis_mode = get_date_axis_mode(figure, )
    row_column, _ = resolve_subplot_reference(figure, subplot_ref, )
    if hasattr(vline_period, "to_plotly_date"):
        vline_period = vline_period.to_plotly_date(mode=date_axis_mode, )
    figure.add_vline(vline_period, **row_column, line={"color": color, "width": width, }, )


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
    row_column, *_ = resolve_subplot_reference(figure, subplot, )
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
        span = tuple(span, )
        start, end = span[0], span[-1]
    else:
        start, end = None, None
    figure_span = figure.layout.xaxis.range
    mode = get_date_axis_mode(figure, )
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
    if isinstance(subplot, Sequence):
        row = subplot[0]
        column = subplot[1]
        index = row * num_columns + column
        row_column = {"row": row+1, "col": column+1, }
        return row_column, index,
    if isinstance(subplot, int):
        if subplot < 0:
            subplot += num_tiles
        index = subplot
        row = index // num_columns
        column = index % num_columns
        row_column = {"row": row+1, "col": column+1, }
        return row_column, index,
    raise TypeError(f"Invalid subplot type: {type(subplot)}")


def update_subplot_title(
    figure: _pg.Figure,
    subplot_ref: tuple[int, int] | int,
    title: str,
) -> None:
    r"""
    """
    row_column, index = resolve_subplot_reference(figure, subplot_ref, )
    annotation = next(figure.select_annotations(index, ), None, )
    if annotation:
        annotation.text = title


def get_date_axis_mode(figure: _pg.Figure, ) -> PlotlyDateAxisModeType | None:
    """
    """
    xaxis = figure.layout.xaxis
    return xaxis.ticklabelmode if xaxis.type == "date" else None


@_dm.reference(
    path=("visualization_reporting", "ez_plotlyper.md", ),
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
    update_subplot_title = staticmethod(update_subplot_title)
    resolve_subplot_reference = staticmethod(resolve_subplot_reference)
    highlight = staticmethod(highlight)
    freeze_span = staticmethod(freeze_span)

    #]

