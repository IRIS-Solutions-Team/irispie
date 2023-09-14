"""
Plotly interface to time series objects
"""


#[
from __future__ import annotations

from types import EllipsisType
import plotly.graph_objects as _pg
import plotly.subplots as _ps

from .. import dates as _dates
#]


builtin_range = range


_COLOR_ORDER = [
  "rgb(  0,   114,   189)",
  "rgb(217,    83,    25)",
  "rgb(237,   177,    32)",
  "rgb(126,    47,   142)",
  "rgb(119,   172,    48)",
  "rgb( 77,   190,   238)",
  "rgb(162,    20,    47)",
]


class Mixin:
    """
    """
    #[
    def plot(
        self,
        *,
        range: Iterable[_dates.Dater] | EllipsisType = ...,
        title: str | None = None,
        legend: Iterable[str] | None = None,
        show_legend: bool | None = None,
        figure = None,
        subplot: tuple[int, int] | int | None = None,
        xline = None,
    ) -> _pg.Figure:
        range = self._resolve_dates(range)
        range = [ t for t in range ]
        data = self.get_data(range)
        num_columns = data.shape[1]
        date_str = [ t.to_plotly_date() for t in range ]
        date_format = range[0].frequency.plotly_format
        figure = _pg.Figure() if figure is None else figure
        subplot = _resolve_subplot(figure, subplot, )
        for i in builtin_range(num_columns):
            figure.add_trace(_pg.Scatter(
                x=date_str, y=data[:, i], name=legend[i] if legend else None,
                line={"color": _COLOR_ORDER[i % len(_COLOR_ORDER)]},
            ), row=subplot[0], col=subplot[1])
        show_legend = show_legend if show_legend is not None else legend is not None
        figure.update_layout(title=title, xaxis={"tickformat": date_format}, showlegend=show_legend, )
        if xline:
            xline = xline.to_plotly_date()
            figure.add_vline(xline)
        return figure
    #]


def _resolve_subplot(
    figure: _pg.Figure,
    subplot: tuple[int, int] | int | None,
    /,
) -> tuple[int, int]:
    """
    """
    if subplot is None:
        return None, None
    if isinstance(subplot, tuple) or isinstance(subplot, list):
        return subplot[0], subplot[1]
    if isinstance(subplot, int):
        position = subplot
        rows, columns = figure._get_subplot_rows_columns()
        num_rows = len(rows)
        num_columns = len(columns)
        row = position // num_columns + 1
        column = position % num_columns + 1
        return row, column

