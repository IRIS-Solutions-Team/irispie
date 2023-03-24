"""
Plotly interface to time series objects
"""


#[
from __future__ import annotations

import plotly.graph_objects as pg
import plotly.subplots as ps
from types import (EllipsisType, )

from ..dataman import (dates as da_, )
#]


builtin_range = range


__all__ = [
    "make_subplots",
]

_COLOR_ORDER = [
  "rgb(  0,   114,   189)",
  "rgb(217,    83,    25)",
  "rgb(237,   177,    32)",
  "rgb(126,    47,   142)",
  "rgb(119,   172,    48)",
  "rgb( 77,   190,   238)",
  "rgb(162,    20,    47)",
]

class PlotlyMixin:
    """
    """
    #[
    def plot(
        self,
        /,
        range: Iterable[da_.Dater] | EllipsisType = ...,
        title: str | None = None,
        legend: Iterable[str] | None = None,
        show_legend: bool | None = None,
        figure = None,
        subplot: tuple[int|None, int|None] = (None, None),
        xline = None,
    ):
        range = self._resolve_dates(range)
        range = [ t for t in range ]
        data = self.get_data(range)
        num_columns = data.shape[1]
        date_str = [ t.to_plotly_date() for t in range ]
        date_format = range[0].frequency.plotly_format
        figure = pg.Figure() if figure is None else figure
        for i in builtin_range(num_columns):
            figure.add_trace(pg.Scatter(
                x=date_str, y=data[:, i], name=legend[i] if legend else None,
                line={"color": _COLOR_ORDER[i % len(_COLOR_ORDER)]},
            ), row=subplot[0], col=subplot[1])
        show_legend = show_legend if show_legend is not None else legend is not None
        figure.update_layout(title=title, xaxis={"tickformat": date_format}, showlegend=show_legend)
        if xline:
            xline = xline.to_plotly_date()
            figure.add_vline(xline)
        return figure
    #]


make_subplots = ps.make_subplots

