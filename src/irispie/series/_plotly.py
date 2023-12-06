"""
Plotly interface to time series objects
"""


#[
from __future__ import annotations

from typing import (Self, Sequence, Iterable, Any, )
from types import EllipsisType
import os as _os
import json as _js
import copy as _cp
import plotly.graph_objects as _pg
import itertools as _it

from .. import dates as _dates
#]


_PLOTLY_STYLES_FOLDER = _os.path.join(_os.path.dirname(__file__), "plotly_styles")
_PLOTLY_STYLES = {
    "layouts": {},
}
with open(_os.path.join(_PLOTLY_STYLES_FOLDER, "plain_layout.json", ), "rt", ) as fid:
    _PLOTLY_STYLES["layouts"]["plain"] = _js.load(fid, )


def _line_plot(color: str, **settings) -> _pg.Scatter:
    """
    """
    settings = {"mode": "lines+markers", } | settings
    return _pg.Scatter(line_color=color, **settings, )


def _bar_plot(color: str, **settings) -> _pg.Bar:
    """
    """
    return _pg.Bar(marker_color=color, **settings, )


_PLOTLY_TRACES_FACTORY = {
    "line": _line_plot,
    "bar": _bar_plot,
}


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
        span: Iterable[_dates.Dater] | EllipsisType = ...,
        figure_title: str | None = None,
        subplot_title: str | None = None,
        legend: Iterable[str] | None = None,
        show_figure: bool = False,
        show_legend: bool | None = None,
        figure = None,
        subplot: tuple[int, int] | int | None = None,
        xline = None,
        type: Literal["line", "bar"] = "line",
        traces: tuple(dict[str, Any], ) | None = None,
        layout: dict[str, Any] | None = None,
    ) -> _pg.Figure:
        """
        """
        span = self._resolve_dates(span, )
        span = [ t for t in span ]
        frequency = span[0].frequency
        data = self.get_data(span, )
        num_variants = data.shape[1]
        date_str = [ t.to_plotly_date() for t in span ]
        date_format = span[0].frequency.plotly_format
        figure = _pg.Figure() if figure is None else figure
        tile, index = _resolve_subplot(figure, subplot, )
        show_legend = show_legend if show_legend is not None else legend is not None

        minor_dtick_string = None
        if frequency.is_regular:
            minor_dtick_months = min(12//frequency, 1, )
            minor_dtick_string = f"M{minor_dtick_months}"

        traces = (traces, ) if isinstance(traces, dict) else traces
        color_cycle = _it.cycle(_COLOR_ORDER)
        traces_cycle = _it.cycle(traces or ({}, ))

        for i, c, ts in zip(range(num_variants, ), color_cycle, traces_cycle, ):
            traces_settings = {
                "x": date_str,
                "y": data[:, i],
                "name": legend[i] if legend else None,
                "showlegend": show_legend,
                "xhoverformat": date_format,
            }
            traces_settings |= ts or {}
            traces_object = _PLOTLY_TRACES_FACTORY[type](c, **traces_settings, )
            figure.add_trace(traces_object, row=tile[0]+1, col=tile[1]+1, )

        # REFACTOR
        xaxis = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"]["xaxis"])
        yaxis = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"]["yaxis"])
        layout = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"])
        del layout["xaxis"]
        del layout["yaxis"]

        xaxis["tickformat"] = date_format
        xaxis["ticklabelmode"] = "period"
        xaxis["minor"] = {"showgrid": True, "dtick": minor_dtick_string, }
        figure.update_xaxes(xaxis, row=tile[0]+1, col=tile[1]+1, )
        figure.update_layout(title={"text": figure_title}, )
        figure.update_layout(layout or {}, )

        if subplot_title:
            _update_subplot_title(figure, subplot_title, index, )

        if xline:
            xline = (
                xline.to_plotly_date()
                if hasattr(xline, "to_plotly_date")
                else xline
            )
            figure.add_vline(xline, )

        if show_figure:
            figure.show()

        return figure

    #]


def _resolve_subplot(
    figure: _pg.Figure,
    subplot: tuple[int, int] | int | None,
    /,
) -> tuple[tuple[int, int], int]:
    """
    """
    rows, columns = figure._get_subplot_rows_columns()
    num_rows = len(rows)
    num_columns = len(columns)
    if subplot is None:
        tile = None
        index = None
    elif isinstance(subplot, Sequence):
        row = subplot[0]
        column = subplot[1]
        index = row * num_columns + column
        tile = row, column,
    elif isinstance(subplot, int):
        index = subplot
        row = index // num_columns
        column = index % num_columns
        tile = row, column,
    else:
        raise TypeError(f"Invalid subplot type: {type(subplot)}")
    return tile, index


def _update_subplot_title(
    figure: _pg.Figure,
    subplot_title: str,
    index: int,
    /,
) -> None:
    """
    """
    annotation = next(figure.select_annotations(index, ), None, )
    if annotation:
        annotation.text = subplot_title

