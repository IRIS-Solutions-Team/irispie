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
from .. import plotly_wrap as _plotly_wrap
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


#_COLOR_ORDER = [
#    "rgb(  0,   114,   189)",
#    "rgb(217,    83,    25)",
#    "rgb(237,   177,    32)",
#    "rgb(126,    47,   142)",
#    "rgb(119,   172,    48)",
#    "rgb( 77,   190,   238)",
#    "rgb(162,    20,    47)",
#]


_COLOR_ORDER = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class Inlay:
    """
    """
    #[

    def plot(
        self,
        *,
        span: Iterable[_dates.Dater] | EllipsisType = ...,
        figure: _pg.Figure | None = None,
        figure_title: str | None = None,
        subplot_title: str | None = None,
        legend: Iterable[str] | None = None,
        show_figure: bool = True,
        show_legend: bool | None = None,
        subplot: tuple[int, int] | int | None = None,
        xline = None,
        type: Literal["line", "bar"] = "line",
        traces: tuple(dict[str, Any], ) | None = None,
        freeze_span: bool = False,
        reverse_plot_order: bool = False,
        round=None,
    ) -> _pg.Figure:
        """
        """
        span = self._resolve_dates(span, )
        span = [ t for t in span ]
        frequency = span[0].frequency
        num_variants = self.num_variants
        data = self.get_data(span, )
        date_strings = [ t.to_plotly_date() for t in span ]
        date_format = span[0].frequency.plotly_format
        figure = _pg.Figure() if figure is None else figure
        tile, index = _plotly_wrap.resolve_subplot(figure, subplot, )
        row, column = (tile[0]+1, tile[1]+1, ) if tile is not None else (None, None, )

        if show_legend is None:
            show_legend = legend is not None

        traces = (traces, ) if isinstance(traces, dict) else traces
        color_cycle = _it.cycle(_COLOR_ORDER)
        traces_cycle = _it.cycle(traces or ({}, ))

        loop = zip(range(num_variants, ), color_cycle, traces_cycle, )
        if reverse_plot_order:
            loop = reversed(list(loop))

        for i, color, ts in loop:
            traces_settings = {
                "x": date_strings,
                "y": data[:, i] if round is None else data[:, i].round(round, ),
                "name": legend[i] if legend else None,
                "showlegend": show_legend,
                "xhoverformat": date_format,
            }
            traces_settings |= ts or {}
            traces_object = _PLOTLY_TRACES_FACTORY[type](color, **traces_settings, )
            figure.add_trace(traces_object, row=row, col=column, )

        # REFACTOR
        xaxis = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"]["xaxis"])
        yaxis = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"]["yaxis"])
        layout = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"])
        del layout["xaxis"]
        del layout["yaxis"]

        xaxis["tickformat"] = date_format
        xaxis["ticklabelmode"] = "period"
        if freeze_span:
            xaxis["range"] = [date_strings[0], date_strings[-1], ]
            xaxis["autorange"] = False

        figure.update_xaxes(xaxis, row=row, col=column, )
        figure.update_yaxes(yaxis, row=row, col=column, )
        if figure_title is not None:
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

