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
import plotly.io as _pi
import itertools as _it
import warnings as _wa
import datetime as _dt

from .. import dates as _dates
from .. import plotly_wrap as _plotly_wrap
#]


# _pi.renderers.default = "browser"


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


_PLOTLY_TRACES_FUNC = {
    "line": _line_plot,
    "bar": _bar_plot,
    "bar_relative": _bar_plot,
    "bar_group": _bar_plot,
}


_BARMODE = {
    "bar": "group",
    "bar_relative": "relative",
    "bar_group": "group",
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
        update_layout: dict[str, Any] | None = None,
        show_figure: bool = True,
        show_legend: bool | None = None,
        subplot: tuple[int, int] | int | None = None,
        xline = None,
        type = None,
        chart_type: Literal["line", "bar_stack", "bar_group", ] = "line",
        highlight: Iterable[Period] | None = None,
        update_traces: tuple(dict[str, Any], ) | dict | None = None,
        freeze_span: bool = False,
        reverse_plot_order: bool = False,
        round: int | None = None,
        return_info: bool = False,
    ) -> dict[str, Any]:
        """
        """
        if type is not None:
            _wa.warn("Use 'chart_type' instead of the deprecated 'type'", DeprecationWarning, )
            chart_type = type
        #
        span = self._resolve_dates(span, )
        span = [ t for t in span ]
        from_until = (span[0], span[-1], )
        #
        frequency = span[0].frequency
        # data = self.get_data(span, )
        date_strings = [ t.to_plotly_date() for t in span ]
        from_to_strings = [ date_strings[0], date_strings[-1], ]
        #
        date_format = span[0].frequency.plotly_format
        figure = _pg.Figure() if figure is None else figure

        # Subplot resolution
        row_column, index = _plotly_wrap.resolve_subplot(figure, subplot, )

        if show_legend is None:
            show_legend = legend is not None

        color_cycle = _it.cycle(_COLOR_ORDER)

        update_traces = (update_traces, ) if isinstance(update_traces, dict) else update_traces
        update_traces_cycle = (
            _it.cycle(update_traces)
            if update_traces
            else _it.repeat(None, )
        )

        if legend is None:
            legend = _it.repeat(None, )

        zipped = zip(
            self.iter_own_data_variants_from_until(from_until, ),
            legend,
            color_cycle,
            update_traces_cycle,
        )

        if reverse_plot_order:
            zipped = reversed(list(zipped))

        out_traces = ()
        traces_offset = len(tuple(figure.select_traces()))
        for tid, (data_v, legend_v, color, update_traces_v) in enumerate(zipped, start=traces_offset, ):
            customdata = (tid, )
            traces_settings = {
                "x": date_strings,
                "y": data_v if round is None else data_v.round(round, ),
                "name": legend_v,
                "showlegend": show_legend,
                "xhoverformat": date_format,
                "customdata": customdata,
            }
            traces_settings.update(update_traces_v or {}, )
            traces_object = _PLOTLY_TRACES_FUNC[chart_type](color, **traces_settings, )
            figure.add_trace(traces_object, **row_column, )
            out_traces += tuple(figure.select_traces({"customdata": customdata}, ))

        # REFACTOR
        xaxis = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"]["xaxis"])
        yaxis = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"]["yaxis"])
        layout = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"])
        del layout["xaxis"]
        del layout["yaxis"]


        if chart_type:
            bar_mode = _BARMODE.get(chart_type, None)
        if not bar_mode:
            bar_mode = figure.layout["barmode"]
        layout["barmode"] = bar_mode

        xaxis["tickformat"] = date_format
        xaxis["ticklabelmode"] = "period"

        figure.update_xaxes(xaxis, **row_column, )
        figure.update_yaxes(yaxis, **row_column, )
        figure.update_layout(layout or {}, )

        if freeze_span:
            _plotly_wrap.freeze_span(figure, span, subplot, )

        if figure_title is not None:
            figure.update_layout({"title.text": figure_title, }, )

        if update_layout is not None:
            figure.update_layout(update_layout, )

        if subplot_title:
            _update_subplot_title(figure, subplot_title, index, )

        if xline:
            xline = (
                xline.to_plotly_date()
                if hasattr(xline, "to_plotly_date")
                else xline
            )
            figure.add_vline(xline, )

        if highlight is not None:
            _plotly_wrap.highlight(figure, highlight, )

        if show_figure:
            figure.show()

        if return_info:
            out_info = {
                "figure": figure,
                "traces": out_traces,
            }
            return out_info
        else:
            return

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

