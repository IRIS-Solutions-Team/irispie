"""
Plotly interface to time series objects
"""


#[

from __future__ import annotations

import os as _os
import json as _js
import copy as _cp
import plotly.graph_objects as _pg
import plotly.io as _pi
import itertools as _it
import warnings as _wa
import datetime as _dt

from ..dates import Period
from .. import plotly_wrap as _plotly_wrap

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self, Sequence, Iterable, Any, Callable, Literal
    from numbers import Real
    from types import EllipsisType
    from .main import Series

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


_PLOTLY_TRACES_CONSTRUCTOR = {
    "line": _line_plot,
    "bar": _bar_plot,
    "bar_relative": _bar_plot,
    "bar_group": _bar_plot,
    "bar_stack": _bar_plot,
    "bar_overlay": _bar_plot,
}


_BARMODE = {
    "bar": "group",
    "bar_relative": "relative",
    "bar_group": "group",
    "bar_stack": "stack",
    "bar_overlay": "overlay",
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
        span: Iterable[Period] | EllipsisType = ...,
        figure: _pg.Figure | None = None,
        figure_title: str | None = None,
        subplot_title: str | None = None,
        legend: Iterable[str] | None = None,
        show_figure: bool = True,
        show_legend: bool | None = None,
        subplot: tuple[int, int] | int | None = None,
        xline = None,
        type = None,
        chart_type: Literal["line", "bar_stack", "bar_group", ] = "line",
        highlight: Iterable[Period] | None = None,
        bar_norm: str | None = None,
        #
        update_layout: dict[str, Any] | None = None,
        update_traces: tuple(dict[str, Any], ) | dict | None = None,
        #
        freeze_span: bool = False,
        reverse_plot_order: bool = False,
        #
        round: int | None = None,
        round_to: int | None = None,
        #
        return_info: bool = False,
    ) -> dict[str, Any]:
        """
        """
        if type is not None:
            _wa.warn("Use 'chart_type' instead of the deprecated 'type'", )
            chart_type = type if chart_type is None else chart_type

        if round is not None:
            _wa.warn("Use 'round_to' instead of the deprecated 'round'", )
            round_to = round if round_to is None else round_to

        span = tuple(self._resolve_dates(span, ))
        from_until = (span[0], span[-1], )
        frequency = span[0].frequency
        date_format = span[0].frequency.plotly_format

        if figure is None:
            figure = _pg.Figure()

        # Subplot resolution
        row_column, index = _plotly_wrap.resolve_subplot(figure, subplot, )

        if show_legend is None:
            show_legend = legend is not None

        color_cycle = _it.cycle(_COLOR_ORDER)

        update_traces = (update_traces, ) if isinstance(update_traces, dict) else update_traces
        update_traces_cycle = (
            _it.cycle(update_traces) if update_traces
            else _it.repeat({}, )
        )

        transform = _create_transform_function(round_to=round_to, )

        if legend is None:
            legend = _it.repeat(None, )

        traces_constructor = _PLOTLY_TRACES_CONSTRUCTOR[chart_type]

        traces_dates, xaxis_type = _get_traces_dates(self, )

        traces_iterable = _iter_traces(
            self,
            traces_dates,
            from_until,
            traces_constructor,
            transform,
            legend,
            color_cycle,
            update_traces_cycle,
            showlegend=show_legend,
            xhoverformat=date_format,
        )

        if reverse_plot_order:
            traces_iterable = reversed(list(traces_iterable))

        out_traces = ()
        traces_offset = len(tuple(figure.select_traces()))
        for tid, traces_v, in enumerate(traces_iterable, start=traces_offset, ):
            custom_data = (tid, )
            traces_v.update(customdata=custom_data, )
            figure.add_trace(traces_v, **row_column, )
            out_traces += tuple(figure.select_traces({"customdata": custom_data}, ))

        # REFACTOR
        xaxis = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"]["xaxis"])
        yaxis = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"]["yaxis"])
        layout = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"])
        del layout["xaxis"]
        del layout["yaxis"]

        bar_mode = _BARMODE.get(chart_type, figure.layout["barmode"], )
        layout["barmode"] = bar_mode
        layout["barnorm"] = bar_norm

        xaxis["tickformat"] = date_format

        if xaxis_type is not None:
            xaxis["type" ] = xaxis_type

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


def _get_traces_dates(series: Series, /, ) -> tuple[tuple[Period, ...], str]:
    """
    """
    #[
    if not series.span:
        return (), None,
    dates = tuple(i.to_plotly_date() for i in series.span)
    xaxis_type = next(iter(series.span)).plotly_xaxis_type
    return dates, xaxis_type,
   #]


def _iter_traces(
    series: Series,
    traces_dates: Sequence[Period],
    from_until: tuple[Period, Period],
    traces_constructor: Callable,
    transform: Callable,
    legends: Iterable[str] | None,
    colors: Iterable[str],
    custom_updates: Iterable[dict],
    **kwargs,
) -> Any:
    """
    """
    #[
    zipped = zip(
        series.iter_own_data_variants_from_until(from_until=from_until, ),
        legends,
        colors,
        custom_updates,
    )
    for data_v, legend_v, color_v, update_v in zipped:
        yield traces_constructor(
            color=color_v,
            x=traces_dates,
            y=tuple(transform(i) for i in data_v),
            name=legend_v,
            **update_v,
            **kwargs,
        )
    #]


def _create_transform_function(
    round_to: int | None = None,
) -> Callable[[Real], Real]:
    """
    """
    #[
    def no_transform(x: Real, ) -> Real:
        return x

    def decorate_round(func: Callable, ) -> Callable:
        def wrapper(x: Real, ) -> Real:
            return round(func(x), round_to, )
        return wrapper

    transform = no_transform

    if round_to is not None:
        transform = decorate_round(transform, )

    return transform
    #]

