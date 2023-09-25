"""
Plotly interface to time series objects
"""


#[
from __future__ import annotations

from types import EllipsisType
import os as _os
import json as _js
import copy as _cp
import plotly.graph_objects as _pg
import plotly.subplots as _ps

from .. import dates as _dates
#]


_PLOTLY_STYLES_FOLDER = _os.path.join(_os.path.dirname(__file__), "plotly_styles")

_PLOTLY_STYLES = {
    "layouts": {},
}
with open(_os.path.join(_PLOTLY_STYLES_FOLDER, "plain_layout.json", ), "rt", ) as fid:
    _PLOTLY_STYLES["layouts"]["plain"] = _js.load(fid, )


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
        axis_id = f"{subplot+1}" if subplot else ""
        subplot = _resolve_subplot(figure, subplot, )
        show_legend = show_legend if show_legend is not None else legend is not None
        for i in builtin_range(num_columns):
            figure.add_trace(
                _pg.Scatter(
                    x=date_str, y=data[:, i], name=legend[i] if legend else None,
                    line={"color": _COLOR_ORDER[i % len(_COLOR_ORDER)]},
                    #marker={"color": _COLOR_ORDER[i % len(_COLOR_ORDER)]},
                    showlegend=show_legend, xhoverformat="%Y-%q",
                    xaxis=f"x{axis_id}", yaxis=f"y{axis_id}",
                ),
                row=subplot[0], col=subplot[1]
            )

        layout = _cp.deepcopy(_PLOTLY_STYLES["layouts"]["plain"])
        layout[f"xaxis{axis_id}"] = layout["xaxis"]
        layout[f"yaxis{axis_id}"] = layout["yaxis"]
        if axis_id:
            del layout["xaxis"]
            del layout["yaxis"]
        layout[f"xaxis{axis_id}"]["tickformat"] = date_format
        layout["title"]["text" ] = title
        figure.update_layout(layout, )

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

