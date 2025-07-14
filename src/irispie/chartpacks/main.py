"""
Time series chartpacks
"""


#[

from __future__ import annotations

import functools as _ft
import re as _re
import documark as _dm
import plotly.graph_objects as _pg
from collections.abc import Sequence

from ..conveniences import descriptions as _descriptions
from ..conveniences import copies as _copies
from .. import ez_plotly as _ez_plotly
from .. import dates as _dates
from ..dates import Period
from ..databoxes import main as _databoxes

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self, Callable
    from collections.abc import Iterator, Iterable
    from types import EllipsisType

#]


__all__ =  (
    "Chartpack",
)


_FIGURE_SETTINGS = {
    "legend": None,
    "show_legend": None,
    "highlight": None,
    "tiles": None,
    "shared_xaxes": False,
    "figure_height": None,
    "vertical_spacing": None,
}
_FIGURE_SETTINGS_KEYS = tuple(_FIGURE_SETTINGS.keys(), )


_CHART_SETTINGS = {
    "transform": None,
    "transforms": {},
    "reverse_plot_order": False,
    "span": ...,
    "chart_type": "line",
    "update_traces": None,
}
_CHART_SETTINGS_KEYS = tuple(_CHART_SETTINGS.keys(), )


_CHART_INPUT_STRING_PATTERN = _re.compile(
    r"^(?:(?P<title>.*?):)?(?P<expression>.*?)(?:\[(?P<transform>.*?)\])?$"
)


@_dm.reference(
    path=("visualization_reporting", "chartpacks.md", ),
    categories={
        "constructor": "Creating new chartpacks",
        "information": "Getting information about chartpacks",
        "property": None,
        "plot": "Plotting chartpacks",
        "add": "Adding figures and charts to chartpacks",
    }
)
class Chartpack(
    _descriptions.DescriptionMixin,
    _copies.Mixin,
):
    """
················································································

Chartpacks
===========

················································································
    """
    #[

    __slots__ = (
        "title",
        "_figures",
        "_figure_settings",
        "_chart_settings",
        "__description__",
    )

    def __init__(
        self,
        title: str = "",
        **kwargs,
    ) -> None:
        """
        """
        self.title = title
        self._figures = None
        self.__description__ = None
        #
        self._figure_settings = {
            n: kwargs[n]
            for n in _FIGURE_SETTINGS_KEYS
            if n in kwargs
        }
        #
        self._chart_settings = {
            n: kwargs[n]
            for n in _CHART_SETTINGS_KEYS
            if n in kwargs
        }

    @classmethod
    @_dm.reference(category="constructor", call_name="Chartpack", )
    def _constructor_doc():
        """
················································································

==Create a new chartpack==

```
self = Chartpack(
    title="",
    span=...,
    tiles=None,
    transforms=None,
    highlight=None,
    legend=None,
    show_legend=None,
    reverse_plot_order=False,
)
```

### Input arguments ###


???+ input "title"
    The title of the chartpack, used as a basis for creating a caption
    shown at the top of each figure.

???+ input "span"
    The date span on which the time series will be plotted.

???+ input "tiles"
    The number of rows and columns of the figure grid. If input "None", the number of
    rows and columns will be determined automatically.

???+ input "transforms"
    A dictionary of functions that will be applied to the input data before
    plotting.

???+ input "highlight"
    A date span that will be highlighted in the charts.

???+ input "legend"
    A list of strings that will be used as the legend for the charts.

???+ input "show_legend"
    Show the legend in the figure. If `None`, the legend will be shown if `legend` is
    not `None` and non-empty.

???+ input "reverse_plot_order"
    If `True`, the order of plotting the individual time series within each
    chart will be reversed.


### Returns


???+ returns "self"

    A new empty `Chartpack` object.

················································································
        """
        pass

    def set_span(
        self,
        span: Iterable[Period] | EllipsisType,
    ) -> None:
        r"""
        """
        for f in self._figures:
            f.set_span(span, )

    @_dm.reference(category="plot", )
    def plot(
        self,
        input_db: _databoxes.Databox,
        show_figures: bool = True,
        return_info: bool = False,
        **kwargs,
    ) -> dict[str, Any] | None:
        """
················································································

==Plot the chartpack==

················································································
        """
        figures = tuple(
            figure.plot(input_db, **self.__dict__, )
            for figure in self
        )
        if show_figures:
            for f in figures: f.show()
        if return_info:
            return {
                "figures": figures,
            }

    def format_figure_titles(
        self,
        **kwargs,
    ) -> None:
        """
        """
        for figure in self:
            figure.format_figure_title(**kwargs, )

    @property
    @_dm.reference(category="property", )
    def num_figures(self, ) -> int:
        """==Total number of figures in the chartpack=="""
        return len(self._figures, ) if self._figures else 0

    def _add_figure(self, figure: _Figure, ) -> None:
        """
        """
        if not self._figures:
            self._figures = []
        self._figures.append(figure, )

    @_dm.reference(category="add", )
    def add_figure(self, figure_string: str, **kwargs, ) -> None:
        """
················································································

==Add a new figure to the chartpack==

················································································
        """
        new_figure = _Figure.from_string(
            figure_string,
            figure_settings_cascaded=self._figure_settings,
            chart_settings_cascaded=self._chart_settings,
            **kwargs,
        )
        self._add_figure(new_figure, )
        return new_figure

    @staticmethod
    @_dm.reference(category="add", call_name="add_chart", )
    def _add_chart_doc():
        """
················································································

==Add a new chart to an existing figure in the chartpack==

················································································
        """
        pass

    def __str__(self, /, ) -> str:
        """
        """
        return repr(self, )

    def __repr__(self, /, ) -> str:
        """
        """
        return _tree_repr(self, )

    def _one_liner(self, /, ) -> str:
        return f"Chartpack({self.title!r}, )"

    def __getitem__(self, item: str | int, ) -> _Figure:
        """
        """
        if isinstance(item, int, ):
            return self._figures[item]
        if isinstance(item, str, ):
            return next(f for f in self._figures if f.title == item)

    def __iter__(self, ) -> Iterator[_Figure]:
        """
        """
        return iter(self._figures, )

    #]


class _Figure:
    """
    """
    #[

    __slots__ = (
        "title",
        "_charts",
        "_chart_settings",
    ) + _FIGURE_SETTINGS_KEYS

    def __init__(
        self,
        title: str | None = None,
    ) -> None:
        """
        """
        for n in self.__slots__:
            setattr(self, n, None, )
        self.title = title
        self._charts = []
        self._chart_settings = {}
        if self.show_legend is None:
            self.show_legend = self.legend is not None and bool(self.legend)

    @classmethod
    def from_string(
        klass,
        input_string: str,
        figure_settings_cascaded: dict[str, Any] | None = None,
        chart_settings_cascaded: dict[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass(title=input_string, )
        figure_settings_cascaded = figure_settings_cascaded or {}
        chart_settings_cascaded = chart_settings_cascaded or {}
        #
        for key in _FIGURE_SETTINGS_KEYS:
            if key in kwargs:
                setattr(self, key, kwargs[key], )
                continue
            if key in figure_settings_cascaded:
                setattr(self, key, figure_settings_cascaded[key], )
                continue
            else:
                setattr(self, key, _FIGURE_SETTINGS[key], )
                continue
        #
        for key in _CHART_SETTINGS_KEYS:
            if key in kwargs:
                self._chart_settings[key] = kwargs[key]
                continue
            if key in chart_settings_cascaded:
                self._chart_settings[key] = chart_settings_cascaded[key]
                continue
        #
        return self

    @property
    def num_charts(self, ) -> int:
        return len(self._charts, )

    def set_span(
        self,
        span: Iterable[Period] | EllipsisType,
    ) -> None:
        r"""
        """
        for ch in self._charts:
            ch.span = span

    def plot(
        self,
        input_db: _databoxes.Databox,
        shared_xaxes: bool = False,
        **kwargs,
    ) -> _pg.Figure:
        r"""
        """
        tiles = _resolve_tiles(self.tiles, self.num_charts, )
        figure = _ez_plotly.make_subplots(
            rows_columns=(tiles[0], tiles[1], ),
            subplot_titles=tuple(chart.caption for chart in self),
            shared_xaxes=self.shared_xaxes,
            figure_height=self.figure_height,
            vertical_spacing=self.vertical_spacing,
        )
        for i, chart in enumerate(self, ):
            chart.plot(
                input_db, figure, i,
                legend=self.legend,
                include_in_legend=(i == 0),
            )
            if self.highlight is not None:
                _ez_plotly.highlight(figure, self.highlight, subplot=i, )
        #
        figure.update_layout(
            title={"text": self.title, },
            showlegend=self.show_legend,
        )
        #
        return figure

    def format_figure_title(
        self,
        **kwargs,
    ) -> None:
        """
        """
        self.title = self.title.format(**kwargs, )

    def __str__(self, /, ) -> str:
        """
        """
        return repr(self, )

    def __repr__(self, /, ) -> str:
        """
        """
        return _tree_repr(self, )

    def _one_liner(self, /, ) -> str:
        return f"Figure({self.title!r}, )"

    def _add_chart(self, chart: _Chart, ) -> None:
        """
        """
        self._charts.append(chart, )

    def add_charts(self, chart_strings: Iterable[str], **kwargs, ) -> None:
        """
        """
        for chart_string in chart_strings:
            self.add_chart(chart_string, **kwargs, )

    def add_chart(self, chart_string: str, **kwargs, ) -> None:
        """
        """
        chart = _Chart.from_string(
            chart_string,
            chart_settings_cascaded=self._chart_settings,
            **kwargs,
        )
        self._add_chart(chart, )

    def __iter__(self, /, ) -> Iterator[_Chart]:
        """
        """
        return iter(self._charts, )

    def __enter__(self, /, ) -> Self:
        return self

    def __exit__(self, *args, **kwargs, ) -> None:
        pass

    #]


class _Chart:
    """
    """
    #[

    __slots__ = (
        "title",
        "expression",
        "transform"
    ) + _CHART_SETTINGS_KEYS

    def __init__(
        self,
        expression: str,
        #
        title: str | None = None,
        transform: str | Callable | None = None,
        chart_settings_cascaded: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        """
        for n in self.__slots__:
            setattr(self, n, None, )
        self.expression = (expression or "").strip()
        self.title = (title or "").strip() or self.expression
        self.transform = (transform or "").strip()
        chart_settings_cascaded = chart_settings_cascaded or {}
        for key in _CHART_SETTINGS_KEYS:
            if getattr(self, key, None) is not None:
                continue
            elif key in kwargs:
                setattr(self, key, kwargs[key], )
                continue
            elif key in chart_settings_cascaded:
                setattr(self, key, chart_settings_cascaded[key], )
                continue
            else:
                setattr(self, key, _CHART_SETTINGS[key], )
                continue

    @classmethod
    def from_string(
        klass,
        input_string: str,
        chart_settings_cascaded: dict[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        match = _CHART_INPUT_STRING_PATTERN.match(input_string, )
        if not match:
            raise ValueError(f"Invalid input string: {input_string!r}")
        match_dict = match.groupdict()
        return klass(**match.groupdict(), chart_settings_cascaded=chart_settings_cascaded, **kwargs, )

    @property
    def caption(self, ) -> str:
        """Caption shown at the top of the chart"""
        return self.title or (self.expression + f" [{self.transform}]" if self.transform else "")

    def plot(
        self,
        input_db: _databoxes.Databox,
        figure: _pg.Figure,
        index: int,
        legend: Iterable[str, ...] | None,
        include_in_legend: bool,
    ) -> None:
        """
        """
        series_to_plot = input_db.evaluate_expression(self.expression, )
        series_to_plot = self._apply_transform(series_to_plot, )
        series_to_plot.plot(
            figure=figure,
            subplot=index,
            span=self.span,
            show_figure=False,
            freeze_span=True,
            legend=legend,
            include_in_legend=include_in_legend,
            show_legend=False,
            reverse_plot_order=self.reverse_plot_order,
            chart_type=self.chart_type,
            update_traces=self.update_traces,
        )

    def _apply_transform(self, x, ):
        """
        """
        if self.transform and self.transforms and self.transform in self.transforms:
            func = self.transforms[self.transform]
            return func(x, )
        if self.transform and isinstance(self.transform, str):
            func = eval("lambda x:" + self.transform, )
            return func(x, )
        return x

    def __str__(self, /, ) -> str:
        """
        """
        return repr(self, )

    def __repr__(self, /, ) -> str:
        """
        """
        return _tree_repr(self, )

    def _one_liner(self, /, ) -> str:
        return f"Chart({self.title!r}, {self.expression!r}, {self.transform!r}, )"

    def __iter__(self, ) -> Iterable:
        yield from ()

    #]


def _add_strings(
    self,
    input_strings: Iterable[str] | str,
    call: Callable,
    **kwargs,
) -> None:
    """
    """
    #[
    if isinstance(input_strings, str, ):
        input_strings = (input_strings, )
    for s in input_strings:
        call(s, )
    #]


def _resolve_tiles(tiles, num_charts, ) -> tuple[int, int]:
    """
    """
    #[
    if tiles is None:
        return _ez_plotly.auto_tiles(num_charts, )
    if isinstance(tiles, int, ):
        return _ez_plotly.auto_tiles(tiles, )
    if isinstance(tiles, Sequence, ):
        return tiles[:2]
    raise TypeError(f"Invalid tiles: {tiles!r}")
    #]


_TREE_INDENT = "    |"
_TREE_BULLET = "--"


def _tree_repr(self, /, indent: tuple[str] = (), is_last: bool = False, ) -> str:
    """
    """
    #[
    child_indent = _create_child_indent(indent, is_last, )
    child_repr_func = _ft.partial(_tree_repr, indent=child_indent, )
    child_objects = tuple(c for c in self)
    num_child_objects = len(child_objects)
    child_tree_repr = tuple(
        _tree_repr(c, indent=child_indent, is_last=(i+1 == len(child_objects)), )
        for i, c in enumerate(child_objects)
    )
    indent_str = "".join(indent, )
    header = indent_str + (_TREE_BULLET if indent_str else "") + self._one_liner()
    return "\n".join((header, *child_tree_repr))
    #]


def _create_child_indent(indent: tuple[str], is_last: bool, ) -> str:
    """
    """
    #[
    if is_last and indent:
        indent = indent[:-1] + (" "*len(_TREE_INDENT), )
    return indent + (_TREE_INDENT, )
    #]

