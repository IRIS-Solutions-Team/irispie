"""
Time series chartpacks
"""


#[
from __future__ import annotations

from typing import (Self, Iterator, Iterable, Sequence, Callable, )
from types import (EllipsisType, )
import functools as _ft
import re as _re
import math as _ma
import plotly.graph_objects as _pg

from ..conveniences import descriptions as _descriptions
from ..conveniences import copies as _copies
from .. import plotly_wrap as _plotly_wrap
from .. import dates as _dates
from .. import pages as _pages
from ..databoxes import main as _databoxes
#]


__all__ =  (
    "Chartpack",
)


_FIGURE_SETTINGS = {
    "legend": None,
    "highlight": None,
    "tiles": None,
    "shared_xaxes": False,
}
_FIGURE_SETTINGS_KEYS = tuple(_FIGURE_SETTINGS.keys(), )


_CHART_SETTINGS = {
    "transforms": {},
    "reverse_plot_order": False,
    "span": ...,
    "chart_type": "line",
}
_CHART_SETTINGS_KEYS = tuple(_CHART_SETTINGS.keys(), )


_CHART_INPUT_STRING_PATTERN = _re.compile(
    r"^(?:(?P<title>.*?):)?(?P<expression>.*?)(?:\[(?P<transform>.*?)\])?$"
)


@_pages.reference(
    path=("visualization_reporting", "chartpacks.md", ),
    categories={
        "constructor": "Creating new chartpacks",
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
        "_description",
        "_figure_settings",
        "_chart_settings",
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
        self._description = None
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
    @_pages.reference(category="constructor", call_name="Chartpack", )
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

???+ input "reverse_plot_order"
    If `True`, the order of plotting the individual time series within each
    chart will be reversed.


### Returns


???+ returns "self"

    A new empty `Chartpack` object.

················································································
        """
        pass

    @_pages.reference(category="plot", )
    def plot(
        self,
        input_db: _databoxes.Databox,
        show_figures: bool = True,
        **kwargs,
    ) -> tuple[_pg.Figure, ...]:
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
        return figures

    def format_figure_titles(
        self,
        **kwargs,
    ) -> None:
        """
        """
        for figure in self:
            figure.format_figure_title(**kwargs, )

    @property
    @_pages.reference(category="property", )
    def num_figures(self, ) -> int:
        """==Total number of figures in the chartpack=="""
        return len(self._figures, ) if self._figures else 0

    def _add_figure(self, figure: _Figure, ) -> None:
        """
        """
        if not self._figures:
            self._figures = []
        self._figures.append(figure, )

    @_pages.reference(category="add", )
    def add_figure(self, figure_string: str, **kwargs, ) -> None:
        """
················································································

==Add a new figure to the chartpack==

················································································
        """
        new_figure = _Figure.from_string(
            figure_string,
            kwargs=kwargs,
            figure_settings_cascaded=self._figure_settings,
            chart_settings_cascaded=self._chart_settings,
        )
        self._add_figure(new_figure, )
        return new_figure

    @staticmethod
    @_pages.reference(category="add", call_name="add_chart", )
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

    @classmethod
    def from_string(
        klass,
        input_string: str,
        /,
        kwargs: dict[str, Any],
        figure_settings_cascaded: dict[str, Any],
        chart_settings_cascaded: dict[str, Any],
    ) -> Self:
        """
        """
        self = klass(title=input_string, )
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

    def plot(
        self,
        input_db: _databoxes.Databox,
        /,
        shared_xaxes: bool = False,
        **kwargs,
    ) -> None:
        """
        """
        tiles = _resolve_tiles(self.tiles, self.num_charts, )
        figure = _plotly_wrap.make_subplots(
            rows=tiles[0],
            columns=tiles[1],
            subplot_titles=tuple(chart.caption for chart in self),
            shared_xaxes=self.shared_xaxes,
        )
        for i, chart in enumerate(self, ):
            chart.plot(
                input_db, figure, i,
                legend=self.legend if i == 0 else None,
            )
            if self.highlight is not None:
                _plotly_wrap.highlight(figure, self.highlight, subplot=i, )
        figure.update_layout(title={"text": self.title, }, )
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
            kwargs=kwargs,
            chart_settings_cascaded=self._chart_settings,
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
        title: str | None = None,
        transform: str | Callable | None = None,
    ) -> None:
        """
        """
        for n in self.__slots__:
            setattr(self, n, None, )
        self.expression = (expression or "").strip()
        self.title = (title or "").strip() or self.expression
        self.transform = (transform or "").strip()

    @classmethod
    def from_string(
        klass,
        input_string: str,
        kwargs: dict[str, Any],
        chart_settings_cascaded: dict[str, Any],
    ) -> Self:
        """
        """
        match = _CHART_INPUT_STRING_PATTERN.match(input_string, )
        if not match:
            raise ValueError(f"Invalid input string: {input_string!r}")
        self = klass(**match.groupdict(), )
        #
        for key in _CHART_SETTINGS_KEYS:
            if key in kwargs:
                setattr(self, key, kwargs[key], )
                continue
            elif key in chart_settings_cascaded:
                setattr(self, key, chart_settings_cascaded[key], )
                continue
            else:
                setattr(self, key, _CHART_SETTINGS[key], )
                continue
        #
        return self

    @property
    def caption(self, ) -> str:
        """Caption shown at the top of the chart"""
        return self.title or (self.expression + f" [{self.transform}]" if self.transform else "")

    def plot(
        self,
        input_db: _databoxes.Databox,
        figure: _pg.Figure,
        index: int,
        legend: Iterable[str, ...] | None = None,
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
            reverse_plot_order=self.reverse_plot_order,
            chart_type=self.chart_type,
        )

    def _apply_transform(self, x, ):
        """
        """
        if self.transform:
            func = self.transforms[self.transform]
        else:
            func = None
        return func(x) if func else x

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

    def __iter__(self, ) -> NoReturn:
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
        return _auto_tiles(num_charts, )
    if isinstance(tiles, int, ):
        return _auto_tiles(tiles, )
    if isinstance(tiles, Sequence, ):
        return tiles[:2]
    raise TypeError(f"Invalid tiles: {tiles!r}")
    #]


def _auto_tiles(num_charts, ) -> tuple[int, int]:
    """
    """
    #[
    n = _ma.ceil(_ma.sqrt(num_charts, ), )
    if n * (n-1) >= num_charts:
        return (n, n-1, )
    return (n, n, )
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

