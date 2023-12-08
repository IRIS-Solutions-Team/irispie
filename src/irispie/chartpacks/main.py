"""
Time series chartpacks
"""


#[
from __future__ import annotations

from typing import (Self, Iterator, Iterable, Sequence, Callable, )
from types import (EllipsisType, )
import re as _re
import math as _ma
import plotly.graph_objects as _pg

from ..conveniences import descriptions as _descriptions
from ..conveniences import copies as _copies
from .. import plotly as _plotly
from .. import dates as _dates
from ..databoxes import main as _databoxes
#]


__all__ =  (
    "Chartpack",
)


class Chartpack(
    _descriptions.DescriptionMixin,
    _copies.CopyMixin,
):
    """
    """
    #[

    def __init__(self, title: str = "", /, **kwargs, ) -> None:
        """
        """
        self.title = title
        self.span = kwargs.get("span", ..., )
        self.tiles = kwargs.get("tiles", None, )
        self.transforms = kwargs.get("transforms", None, )
        self.highlight = kwargs.get("highlight", None, )
        self.legend = kwargs.get("legend", None, )
        self.reverse_plot_order = kwargs.get("reverse_plot_order", False, )
        self._figures = []
        self._description = None

    def plot(
        self,
        input_db: _databoxes.Databox,
        /,
        transforms: dict[str, Callable] | None = None,
        show_charts: bool = True,
    ) -> tuple[_pg.Figure, ...]:
        """
        """
        transforms = transforms if transforms is not None else self.transforms
        return tuple(
            figure.plot(input_db, **self.__dict__, )
            for figure in self
        )

    def format_figure_titles(
        self,
        **kwargs,
    ) -> None:
        """
        """
        for figure in self:
            figure.format_figure_title(**kwargs, )

    @property
    def num_figures(self, ) -> int:
        """
        """
        return len(self._figures, )

    def _add_figure(self, figure: _Figure, ) -> None:
        """
        """
        self._figures.append(figure, )

    def _add_figure_from_string(self, *args, **kwargs, ) -> None:
        """
        """
        self._add_figure(_Figure.from_string(*args, **kwargs, ), )

    def add_figure(self, *args, ) -> None:
        """
        """
        for figure_strings in args:
            _add_strings(self, figure_strings, self._add_figure_from_string, )

    add_figures = add_figure
    __add__ = add_figure

    def add_chart(self, *args, **kwargs, ) -> None:
        """
        """
        self._figures[-1].add_chart(*args, **kwargs, )

    add_charts = add_chart
    __lshift__ = add_chart

    def __str__(self, /, ) -> str:
        """
        """
        return repr(self, )

    def __repr__(self, /, ) -> str:
        """
        """
        return self._repr()

    def _repr(self, /, indent: str = "", ) -> str:
        """
        """
        child_indent = indent + "   |"
        lines = [f"{indent}Chartpack({self.title!r}, )", ]
        lines.extend(f"{c._repr(child_indent, )}" for c in self)
        return "\n".join(lines)

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
        "_charts",
        "title",
    )

    def __init__(
        self,
        /,
        title: str | None = None,
    ) -> None:
        """
        """
        self._charts = []
        self.title = title

    @classmethod
    def from_string(
        klass,
        input_string: str,
        /,
    ) -> Self:
        """
        """
        return klass(title=input_string, )

    @property
    def num_charts(self, ) -> int:
        """
        """
        return len(self._charts, )

    def plot(
        self,
        input_db: _databoxes.Databox,
        /,
        span: Iterable[_dates.Dater] | EllipsisType = ...,
        tiles: Sequence[int] | int | None = None,
        transforms: dict[str, Callable] | None = None,
        show_charts: bool = True,
        highlight: tuple[_dates.Dater, ...] | None = None,
        legend: Iterable[str, ...] | None = None,
        reverse_plot_order: bool = False,
        **kwargs,
    ) -> None:
        """
        """
        tiles = _resolve_tiles(tiles, self.num_charts, )
        figure = _plotly.make_subplots(
            rows=tiles[0], columns=tiles[1],
            subplot_titles=tuple(c.caption for c in self),
        )
        for i, chart in enumerate(self, ):
            chart.plot(
                input_db,
                figure,
                i,
                span=span,
                transforms=transforms,
                legend=legend if i==0 else None,
                reverse_plot_order=reverse_plot_order,
            )
            if highlight is not None:
                _plotly.highlight(figure, highlight, subplot=i, )
        figure.update_layout(title={"text": self.title, }, )
        if show_charts:
            figure.show()
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
        return self._repr()

    def _repr(self, /, indent: str = "", ) -> str:
        """
        """
        child_indent = indent + "   |"
        lines = [f"{indent}_Figure({self.title!r}, )", ]
        lines.extend(f"{c._repr(child_indent, )}" for c in self)
        return "\n".join(lines)

    def _add_chart(self, chart: _Chart, ) -> None:
        """
        """
        self._charts.append(chart, )

    def _add_chart_from_string(self, input_string: str, /, ) -> None:
        """
        """
        self._add_chart(_Chart.from_string(input_string, ), )

    def add_chart(self, *args, ) -> None:
        """
        """
        for chart_strings in args:
            _add_strings(self, chart_strings, self._add_chart_from_string, )

    add_charts = add_chart
    __lshift__ = add_chart

    def __iter__(self, /, ) -> Iterator[_Chart]:
        """
        """
        return iter(self._charts, )

    #]


class _Chart:
    """
    """
    #[

    _INPUT_STRING_PATTERN = _re.compile(r"^(?:(?P<title>.*?):)?(?P<expression>.*?)(?:\[(?P<transform>.*?)\])?$")

    __slots__ = (
        "title",
        "_expression",
        "_transform"
    )

    def __init__(
        self,
        expression: str,
        title: str | None = None,
        transform: str | Callable | None = None,
    ) -> None:
        """
        """
        self._expression = (expression or "").strip()
        self.title = (title or "").strip()
        self._transform = (transform or "").strip()

    @classmethod
    def from_string(
        klass: type(Self),
        input_string: str,
        /,
    ) -> Self:
        """
        """
        match = klass._INPUT_STRING_PATTERN.match(input_string, )
        if not match:
            raise ValueError(f"Invalid input string: {input_string!r}")
        return klass(**match.groupdict(), )

    @property
    def caption(self, ) -> str:
        """
        """
        return self.title or (self._expression + f" [{self._transform}]" if self._transform else "")

    def plot(
        self,
        input_db: _databoxes.Databox,
        figure: _pg.Figure,
        index: int,
        /,
        span: Iterable[_dates.Dater] | EllipsisType = ...,
        transforms: dict[str, Callable] | None = None,
        legend: Iterable[str, ...] | None = None,
        reverse_plot_order: bool = False,
    ) -> None:
        """
        """
        series_to_plot = input_db.evaluate_expression(self._expression, )
        series_to_plot = self._apply_transform(series_to_plot, transforms, )
        series_to_plot.plot(
            figure=figure,
            subplot=index,
            span=span,
            show_figure=False,
            freeze_span=True,
            legend=legend,
            reverse_plot_order=reverse_plot_order,
        )

    def _apply_transform(self, x, transforms, ):
        """
        """
        if self._transform:
            func = transforms[self._transform]
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
        return self._repr()

    def _repr(self, /, indent: str = "", ) -> str:
        """
        """
        return f"{indent}_Chart({self.title!r}, {self._expression!r}, {self._transform!r}, )"

    #]


def _add_strings(self, input_strings: Iterable[str] | str, call: Callable ) -> None:
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


