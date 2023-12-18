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
import dataclasses as _dc

from ..conveniences import descriptions as _descriptions
from ..conveniences import copies as _copies
from .. import plotly_wrap as _plotly_wrap
from .. import dates as _dates
from ..databoxes import main as _databoxes
#]


__all__ =  (
    "Chartpack",
)


@_dc.dataclass
class Chartpack(
    _descriptions.DescriptionMixin,
    _copies.CopyMixin,
):
    """
    """
    #[

    title: str = ""
    span: Iterable[_dates.Dater] | EllipsisType = ...
    tiles: tuple[int, int] | None = None
    transforms: dict[str, Callable] | None = None
    highlight: Iterable[_dates.Dater] | None = None
    legend: Iterable[str] | None = None
    reverse_plot_order: bool = False

    def __init__(self, /, **kwargs, ) -> None:
        """
        """
        self._figures = []
        self._description = None

    def plot(
        self,
        input_db: _databoxes.Databox,
        /,
        transforms: dict[str, Callable] | None = None,
        show_figures: bool = True,
    ) -> tuple[_pg.Figure, ...]:
        """
        """
        transforms = transforms if transforms is not None else self.transforms
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
    def num_figures(self, ) -> int:
        """
        """
        return len(self._figures, )

    def _add_figure(self, figure: _Figure, ) -> None:
        """
        """
        self._figures.append(figure, )

    def add_figure(self, figure_string: str, **kwargs, ) -> None:
        """
        """
        new_figure = _Figure.from_string(figure_string, **kwargs, )
        self._add_figure(new_figure, )
        return new_figure

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
        highlight: tuple[_dates.Dater, ...] | None = None,
        legend: Iterable[str, ...] | None = None,
        reverse_plot_order: bool = False,
        **kwargs,
    ) -> None:
        """
        """
        tiles = _resolve_tiles(tiles, self.num_charts, )
        figure = _plotly_wrap.make_subplots(
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
                _plotly_wrap.highlight(figure, highlight, subplot=i, )
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

    def add_chart(self, chart_string: str, ) -> None:
        """
        """
        self._add_chart(_Chart.from_string(chart_string, ), )

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
        return _tree_repr(self, )

    def _one_liner(self, /, ) -> str:
        return f"Chart({self.title!r}, {self._expression!r}, {self._transform!r}, )"

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

