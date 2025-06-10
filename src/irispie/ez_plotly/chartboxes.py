r"""
"""


#[

from __future__ import annotations

from typing import Literal
import plotly.graph_objects as _pg
import plotly.subplots as _ps

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence

#]


__all__ = (
    "Chartbox",
)


BarMode = Literal["group", "relative", "stack", "overlay", ]
BarNorm = Literal["fraction", "percent", ]


class Chartbox:
    r"""
    """
    #[

    def __init__(
        self,
        *,
        subplots: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        r"""
        """
        self.subplots = subplots if subplots is None else (int(subplots[0]), int(subplots[1]), )
        #
        if not self.subplots:
            self.figure = _pg.Figure(**kwargs, )
        else:
            self.figure = _make_subplots(
                num_rows=self.subplots[0],
                num_columns=self.subplots[1],
            )

    def show(self):
        self.figure.show()

    def add_trace(
        self,
        trace,
        subplot: tuple[int, int] | int | None = None,
        subplot_title: str | None = None,
    ) -> dict[str, int] | None:
        r"""
        """
        row_column, _ = self.resolve_subplot_reference(subplot, )
        self.figure.add_trace(trace, **row_column, )
        return row_column

    def update_layout(self, *args, **kwargs):
        self.figure.update_layout(*args, **kwargs)

    def update_traces(self, *args, **kwargs):
        self.figure.update_traces(*args, **kwargs)

    def add_line(
        self,
        *,
        subplot: tuple[int, int] | int | None = None,
        subplot_title: str | None = None,
        **kwargs,
    ) -> pg.Scatter:
        r"""
        """
        trace = new_line(**kwargs, )
        self.add_trace(trace, subplot=subplot, subplot_title=subplot_title, )
        return trace

    def add_bar(
        self,
        *,
        subplot: tuple[int, int] | int | None = None,
        subplot_title: str | None = None,
        mode: BarMode | None = None,
        norm: BarNorm | None = None,
        **kwargs,
    ) -> pg.Scatter:
        r"""
        """
        trace = new_bar(**kwargs, )
        row_column = self.add_trace(trace, subplot=subplot, subplot_title=subplot_title, )
        self.set_bar_style(mode=mode, norm=norm, )
        return trace

    def set_bar_style(
        self,
        mode: BarMode | None = None,
        norm: BarNorm | None = None,
    ) -> None:
        layout_update = {}
        if mode:
            layout_update["barmode"] = mode
        if norm:
            layout_update["barnorm"] = norm
        if layout_update:
            self.figure.update_layout(**layout_update, )

    def resolve_subplot_reference(self, *args, **kwargs, ) -> tuple[dict[str, int], int | None]:
        r"""
        """
        return resolve_subplot_reference(self.figure, *args, **kwargs, )

    def update_subplot_title(self, *args, **kwargs, ) -> _pg.layout.Annotation | None:
        r"""
        """
        return update_subplot_title(self.figure, *args, **kwargs, )

    #]


_EMPTY_SUBPLOT_TITLE = " "
_DEFAULT_VERTICAL_SPACING = 0.1
_DEFAULT_HORIZONTAL_SPACING = 0.05


def _make_subplots(
    num_rows: int,
    num_columns: int,
    #
    subplot_titles: Sequence[str] | bool | None = True,
    vertical_spacing: float | None = None,
    horizontal_spacing: float | None = None,
    figure_height: int | None = None,
    figure_title: str | None = None,
    show_legend: bool = False,
    **kwargs,
) -> _pg.Figure:
    r"""
    """
    if vertical_spacing is None:
        vertical_spacing = _DEFAULT_VERTICAL_SPACING
    if horizontal_spacing is None:
        horizontal_spacing = _DEFAULT_HORIZONTAL_SPACING
    total_numsubplots = num_rows * num_columns
    if subplot_titles is True:
        subplot_titles = (_EMPTY_SUBPLOT_TITLE, ) * total_numsubplots
    figure = _ps.make_subplots(
        rows=num_rows,
        cols=num_columns,
        subplot_titles=subplot_titles or None,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        **kwargs,
    )
    #
    if figure_title is not None:
        figure.update_layout(title=figure_title, )
    #
    if figure_height is not None:
        figure.update_layout(height=figure_height, )
    #
    figure.update_layout(showlegend=show_legend, )
    #
    return figure


def resolve_subplot_reference(
    figure: _pg.Figure,
    subplot: tuple[int, int] | int | None,
) -> tuple[dict[str, int], int | None]:
    r"""
    """
    if subplot is None:
        return {}, None
    rows, columns = figure._get_subplot_rows_columns()
    num_rows = len(rows)
    num_columns = len(columns)
    numsubplots = num_rows * num_columns
    if isinstance(subplot, int, ):
        if subplot < 0:
            subplot += numsubplots
        index = subplot
        row = index // num_columns
        column = index % num_columns
        row_column = {"row": row+1, "col": column+1, }
        return row_column, index,
    else:
        row = subplot[0]
        column = subplot[1]
        index = row * num_columns + column
        row_column = {"row": row+1, "col": column+1, }
        return row_column, index,


def update_subplot_title(
    figure: _pg.Figure,
    title: str,
    subplot: tuple[int, int] | int | None,
) -> _pg.layout.Annotation | None:
    r"""
    """
    row_column, index, = resolve_subplot_reference(figure, subplot, )
    annotation = next(figure.select_annotations(index, ), None, )
    if annotation:
        annotation.text = title
    return annotation


def new_line(
    *,
    y: Sequence[float],
    x: Sequence[float] | None = None,
    mode: Literal["lines", "markers", "lines+markers", ] = "lines",
    **kwargs,
) -> pg.Scatter:
    r"""
    """
    x, y, = _get_xy_data(y, x, )
    return _pg.Scatter(x=x, y=y, mode=mode, **kwargs, )


def new_bar(
    *,
    y: Sequence[float],
    x: Sequence[float] | None = None,
    **kwargs,
) -> pg.Scatter:
    r"""
    """
    x, y, = _get_xy_data(y=y, x=x, )
    trace = _pg.Bar(x=x, y=y, **kwargs, )
    return trace


def _get_xy_data(
    y: Sequence[float],
    x: Sequence[float] | None = None,
) -> tuple[tuple[float], tuple[float]]:
    r"""
    """
    y = tuple(y)
    if x is not None:
        x = tuple(x)
    else:
        x = tuple(range(len(y)))
    return x, y,

