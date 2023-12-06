"""
"""


#[
from __future__ import absolute_import

from typing import (Sequence, )
import plotly.graph_objects as _pg
import plotly.subplots as _ps
#]


__all__ = (
    "make_subplots",
)


_EMPTY_SUBPLOT_TITLE = " "


def make_subplots(
    rows: int,
    columns: int | None = None,
    cols: int | None = None,
    subplot_titles: Sequence[str] | bool | None = None,
    **kwargs,
) -> _pg.Figure:
    """
    """
    if cols is not None and columns is None:
        columns = cols
    total_num_subplots = rows * columns
    if subplot_titles is True:
        subplot_titles = (_EMPTY_SUBPLOT_TITLE, )*total_num_subplots
    return _ps.make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=subplot_titles or None,
        **kwargs,
    )

