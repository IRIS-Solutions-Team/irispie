"""
"""


#[
from __future__ import annotations

import numpy as _np
import dataclasses as _dc
#]


__all__ = (
    "DimensionNames",
)


@_dc.dataclass
class DimensionNames:
    """
    """
    #[
    rows: tuple[str, ...] | None = None
    columns: tuple[str, ...] | None = None

    def select(
        self,
        array: _np.ndarray,
        select_names: tuple[tuple[str, ...], tuple[str, ...]] | tuple[str, ...],
    ) -> _np.ndarray:
        """
        """
        select_row_names, select_column_names = _extract_row_and_column_names(select_names, )
        row_index = _lookup_names(self.rows, select_row_names, )
        column_index = _lookup_names(self.columns, select_column_names, )
        select_array = _np.copy(array)
        select_array = select_array[row_index, :, ...]
        select_array = select_array[:, column_index, ...]
        return select_array
    #]


def _extract_row_and_column_names(
    names: tuple[tuple[str, ...], tuple[str, ...]] | tuple[str, ...],
    /,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    """
    if isinstance(names, str):
        row_names = (names, )
        column_names = (names, )
    elif isinstance(names[0], str):
        row_names = names
        column_names = names
    else:
        row_names = names[0]
        column_names = names[1]
    return row_names, column_names


def _lookup_names(
    names: tuple[str, ...],
    select_names: tuple[str, ...],
) -> tuple[int, ...]:
    """
    """
    return tuple(
        names.index(n)
        for n in select_names
    )

