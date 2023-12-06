"""
"""

#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, )
from types import (EllipsisType, )
#]


class ItemMixin:
    """
    """
    #[

    def __getitem__(
        self,
        index: int | slice,
    ) -> Self:
        """
        """
        row_index, column_index = index[0], index[1]
        new = self.copy()
        if not _is_all(row_index):
            raise NotImplementedError
        if not _is_all(column_index):
            new._select_columns(new, column_index, )
        return new

    def _select_columns(
        self,
        column_index: int | slice | EllipsisType,
    ) -> Self:
        """
        """
        if isinstance(column_index, int):
            column_index = slice(column_index, column_index + 1, )
        self.base_span = self.base_span[column_index]
        for r in ("exogenized", "endogenized", "anticipated", ):
            register = getattr(self, f"_{r}_register")
            for k, v in register.items():
                register[k] = v[column_index]
    #]


def _is_all(index, /, ) -> bool:
    """
    """
    return index is ... or index == slice(None, )

