"""
"""


#[
from __future__ import annotations

from typing import (Callable, )
from numbers import (Real, )
import dataclasses as _dc
import copy as _cp
import numpy as _np

from .. import dates as _dates
from .. import has_variants as _has_variants
from ..conveniences import descriptions as _descriptions
from ..series import main as _series
from ..databoxes import main as _databoxes
#]


#
# REFACTOR:
# * rename `index_base_columns` to `are_base_columns` or use int indexes
#


class Invariant:
    """
    """
    #[

    __slots__ = (
        "descriptions",
        "logly_indexes",
        "index_base_columns",
        "names",
        "output_qids",
        "dates",
        "min_max_shift",
    )

    def __init__(
        self,
        names: Iterable[str],
        dates: Iterable[_dates.Dater],
        /,
        *,
        base_columns: Iterable[int] | None = None,
        descriptions: Iterable[str | None] | None = None,
        output_names: Iterable[str] | None = None,
        qid_to_logly: dict[int, bool | None] | None = None,
        min_max_shift: tuple[int, int] = (0, 0),
        frequency: _dates.Frequency | None = None,
        tag_alongs: dict[str, Any] | None = None,
    ) -> None:
        """
        """
        self.names = tuple(names)
        self.dates = tuple(dates)
        base_columns = base_columns or ()
        self.index_base_columns = tuple(i in base_columns for i in range(self.num_periods))
        self._populate_descriptions(descriptions, )
        self._populate_logly_index(qid_to_logly, )
        self.min_max_shift = min_max_shift
        self._populate_output_qids(output_names, )

    def _populate_output_qids(
        self,
        output_names: Iterable[str] | None,
    ) -> None:
        """
        """
        output_names = (
            tuple(set(output_names) & set(self.names))
            if output_names is not None
            else self.names
        )
        self.output_qids = tuple(
            qid
            for qid, name in enumerate(self.names, )
            if name in output_names
        )

    @property
    def num_periods(self, /, ) -> int:
        """
        """
        return len(self.dates)

    @property
    def num_names(self, /, ) -> int:
        """
        """
        return len(self.names)

    @property
    def from_to(self, /, ) -> tuple[_dates.Dater, _dates.Dater]:
        """
        """
        return self.dates[0], self.dates[-1]

    @property
    def base_columns(self, /, ) -> tuple[int]:
        """
        """
        return tuple(i for i in range(self.num_periods) if self.index_base_columns[i])

    @property
    def nonbase_columns(self, /, ) -> tuple[int]:
        """
        """
        return tuple(i for i in range(self.num_periods) if not self.index_base_columns[i])

    @property
    def base_slice(self, /, ) -> slice:
        """
        """
        base_columns = self.base_columns
        return slice(base_columns[0], base_columns[-1]+1)

    def create_name_to_row(
        self,
        /,
    ) -> dict[str, int]:
        return {
            name: row
            for row, name in enumerate(self.names, )
        }

    def _populate_descriptions(
        self,
        descriptions: Iterable[str | None] | None,
    ) -> None:
        """
        """
        if descriptions is None:
            self.descriptions = tuple("" for _ in self.names)
        else:
            self.descriptions = tuple((d or "") for d in descriptions)

    def _populate_logly_index(
        self,
        qid_to_logly: dict[int, bool] | None,
        /,
    ) -> None:
        """
        """
        qid_to_logly = qid_to_logly or {}
        self.logly_indexes = tuple(
            i for i in range(len(self.names), )
            if qid_to_logly.get(i, False)
        )

    def remove_periods_from_start(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove:
            self.dates = self.dates[remove:]
            self.index_base_columns = self.index_base_columns[remove:]

    def remove_periods_from_end(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove:
            self.dates = self.dates[:-remove]
            self.index_base_columns = self.index_base_columns[:-remove]


