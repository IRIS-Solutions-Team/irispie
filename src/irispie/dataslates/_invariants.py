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


@_dc.dataclass(slots=True, )
class Invariant:
    """
    """
    #[

    descriptions: Iterable[str] | None = None
    boolex_logly: tuple[bool, ...] | None = None
    index_base_columns: tuple[int, ...] | None = None
    names: tuple[str] | None = None
    dates: tuple[_dates.Dater, ...] | None = None
    min_max_shift: tuple[int, int] | None = None
    databox_value_creator: tuple[Callable, ...] | None = None

    def __init__(
        self,
        names: Iterable[str],
        dates: Iterable[_dates.Dater],
        /,
        *,
        base_columns: Iterable[int] | None = None,
        descriptions: Iterable[str | None] | None = None,
        scalar_names: Iterable[str] | None = None,
        qid_to_logly: dict[int, bool | None] | None = None,
        min_max_shift: tuple[int, int] = (0, 0),
    ) -> None:
        """
        """
        self.names = tuple(names)
        self.dates = tuple(_dates.ensure_dater(d) for d in dates)
        base_columns = base_columns or ()
        self.index_base_columns = tuple(i in base_columns for i in range(self.num_periods))
        self._populate_descriptions(descriptions, )
        self._populate_boolex_logly(qid_to_logly, )
        self.min_max_shift = min_max_shift
        self._populate_databox_value_creator(scalar_names, )

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

    def _populate_boolex_logly(
        self,
        qid_to_logly: dict[int, bool] | None,
        /,
    ) -> None:
        """
        """
        qid_to_logly = qid_to_logly or {}
        self.boolex_logly = tuple(
            qid_to_logly.get(i, False) or False
            for i in range(len(self.names), )
        )

    def _populate_databox_value_creator(
        self,
        scalar_names: Iterable[str] | None,
        /,
    ) -> None:
        """
        """
        if not scalar_names:
            scalar_names = ()
        self.databox_value_creator = tuple(
            _scalar_from_values if n in scalar_names else _series_from_values
            for n in self.names
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


def _series_from_values(
    num_variants: int,
    values: _np.ndarray,
    start_date: _dates.Dater,
    description: str,
) -> _series.Series:
    """
    """
    return _series.Series(
        num_variants=num_variants,
        start_date=start_date,
        values=values.T,
        description=description,
    )


def _scalar_from_values(
    num_variants: int,
    values: _np.ndarray,
    **kwargs,
) -> list[Real] | Real:
    """
    """
    scalars = list(values[:, 0])
    is_singleton = _has_variants.is_singleton(num_variants, )
    return _has_variants.unpack_singleton(scalars, is_singleton, )

