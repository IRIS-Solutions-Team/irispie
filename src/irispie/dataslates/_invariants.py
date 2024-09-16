"""
"""


#[
from __future__ import annotations

from typing import (Callable, )
from numbers import (Real, )
import dataclasses as _dc
import copy as _cp
import numpy as _np

from ..dates import (Period, Frequency, )
from .. import dates as _dates
from ..conveniences import descriptions as _descriptions
from ..series import main as _series
from ..databoxes import main as _databoxes
from .. import has_variants as _has_variants
#]


class Invariant:
    """
    """
    #[

    __slots__ = (
        "names",
        "periods",
        "descriptions",
        "logly_indexes",
        "base_columns",
        "output_qids",
        "min_max_shift",
    )

    def __init__(
        self,
        names: Iterable[str],
        periods: Iterable[Period],
        /,
        *,
        base_columns: Iterable[int] | None = None,
        descriptions: Iterable[str | None] | None = None,
        output_names: Iterable[str] | None = None,
        qid_to_logly: dict[int, bool | None] | None = None,
        min_max_shift: tuple[int, int] = (0, 0),
        frequency: Frequency | None = None,
    ) -> None:
        """
        """
        self.names = tuple(names)
        self.periods = tuple(periods)
        self.base_columns = tuple(sorted(base_columns)) if base_columns else ()
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
        return len(self.periods)

    @property
    def num_base_periods(self, /, ) -> int:
        """
        """
        return len(self.base_columns)

    @property
    def num_names(self, /, ) -> int:
        """
        """
        return len(self.names)

    @property
    def from_until(self, /, ) -> tuple[Period, Period]:
        """
        """
        return self.periods[0], self.periods[-1]

    from_to = from_until

    @property
    def base_periods(self, /, ) -> tuple[Period]:
        """
        """
        return tuple(self.periods[i] for i in self.base_columns)

    @base_periods.setter
    def base_periods(self, base_periods: Iterable[Period], /, ) -> None:
        """
        """
        base_periods = set(base_periods)
        self.base_columns = tuple(
            i for i, period in enumerate(self.periods)
            if period in base_periods
        )

    @property
    def nonbase_columns(self, /, ) -> tuple[int]:
        """
        """
        all_columns = set(range(self.num_periods))
        return tuple(sorted(all_columns - set(self.base_columns)))

    @property
    def base_slice(self, /, ) -> slice:
        """
        """
        base_columns = self.base_columns
        return slice(base_columns[0], base_columns[-1]+1, )

    def copy(self, /, ) -> Invariant:
        """
        """
        return _cp.deepcopy(self, )

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
        num_periods_to_remove: int,
        /,
    ) -> None:
        """
        """
        if num_periods_to_remove:
            self.periods = self.periods[num_periods_to_remove:]
            self.base_columns = tuple(
                i - num_periods_to_remove for i in self.base_columns
                if i >= num_periods_to_remove
            )

    def remove_periods_from_end(
        self,
        num_periods_to_remove: int,
        /,
    ) -> None:
        """
        """
        if num_periods_to_remove:
            self.periods = self.periods[:-num_periods_to_remove]
            num_periods = len(self.periods)
            self.base_columns = tuple(
                i for i in self.base_columns
                if i < num_periods
            )

    def add_periods_to_end(
        self,
        num_periods_to_add: int,
        /,
    ) -> None:
        """
        """
        if num_periods_to_add > 0:
            end_period = self.periods[-1]
            new_end_period = end_period + num_periods_to_add
            self.periods = self.periods + _dates.periods_from_until(end_period, new_end_period, )


