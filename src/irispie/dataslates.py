"""
Data arrays with row and column names
"""


#[
from __future__ import annotations

from typing import (Self, Protocol, )
from numbers import (Number, )
from collections.abc import (Iterable, )
import numpy as _np
import dataclasses as _dc

from .series import main as _series
from .databanks import main as _databanks
from .plans import main as _plans
from . import dates as _dates
#]


__all__ = (
    "Dataslate",
)


class SlatableProtocol(Protocol, ):
    """
    """
    def get_min_max_shift(self, /, ) -> tuple[int, int]:  ...
    def get_databank_names(self, /, ) -> tuple[str, ...]:  ...


@_dc.dataclass
class Dataslate:
    """
    """
    #[
    data: _np.ndarray | None = None
    row_names: Iterable[str] | None = None
    missing_names: tuple[str, ...] | None = None
    column_dates: tuple[_dates.Dater, ...] | None = None
    base_columns: tuple[int, ...] | None = None

    def __init__(
        self,
        slatable: SlatableProtocol,
        databank: _databanks.Databank,
        base_range: Iterable[_dates.Dater],
        /,
        slate: int = 0,
        plan: _plans.Plan | None = None,
    ) -> Self:
        """
        """
        self.row_names = slatable.get_databank_names(plan, )
        self.missing_names = databank.get_missing_names(self.row_names, )
        self._resolve_column_dates(slatable, base_range, )
        self._populate_data(databank, slate, )

    def _populate_data(
        self,
        databank: _databanks.Databank,
        slate: int,
    ) -> None:
        """
        """
        generate_data = tuple(
            _extract_data_from_record(databank[n], self.from_to, self.num_periods, slate, )
            if n not in self.missing_names else self.nan_row
            for n in self.row_names
        )
        self.data = _np.vstack(generate_data)

    def to_databank(self, *args, **kwargs, ) -> _dates.Databank:
        """
        """
        return multiple_to_databank((self,), *args, **kwargs, )

    @property
    def num_periods(self, /, ) -> int:
        """
        """
        return len(self.column_dates)

    @property
    def from_to(self, /, ) -> tuple[_dates.Dater, _dates.Dater]:
        """
        """
        return self.column_dates[0], self.column_dates[-1]

    @property
    def nan_row(self, /, ) -> int:
        """
        """
        return _np.full((1, self.num_periods), _np.nan, dtype=float)

    @property
    def num_rows(self, /, ) -> int:
        """
        """
        return self.data.shape[0] if self.data is not None else 0

    def remove_terminal(self, /, ) -> None:
        """
        """
        last_base_column = self.base_columns[-1]
        self.data = self.data[:, :last_base_column+1]
        self.column_dates = self.column_dates[:last_base_column+1]

    def remove_columns(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove > 0:
            self.data = self.data[:, remove:]
            self.column_dates = self.column_dates[remove:]
        elif remove < 0:
            self.data = self.data[:, :remove]
            self.column_dates = self.column_dates[:remove]

    def copy_data(self, /, ) -> _np.ndarray:
        """
        """
        return self.data.copy()

    def fill_missing_in_base_columns(
        self,
        names,
        /,
        fill: Number = 0,
    ) -> None:
        """
        """
        base_slice = slice(self.base_columns[0], self.base_columns[-1]+1)
        for i, n in enumerate(self.row_names):
            if names is not Ellipsis and n not in names:
                continue
            values = self.data[i, base_slice]
            values[_np.isnan(values)] = fill

    def create_name_to_row(
        self,
        /,
    ) -> dict[str, int]:
        return { name: row for row, name in enumerate(self.row_names, ) }

    def _resolve_column_dates(
        self,
        slatable: SlatableProtocol,
        base_range: Iterable[_dates.Dater],
        /,
    ) -> None:
        self.column_dates, self.base_columns = get_extended_range(slatable, base_range, )
    #]


def multiple_to_databank(
    selves,
    /,
    target_databank: _databanks.Databank | None = None,
) -> _databanks.Databank:
    """
    Add data from a dataslate to a new or existing databank
    """
    #[
    if target_databank is None:
        target_databank = _databanks.Databank()
    self = selves[0]
    num_columns = len(selves)
    for row, n in enumerate(self.row_names):
        data = _np.hstack(tuple(ds.data[(row,), :].T for ds in selves))
        x = _series.Series(num_columns=num_columns, )
        x.set_data(self.column_dates, data)
        target_databank[n] = x
    return target_databank
    #]


def get_extended_range(
    slatable: SlatableProtocol,
    base_range: Iterable[_dates.Dater],
    /,
) -> Iterable[_dates.Dater]:
    """
    """
    base_range = tuple(t for t in base_range)
    num_base_periods = len(base_range)
    min_shift, max_shift = slatable.get_min_max_shifts()
    if min_shift == 0:
        min_shift = -1
    min_base_date = min(base_range)
    max_base_date = max(base_range)
    start_date = min_base_date + min_shift
    end_date = max_base_date + max_shift
    base_columns = tuple(range(-min_shift, -min_shift+num_base_periods))
    column_dates = tuple(_dates.Ranger(start_date, end_date))
    return column_dates, base_columns


def _extract_data_from_record(record, from_to, num_periods, column, /, ):
    """
    """
    try:
        return record.get_data_column_from_to(from_to, column).reshape(1, -1)
    except AttributeError:
        return _np.full((1, num_periods), float(record), dtype=float, )

