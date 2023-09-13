"""
Data arrays with row and column names
"""


#[
from __future__ import annotations

import numpy as np_
from typing import (Self, Protocol, )
from numbers import (Number, )
from collections.abc import (Iterable, )

from .series import main as _series
from .databanks import main as _databanks
from . import dates as _dates
#]


__all__ = (
    "Dataslab",
)


class SimulatableProtocol(Protocol):
    def get_min_max_shift(self, /, ) -> tuple[int, int]:  ...
    def get_databank_names(self, /, ) -> tuple[str, ...]:  ...


class Dataslab:
    """
    """
    #[
    data: np_.ndarray | None = None
    row_names: Iterable[str] | None = None
    missing_names: tuple[str, ...] | None = None
    column_dates: tuple[Dater, ...] | None = None
    base_columns: tuple[int, ...] | None = None

    def to_databank(self, *args, **kwargs, ) -> _dates.Databank:
        """
        """
        return multiple_to_databank((self,), *args, **kwargs, )

    @classmethod
    def from_databank(
        cls,
        databank: _databanks.Databank,
        names: Iterable[str],
        ext_range: Ranger,
        /,
        column: int = 0,
    ) -> Self:
        """
        Add data from a databank to this dataslab
        """
        self = cls()
        missing_names = [
            n for n in names
            if n not in databank
        ]
        #
        num_periods = len(ext_range)
        nan_row = np_.full((1, num_periods), np_.nan, dtype=float)
        generate_data = tuple(
            _extract_data_from_record(databank[n], ext_range, column, )
            if n not in missing_names else nan_row
            for n in names
        )
        #
        self.data = np_.vstack(generate_data)
        self.row_names = tuple(names)
        self.column_dates = tuple(ext_range)
        self.missing_names = missing_names
        return self

    @classmethod
    def from_databank_for_simulation(
        cls,
        simulatable: SimulatableProtocol,
        databank: _databanks.Databank,
        base_range: Iterable[Dater],
        /,
        column: int = 0,
    ) -> Self:
        """
        """
        ext_range, base_columns = (
            get_extended_range(
                simulatable,
                base_range,
            )
        )
        names = simulatable.get_databank_names()
        self = cls.from_databank(databank, names, ext_range, column=column, )
        self.base_columns = base_columns
        return self

    @property
    def num_ext_periods(self, /, ) -> int:
        """
        """
        return len(self.column_dates)

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
        value: Number = 0,
    ) -> None:
        """
        """
        for i, n in enumerate(self.row_names):
            if n not in names:
                continue
            self.data[i, self.base_columns[0]:self.base_column_data[-1]+1] = value

def _extract_data_from_record(record, ext_range, column, /, ):
    """
    """
    return (
        record.get_data_column(ext_range, column).reshape(1, -1) 
        if hasattr(record, "get_data")
        else np_.full((1, len(ext_range)), float(record), dtype=float, )
    )


def multiple_to_databank(
    selves,
    /,
    target_databank: _databanks.Databank | None = None,
) -> _databanks.Databank:
    """
    Add data from a dataslab to a new or existing databank
    """
    #[
    if target_databank is None:
        target_databank = _databanks.Databank()
    self = selves[0]
    num_columns = len(selves)
    for row, n in enumerate(self.row_names):
        data = np_.hstack(tuple(ds.data[(row,), :].T for ds in selves))
        x = _series.Series(num_columns=num_columns, )
        x.set_data(self.column_dates, data)
        target_databank[n] = x
    return target_databank
    #]


def get_extended_range(
    simulatable: SimulatableProtocol,
    base_range: Iterable[_dates.Dater],
    /,
) -> Iterable[_dates.Dater]:
    """
    """
    base_range = tuple(t for t in base_range)
    num_base_periods = len(base_range)
    min_shift, max_shift = simulatable.get_min_max_shifts()
    start_date = base_range[0] + min_shift
    end_date = base_range[-1] + max_shift
    base_columns = tuple(range(-min_shift, -min_shift+num_base_periods))
    ext_range = tuple(_dates.Ranger(start_date, end_date))
    return ext_range, base_columns

