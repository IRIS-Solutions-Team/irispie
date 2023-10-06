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
import numpy as _np

from .series import main as _series
from .databoxes import main as _databoxes
from .plans import main as _plans
from .incidences import main as _incidences
from . import dates as _dates
#]


__all__ = (
    "Dataslate",
)


class SlatableProtocol(Protocol, ):
    """
    """
    def get_min_max_shift(self, /, ) -> tuple[int, int]:  ...
    def get_databox_names(self, /, ) -> tuple[str, ...]:  ...


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
        databox: _databoxes.Databox,
        base_range: Iterable[_dates.Dater],
        /,
        slate: int = 0,
        plan: _plans.Plan | None = None,
    ) -> Self:
        """
        """
        self.row_names = slatable.get_databox_names(plan, )
        self.missing_names = databox.get_missing_names(self.row_names, )
        self._resolve_column_dates(slatable, base_range, )
        self._populate_data(databox, slate, )

    def _populate_data(
        self,
        databox: _databoxes.Databox,
        slate: int,
    ) -> None:
        """
        """
        data = tuple(
            _extract_data_from_record(databox[n], self.from_to, self.num_periods, slate, )
            if n not in self.missing_names else self.nan_row
            for n in self.row_names
        )
        self.data = _np.vstack(data)

    def to_databox(self, *args, **kwargs, ) -> _dates.Databox:
        """
        """
        return multiple_to_databox((self,), *args, **kwargs, )

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

    def retrieve_vector(
        self,
        tokens: tuple[str, ...],
        column_zero: int,
        /,
    ) -> _np.ndarray:
        """
        """
        return retrieve_vector_from_data_array(self.data, tokens, column_zero, )

    def store_vector(
        self,
        tokens: tuple[str, ...],
        vector: _np.ndarray,
        column_zero: int,
        /,
    ) -> None:
        """
        """
        store_vector_in_data_array(vector, self.data, tokens, column_zero, )

    #]


def multiple_to_databox(
    selves,
    /,
    target_databox: _databoxes.Databox | None = None,
) -> _databoxes.Databox:
    """
    Add data from a dataslate to a new or existing databox
    """
    #[
    if target_databox is None:
        target_databox = _databoxes.Databox()
    self = selves[0]
    num_columns = len(selves)
    for row, n in enumerate(self.row_names):
        data = _np.hstack(tuple(ds.data[(row,), :].T for ds in selves))
        x = _series.Series(num_columns=num_columns, )
        x.set_data(self.column_dates, data)
        target_databox[n] = x
    return target_databox
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


def retrieve_vector_from_data_array(
    data: _np.ndarray,
    tokens: Iterable[_incidences.Incidence],
    column_zero: int
) -> _np.ndarray:
    """
    """
    rows, columns = _incidences.rows_and_columns_from_tokens(tokens, column_zero, )
    return data[rows, columns].reshape(-1, 1)


def store_vector_in_data_array(
    vector: _np.ndarray,
    data: _np.ndarray,
    tokens: Iterable[_incidences.Incidence],
    column_zero: int
) -> None:
    """
    """
    rows, columns = _incidences.rows_and_columns_from_tokens(tokens, column_zero, )
    data[rows, columns] = vector


