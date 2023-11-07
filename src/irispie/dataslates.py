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
    "HorizontalDataslate",
    "VerticalDataslate",
)


class SlatableProtocol(Protocol, ):
    """
    """
    def get_min_max_shift(self, /, ) -> tuple[int, int]:  ...
    def get_databox_names(self, /, ) -> tuple[str, ...]:  ...


@_dc.dataclass
class _Dataslate:
    """
    """
    #[

    data: _np.ndarray | None = None
    missing_names: tuple[str, ...] | None = None
    base_indices: tuple[int, ...] | None = None
    _names: Iterable[str] | None = None
    _descriptions: Iterable[str] | None = None
    _dates: tuple[_dates.Dater, ...] | None = None
    _record_shape: tuple[int, int] | None = None

    @classmethod
    def for_slatables(
        klass,
        slatables: SlatableProtocol,
        databox: _databoxes.Databox,
        base_range: Iterable[_dates.Dater],
        /,
        variant: int = 0,
    ) -> Self:
        """
        """
        self = klass()
        self._names = tuple()
        for s in slatables:
            self._names += tuple(
                n for n in s.get_databox_names()
                if n not in self._names
            )
        self.missing_names = tuple(databox.get_missing_names(self._names, ))
        self._dates, self._base_indices = get_extended_range(slatables[0], base_range, )
        self._populate_data(databox, variant, )
        self._populate_descriptions(databox, )
        return self

    @classmethod
    def from_databox(
        klass,
        databox: _databoxes.Databox,
        names: Iterable[str],
        base_range: Iterable[_dates.Dater],
        /,
        variant: int = 0,
    ) -> Self:
        """
        """
        self = klass()
        self._names = tuple(names)
        self.missing_names = tuple(databox.get_missing_names(self._names, ))
        self._dates = tuple(base_range)
        self._base_indices = tuple(range(len(self._dates)))
        self._populate_data(databox, variant, )
        self._populate_descriptions(databox, )
        return self

    def _populate_data(
        self,
        databox: _databoxes.Databox,
        variant: int,
    ) -> None:
        """
        """
        data = tuple(
            _extract_data_from_record(databox[n], self.from_to, self.num_periods, variant, ).reshape(*self._record_reshape, )
            if n not in self.missing_names else self.nan_row.reshape(*self._record_reshape, )
            for n in self._names
        )
        self.data = self._stack_in(data, )

    def _populate_descriptions(
        self,
        databox: _databoxes.Databox,
        /,
    ) -> None:
        """
        """
        self._descriptions = tuple(
            _extract_descriptions_from_record(databox[n], )
            if n not in self.missing_names else ""
            for n in self._names
        )

    def to_databox(self, *args, **kwargs, ) -> _dates.Databox:
        """
        """
        return multiple_to_databox((self,), *args, **kwargs, )

    @property
    def num_periods(self, /, ) -> int:
        """
        """
        return len(self._dates)

    @property
    def from_to(self, /, ) -> tuple[_dates.Dater, _dates.Dater]:
        """
        """
        return self._dates[0], self._dates[-1]

    @property
    def nan_row(self, /, ) -> int:
        """
        """
        return _np.full((self.num_periods, ), _np.nan, dtype=float)

    @property
    def num_rows(self, /, ) -> int:
        """
        """
        return self.data.shape[0] if self.data is not None else 0

    @property
    def base_slice(self, /, ) -> slice:
        """
        """
        return slice(self._base_indices[0], self._base_indices[-1]+1)

    def remove_terminal(self, /, ) -> None:
        """
        """
        last_base_index = self._base_indices[-1]
        self.data = self.data[:, :last_base_index+1]
        self._dates = self._dates[:last_base_index+1]

    def copy_data(self, /, ) -> _np.ndarray:
        """
        """
        return self.data.copy()

    def create_name_to_row(
        self,
        /,
    ) -> dict[str, int]:
        return { name: row for row, name in enumerate(self._names, ) }

    @staticmethod
    def retrieve_vector_from_data_array(
        data: _np.ndarray,
        tokens: Iterable[_incidences.Incidence],
        index_zero: int,
        /,
    ) -> _np.ndarray:
        ...

    def retrieve_vector(
        self,
        tokens: tuple[str, ...],
        index_zero: int,
        /,
    ) -> _np.ndarray:
        """
        """
        return self.retrieve_vector_from_data_array(self.data, tokens, index_zero, )

    @staticmethod
    def store_vector_in_data_array(
        vector: _np.ndarray,
        data: _np.ndarray,
        tokens: Iterable[_incidences.Incidence],
        index_zero: int,
        /,
    ) -> None:
        ...

    def store_vector(
        self,
        tokens: tuple[str, ...],
        vector: _np.ndarray,
        index_zero: int,
        /,
    ) -> None:
        """
        """
        store_vector_in_horizontal_data_array(vector, self.data, tokens, index_zero, )

    def retrieve_record(
        self,
        record_id: int,
        /,
        slice_: slice | None = None,
    ) -> _np.ndarray:
        """
        """
        ...

    def store_record(
        self,
        values: _np.ndarray,
        record_id: int,
        /,
        slice_: slice | None = None,
    ) -> None:
        """
        """
        ...

    def fill_missing_in_base_periods(
        self,
        names,
        /,
        fill: Number = 0,
    ) -> None:
        """
        """
        for i, n in enumerate(self._names):
            if names is not Ellipsis and n not in names:
                continue
            values = self.retrieve_record(i, self._base_indices, )
            values[_np.isnan(values)] = fill
            self.store_record(values, i, self._base_indices, )

    #]


class HorizontalDataslate(_Dataslate, ):
    """
    """
    #[
    _record_reshape = (1, -1, )

    @staticmethod
    def _stack_in(data, /, ) -> _np.ndarray:
        """
        """
        return _np.vstack(data)

    @staticmethod
    def retrieve_vector_from_data_array(
        data: _np.ndarray,
        tokens: Iterable[_incidences.Incidence],
        index_zero: int,
        /,
    ) -> _np.ndarray:
        """
        """
        rows, columns = _incidences.rows_and_columns_from_tokens_in_horizontal(tokens, index_zero, )
        return data[rows, columns].reshape(-1, 1)

    @staticmethod
    def store_vector_in_data_array(
        vector: _np.ndarray,
        data: _np.ndarray,
        tokens: Iterable[_incidences.Incidence],
        index_zero: int,
    ) -> None:
        """
        """
        rows, columns = _incidences.rows_and_columns_from_tokens_in_horizontal(tokens, index_zero, )
        data[rows, columns] = vector

    def remove_periods_from_start(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove < 0:
            raise ValueError("Cannot remove negative number of columns from start")
        if remove:
            self.data = self.data[:, remove:]
            self._dates = self._dates[remove:]

    def remove_periods_from_end(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove > 0:
            raise ValueError("Cannot remove positive number of columns from end")
        if remove:
            self.data = self.data[:, :remove]
            self._dates = self._dates[:remove]

    @property
    def column_dates(self, /, ) -> tuple[_dates.Dater, ...]:
        """
        """
        return self._dates

    @property
    def base_columns(self, /, ) -> tuple[int, ...]:
        """
        """
        return self._base_indices

    @property
    def row_names(self, /, ) -> Iterable[str]:
        """
        """
        return self._names

    def retrieve_record(
        self,
        record_id: int,
        /,
        columns: slice | Iterable[int] | None = None,
    ) -> _np.ndarray:
        """
        """
        if columns is None:
            return self.data[record_id, :]
        else:
            return self.data[record_id, columns]

    def store_record(
        self,
        values: _np.ndarray,
        record_id: int,
        /,
        columns: slice | Iterable[int] | None = None,
    ) -> None:
        """
        """
        if columns is None:
            self.data[record_id, :] = values
        else:
            self.data[record_id, columns] = values

    #]


class VerticalDataslate(_Dataslate, ):
    """
    """
    #[
    _record_reshape = (-1, 1, )

    @staticmethod
    def _stack_in(data, /, ) -> _np.ndarray:
        """
        """
        return _np.hstack(data)

    @property
    def row_dates(self, /, ) -> tuple[_dates.Dater, ...]:
        """
        """
        return self._dates

    @property
    def column_names(self, /, ) -> tuple[str]:
        """
        """
        return self._names

    def retrieve_record(
        self,
        record_id: int,
        /,
        slice_: slice | None = None,
    ) -> _np.ndarray:
        """
        """
        if slice_ is None:
            return self.data[:, record_id]
        else:
            return self.data[slice_, record_id]

    def store_record(
        self,
        values: _np.ndarray,
        record_id: int,
        /,
        slice_: slice | None = None,
    ) -> None:
        """
        """
        if slice_ is None:
            self.data[:, record_id] = values
        else:
            self.data[slice_, record_id] = values

    #]


def multiple_to_databox(
    slates,
    /,
    target_databox: _databoxes.Databox | None = None,
) -> _databoxes.Databox:
    """
    Add data from a dataslate to a new or existing databox
    """
    #[
    if target_databox is None:
        target_databox = _databoxes.Databox()
    num_columns = len(slates)
    start_date = slates[0]._dates[0]
    for record_id, n in enumerate(slates[0]._names):
        data = _np.hstack(tuple(
            ds.retrieve_record(record_id, ).reshape(-1, 1)
            for ds in slates
        ))
        target_databox[n] = _series.Series(
            num_columns=num_columns,
            start_date=start_date,
            values=data,
            description=slates[0]._descriptions[record_id],
        )
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
    base_indices = tuple(_dates.date_index(base_range, start_date))
    # base_indices = tuple(range(-min_shift, -min_shift+num_base_periods))
    extended_dates = tuple(_dates.Ranger(start_date, end_date))
    return extended_dates, base_indices


def _extract_data_from_record(record, from_to, num_periods, column, /, ):
    """
    """
    try:
        # Record is a time series
        return record.get_data_column_from_to(from_to, column)
    except AttributeError:
        # Record is a numeric scalar
        return _np.full((num_periods, ), float(record), dtype=float, )


def _extract_descriptions_from_record(record, /, ):
    """
    """
    try:
        # Record is a time series
        return str(record.get_description())
    except AttributeError:
        # Record is a numeric scalar
        return ""
    except ValueError:
        # Record is a numeric scalar
        return ""

