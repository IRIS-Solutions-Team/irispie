"""
Data arrays with row and column names
"""


#[
from __future__ import annotations

from typing import (Self, Protocol, )
from numbers import (Number, )
from collections.abc import (Iterable, Iterator, )
import numpy as _np
import dataclasses as _dc
import numpy as _np
import functools as _ft
import itertools as _it

from .series import main as _series
from .databoxes import main as _databoxes
from .plans import main as _plans
from .incidences import main as _incidences
from .conveniences import iterators as _iterators
from . import dates as _dates
#]


__all__ = (
    "HorizontalDataslate",
    "VerticalDataslate",
    "multiple_to_databox",
)


class SlatableProtocol(Protocol, ):
    """
    """
    def get_min_max_shift(self, /, ) -> tuple[int, int]: ...
    def get_databox_names(self, /, ) -> tuple[str, ...]: ...
    def get_fallbacks(self, /, ) -> dict[str, Number]: ...
    def get_overwrites(self, /, ) -> dict[str, Number]: ...
    def get_scalar_names(self, /, ) -> Iterable[str, ...]: ...
    def create_qid_to_logly(self, /, ) -> dict[int, bool]: ...


@_dc.dataclass
class _Dataslate:
    """
    """
    #[

    data: _np.ndarray | None = None
    missing_names: tuple[str, ...] | None = None
    descriptions: Iterable[str] | None = None
    boolex_logly: tuple[bool, ...] | None = None
    _base_indexes: tuple[int, ...] | None = None
    _names: Iterable[str] | None = None
    _dates: tuple[_dates.Dater, ...] | None = None
    _record_shape: tuple[int, int] | None = None
    _min_max_shift: tuple[int, int] | None = None
    _boolex_scalar: tuple[bool, ...] | None = None

    @classmethod
    def from_databox_variant(
        klass,
        databox_variant: _databoxes.Databox | dict,
        names: Iterable[str],
        dates: Iterable[_dates.Dater],
        /,
        *,
        base_indexes: Iterable[int] = None,
        descriptions: dict[str, str | None] | None = None,
        fallbacks: dict[str, Number] | None = None,
        overwrites: dict[str, Number] | None = None,
        scalar_names: Iterable[str] | None = None,
        qid_to_logly: dict[int, bool | None] | None = None,
        min_max_shift: tuple[int, int] = (0, 0),
    ) -> Self:
        """
        Create a dataslate from a databox or dict variant
        """
        self = klass()
        self._names = tuple(names)
        self.missing_names = tuple(_databoxes.Databox.get_missing_names(databox_variant, self._names, ))
        self._dates = dates
        self._base_indexes = base_indexes
        self._populate_data(databox_variant, )
        self._populate_descriptions(descriptions, )
        self._populate_boolex_logly(qid_to_logly, )
        self._fill_fallbacks(fallbacks, )
        self._fill_overwrites(overwrites, )
        self._min_max_shift = min_max_shift
        self._populate_boolex_scalar(scalar_names, )
        return self

    @classmethod
    def iter_variants_from_databox_for_slatable(
        klass,
        slatable: SlatableProtocol,
        databox: _databoxes.Databox | dict,
        base_span: Iterable[_dates.Dater],
        /,
    ) -> Iterator[Self]:
        """
        """
        names = slatable.get_databox_names()
        dates, base_indexes, *min_max_shift = \
            _get_extended_range(slatable, base_span, )
        qid_to_logly = slatable.create_qid_to_logly()
        return klass.iter_variants_from_databox(
            databox, names, dates,
            base_indexes=base_indexes,
            fallbacks=slatable.get_fallbacks(),
            overwrites=slatable.get_overwrites(),
            scalar_names=slatable.get_scalar_names(),
            min_max_shift=min_max_shift,
            qid_to_logly=qid_to_logly,
        )

    @classmethod
    def iter_variants_from_databox(
        klass,
        databox: _databoxes.Databox | dict,
        names: Iterable[str] | None,
        dates: Iterable[_dates.Dater],
        /,
        *,
        fallbacks: dict[str, Number] | None = None,
        overwrites: dict[str, Number] | None = None,
        scalar_names: Iterable[str] | None = None,
        **kwargs,
    ) -> Iterator[Self]:
        """
        """
        names = tuple(names or databox.keys())
        dates = tuple(dates)
        #
        from_to = dates[0], dates[-1] if dates else ()
        item_iterator = \
            _ft.partial(_slate_value_variant_iterator, from_to=from_to, )
        #
        databox_variant_iterator = \
            _databoxes.Databox.iter_variants(databox, item_iterator=item_iterator, names=names, )
        #
        fallbacks_variant_iterator = \
            _databoxes.Databox.iter_variants(fallbacks, item_iterator=item_iterator, names=names, ) \
            if fallbacks else _it.repeat(None, )
        #
        overwrites_variant_iterator = \
            _databoxes.Databox.iter_variants(overwrites, item_iterator=item_iterator, names=names, ) \
            if overwrites else _it.repeat(None, )
        #
        for databox_variant, fallbacks_variant, overwrites_variant \
            in zip(databox_variant_iterator, fallbacks_variant_iterator, overwrites_variant_iterator, ):
            #
            yield klass.from_databox_variant(
                databox_variant, names, dates,
                descriptions=_retrieve_descriptions(databox_variant, names, ),
                fallbacks=fallbacks_variant,
                overwrites=overwrites_variant,
                scalar_names=scalar_names,
                **kwargs,
            )

    @classmethod
    def from_databox_for_slatable(
        klass,
        *args,
        variant: int = 0,
        **kwargs,
    ) -> Self:
        """
        """
        iterator = klass.iter_variants_from_databox_for_slatable(*args, **kwargs, )
        for _ in range(variant, ):
            next(iterator, )
        return next(iterator, )

    @classmethod
    def from_databox(
        klass,
        *args,
        variant: int = 0,
        **kwargs,
    ) -> Self:
        """
        """
        iterator = klass.iter_variants_from_databox(*args, **kwargs, )
        for _ in range(variant, ):
            next(iterator, )
        return next(iterator, )

    @property
    def num_names(self, /, ) -> int:
        """
        """
        return len(self._names)

    @property
    def num_initials(self, /, ) -> int:
        """
        """
        return -self._min_max_shift[0]

    @property
    def num_terminals(self, /, ) -> int:
        """
        """
        return self._min_max_shift[1]

    def remove_initial_data(self, /, ) -> None:
        """
        """
        if self.num_initials:
            self.data = self.data[:, self.num_initials:]

    def _populate_data(
        self,
        databox: _databoxes.Databox | dict,
        /,
        ) -> None:
        """
        Populate slatable data from databox or dict using the first variant
        """
        data = []
        for n in self._names:
            new_data = self.create_nan_vector()
            if n in databox:
                new_data[:] = databox[n]
            data.append(new_data, )
        self.data = self._stack_in(data, )

    def _populate_descriptions(
        self,
        descriptions: dict[str, str] | None,
    ) -> None:
        """
        """
        descriptions = descriptions or {}
        self.descriptions = tuple(
            descriptions.get(n, "") or ""
            for n in self._names
        )

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
            for i in range(len(self._names), )
        )

    def _populate_boolex_scalar(
        self,
        scalar_names: Iterable[str] | None,
        /,
    ) -> None:
        """
        """
        if not scalar_names:
            return
        self._boolex_scalar = tuple(
            n in scalar_names
            for n in self._names
        )

    def _fill_fallbacks(
        self,
        fallbacks: dict[str, Number] | None,
        /,
    ) -> None:
        """
        """
        if not fallbacks:
            return
        for record_id, name in enumerate(self._names, ):
            if name in fallbacks:
                record = self.retrieve_record(record_id, )
                index_nan = _np.isnan(record)
                record[index_nan] = _np.float64(fallbacks[name])
                self.store_record(record, record_id, )

    def _fill_overwrites(
        self,
        overwrites: dict[str, Number] | None,
        /,
    ) -> None:
        """
        """
        if not overwrites:
            return
        for record_id, name in enumerate(self._names, ):
            if name in overwrites:
                record = self.retrieve_record(record_id, )
                record[:] = _np.float64(overwrites[name])
                self.store_record(record, record_id, )

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

    def create_nan_vector(self, /, ) -> int:
        """
        """
        return _np.full((self.num_periods, ), _np.nan, dtype=_np.float64, )

    @property
    def num_rows(self, /, ) -> int:
        """
        """
        return self.data.shape[0] if self.data is not None else 0

    @property
    def base_slice(self, /, ) -> slice:
        """
        """
        return slice(self._base_indexes[0], self._base_indexes[-1]+1)

    def remove_terminal(self, /, ) -> None:
        """
        """
        last_base_index = self._base_indexes[-1]
        self._dates = self._dates[:last_base_index+1]
        self._remove_terminal_data(last_base_index, )

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

    def _remove_terminal_data(self, /, ) -> None:
        """
        """
        ...

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
        return _np.vstack(data, )

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

    def _remove_terminal_data(
        self,
        last_base_index: int,
        /,
    ) -> None:
        self.data = self.data[:, :last_base_index+1]

    @property
    def column_dates(self, /, ) -> tuple[_dates.Dater, ...]:
        """
        """
        return self._dates

    @property
    def base_columns(self, /, ) -> tuple[int, ...]:
        """
        """
        return self._base_indexes

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

    @property
    def base_rows(self, /, ) -> tuple[int, ...]:
        """
        """
        return self._base_indexes

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

    def _remove_terminal_data(
        self,
        last_base_index: int,
        /,
    ) -> None:
        self.data = self.data[:last_base_index+1, :]

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
    dataslates,
    /,
    target_databox: _databoxes.Databox | None = None,
) -> _databoxes.Databox:
    """
    Add data from a dataslate to a new or existing databox
    """
    #[
    if target_databox is None:
        target_databox = _databoxes.Databox()
    num_names = dataslates[0].num_names
    num_variants = len(dataslates)
    start_date = dataslates[0]._dates[0]
    boolex_scalar = dataslates[0]._boolex_scalar or _it.repeat(False, )
    for record_id, name, is_scalar in zip(range(num_names), dataslates[0]._names, boolex_scalar, ):
        data = _np.hstack(tuple(
            ds.retrieve_record(record_id, ).reshape(-1, 1)
            for ds in dataslates
        ))
        if is_scalar:
            value = list(data[0, :])
        else:
            value = _series.Series(
                num_variants=num_variants,
                start_date=start_date,
                values=data,
                description=dataslates[0].descriptions[record_id],
            )
        target_databox[name] = value
    return target_databox
    #]


def _get_extended_range(
    slatable: SlatableProtocol,
    base_span: Iterable[_dates.Dater],
    /,
) -> tuple[Iterable[_dates.Dater], tuple[int, ...]]:
    """
    """
    base_span = tuple(t for t in base_span)
    num_base_periods = len(base_span)
    min_shift, max_shift = slatable.get_min_max_shifts()
    if min_shift == 0:
        min_shift = -1
    min_base_date = min(base_span)
    max_base_date = max(base_span)
    start_date = min_base_date + min_shift
    end_date = max_base_date + max_shift
    base_indexes = tuple(_dates.date_index(base_span, start_date))
    extended_dates = tuple(_dates.Ranger(start_date, end_date))
    return extended_dates, base_indexes, min_shift, max_shift


def _slate_value_variant_iterator(
    value: Any,
    /,
    from_to: tuple[_dates.Dater, _dates.Dater],
) -> Iterator[Any]:
    """
    """
    #[
    if hasattr(value, "iter_data_variants_from_to"):
        return value.iter_data_variants_from_to(from_to, )
    elif isinstance(value, Iterable):
        return _iterators.exhaust_then_last(value, )
    else:
        return _iterators.exhaust_then_last([], value, )
    #]


def _retrieve_descriptions(
    databox: _databoxes.Databox | dict,
    names: Iterable[str],
) -> tuple[str]:
    """
    """
    def _retrieve_item_description(n: str, /, ) -> str:
        if n in databox and hasattr(databox[n], "get_description"):
            return databox[n].get_description()
        else:
            return None
    return { n: _retrieve_item_description(n, ) for n in names }

