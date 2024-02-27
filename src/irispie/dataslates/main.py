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

from ..series import main as _series
from ..databoxes import main as _databoxes
from ..plans import main as _plans
from ..incidences import main as _incidences
from ..conveniences import iterators as _iterators
from .. import dates as _dates
from .. import has_variants as _has_variants

from . import _invariants as _invariants
from . import _variants as _variants
#]


__all__ = (
    "Dataslate",
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
class Dataslate(
    _has_variants.HasVariantsMixin,
):
    """
    """
    #[

    _invariant: _invariants.Invariant | None = None
    _variants: list[_variants.Variant] | None = None

    @classmethod
    def skeleton(
        klass,
        other: Self,
    ) -> Self:
        """
        """
        self = klass()
        self._invariant = other._invariant
        self._variants = []
        return self

    @classmethod
    def nan_from_names_dates(
        klass,
        names: Iterable[str],
        dates: Iterable[_dates.Dater],
        /,
        num_variants: int = 1,
        **kwargs,
    ) -> Iterator[Self]:
        """
        """
        names = tuple(names or databox.keys())
        dates = tuple(dates)
        num_names = len(names)
        num_dates = len(dates)
        self = klass()
        self._invariant = _invariants.Invariant(names, dates, **kwargs, )
        self._variants = [
            _variants.Variant.nan_data_array(self._invariant, )
            for _ in range(num_variants, )
        ]
        return self

    @classmethod
    def from_databox(
        klass,
        databox: _databoxes.Databox | dict,
        names: Iterable[str],
        dates: Iterable[_dates.Dater],
        /,
        num_variants: int = 1,
        fallbacks: dict[str, Number] | None = None,
        overwrites: dict[str, Number] | None = None,
        **kwargs,
    ) -> Self:
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
        fallbacks_variant_iterator = (
            _databoxes.Databox.iter_variants(fallbacks, item_iterator=item_iterator, names=names, )
            if fallbacks else _it.repeat(None, )
        )
        #
        overwrites_variant_iterator = (
            _databoxes.Databox.iter_variants(overwrites, item_iterator=item_iterator, names=names, )
            if overwrites else _it.repeat(None, )
        )
        #
        zipped = zip(
            range(num_variants),
            databox_variant_iterator,
            fallbacks_variant_iterator,
            overwrites_variant_iterator,
        )
        #
        self = klass()
        self._invariant = _invariants.Invariant(names, dates, **kwargs, )
        self._variants = [
            _variants.Variant.from_databox_variant(
                databox_v, self._invariant,
                fallbacks=fallbacks_v,
                overwrites=overwrites_v,
            )
            for vid, databox_v, fallbacks_v, overwrites_v in zipped
        ]
        return self

    @classmethod
    def from_databox_for_slatable(
        klass,
        slatable: SlatableProtocol,
        databox: _databoxes.Databox | dict,
        base_span: Iterable[_dates.Dater],
        /,
        extra_databox_names: Iterable[str] | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        names = tuple(slatable.get_databox_names())
        if extra_databox_names:
            names = names + tuple(i for i in extra_databox_names if i not in names)
        dates, base_columns, *min_max_shift = _get_extended_range(slatable, base_span, )
        qid_to_logly = slatable.create_qid_to_logly()
        #
        return klass.from_databox(
            databox, names, dates,
            fallbacks=slatable.get_fallbacks(),
            overwrites=slatable.get_overwrites(),
            base_columns=base_columns,
            scalar_names=slatable.get_scalar_names(),
            min_max_shift=min_max_shift,
            qid_to_logly=qid_to_logly,
            **kwargs,
        )

    @property
    def descriptions(self, /, ) -> tuple[str, ...] | None:
        return self._invariant.descriptions

    @property
    def boolex_logly(self, /, ) -> tuple[str, ...] | None:
        return self._invariant.boolex_logly

    @property
    def dates(self, /, ) -> tuple[str]:
        """
        """
        return self._invariant.dates

    @property
    def names(self, /, ) -> tuple[str]:
        """
        """
        return self._invariant.names

    @property
    def num_names(self, /, ) -> int:
        """
        """
        return len(self.names)

    @property
    def num_initials(self, /, ) -> int:
        """
        """
        return -self._invariant.min_max_shift[0]

    @property
    def num_terminals(self, /, ) -> int:
        """
        """
        return self._invariant.min_max_shift[1]

    def copy(self, /, ) -> Self:
        """
        """
        new = self.skeleton(self, )
        new._variants = [i.copy() for i in self._variants]
        return new

    def nan_copy(self, /, ) -> Self:
        """
        """
        new = self.skeleton(self, )
        new._variants = [
            type(i).nan_data_array(new._invariant, )
            for i in self._variants
        ]
        return new

    def remove_initial_data(self, /, ) -> None:
        """
        """
        if self.num_initials:
            self.data = self.data[:, self.num_initials:]

    def to_databox(
        self,
        /,
        target_databox: _databoxes.Databox | None = None,
    ) -> _databoxes.Databox:
        """
        Add data from a dataslate to a new or existing databox
        """
        #[
        if target_databox is None:
            target_databox = _databoxes.Databox()
        num_names = self.num_names
        num_variants = self.num_variants
        start_date = self._invariant.dates[0]
        descriptions = self._invariant.descriptions
        zipped = enumerate(zip(self._invariant.names, self._invariant.databox_value_creator, ), )
        for record_id, (name, creator) in zipped:
            values = _np.vstack(tuple(self._generate_record_from_all_variants(record_id, )))
            target_databox[name] = creator(
                num_variants=num_variants,
                start_date=start_date,
                values=values,
                description=descriptions[record_id],
            )
        return target_databox
        #]

    def _generate_record_from_all_variants(
        self,
        record_id: int,
        /,
    ) -> Iterator[_np.ndarray]:
        """
        """
        return ( v.retrieve_record(record_id, ) for v in self._variants )

    for n in ["num_periods", "from_to", "num_row", "base_slice", "base_columns", "nonbase_columns", ]:
        exec(f"@property\ndef {n}(self, /, ): return self._invariant.{n}", )

    def get_data_variant(
        self,
        vid: int | None = None,
    ) -> _np.ndarray:
        """
        """
        vid = vid or 0
        return self._variants[vid].data

    get_data_array_variant = get_data_variant

    def remove_initial(self, /, ) -> None:
        """
        """
        remove = -self._invariant.min_max_shift[0]
        self.remove_periods_from_start(remove, )

    def remove_terminal(self, /, ) -> None:
        """
        """
        remove = self._invariant.min_max_shift[1]
        self.remove_periods_from_end(remove, )

    def create_name_to_row(
        self,
        /,
    ) -> dict[str, int]:
        return self._invariant.create_name_to_row()

    def remove_periods_from_start(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove < 0:
            raise ValueError("Cannot remove negative number of columns")
        self._invariant.remove_periods_from_start(remove, )
        for v in self._variants:
            v.remove_periods_from_start(remove, )

    def remove_periods_from_end(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove < 0:
            raise ValueError("Cannot remove negative number of columns")
        self._invariant.remove_periods_from_end(remove, )
        for v in self._variants:
            v.remove_periods_from_end(remove, )

    #]


#class VerticalDataslate(_Dataslate, ):
#    """
#    """
#    #[
#    _record_reshape = (-1, 1, )
#
#    @property
#    def row_dates(self, /, ) -> tuple[_dates.Dater, ...]:
#        """
#        """
#        return self.dates
#
#    @property
#    def column_names(self, /, ) -> tuple[str]:
#        """
#        """
#        return self.names
#
#    @property
#    def base_rows(self, /, ) -> tuple[int, ...]:
#        """
#        """
#        return self._base_indexes
#
#    def retrieve_record(
#        self,
#        record_id: int,
#        /,
#        slice_: slice | None = None,
#    ) -> _np.ndarray:
#        """
#        """
#        if slice_ is None:
#            return self.data[:, record_id]
#        else:
#            return self.data[slice_, record_id]
#
#    def _remove_terminal_data(
#        self,
#        last_base_index: int,
#        /,
#    ) -> None:
#        self.data = self.data[:last_base_index+1, :]
#
#    def store_record(
#        self,
#        values: _np.ndarray,
#        record_id: int,
#        /,
#        slice_: slice | None = None,
#    ) -> None:
#        """
#        """
#        if slice_ is None:
#            self.data[:, record_id] = values
#        else:
#            self.data[slice_, record_id] = values
#
#    #]

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
    base_columns = tuple(_dates.date_index(base_span, start_date))
    extended_dates = tuple(_dates.Ranger(start_date, end_date))
    return extended_dates, base_columns, min_shift, max_shift


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


def retrieve_vector_from_data_array(
    data: _np.ndarray,
    tokens: tuple[str, ...],
    index_zero: int,
    /,
) -> _np.ndarray:
    """
    """
    rows, columns = _incidences.rows_and_columns_from_tokens(tokens, index_zero, )
    return data[rows, columns]


def store_vector_in_data_array(
    vector: _np.ndarray,
    data: _np.ndarray,
    tokens: Iterable[_incidences.Incidence],
    index_zero: int,
) -> None:
    """
    """
    rows, columns = _incidences.rows_and_columns_from_tokens(tokens, index_zero, )
    data[rows, columns] = vector

