"""
Data arrays with row and column names
"""


#[
from __future__ import annotations

from typing import (Self, Protocol, )
from numbers import (Number, )
from collections.abc import (Iterable, Iterator, )
import numpy as _np
import numpy as _np
import functools as _ft
import itertools as _it

from ..series.main import (Series, )
from ..databoxes.main import (Databox, )
from ..incidences import main as _incidences
from ..conveniences import iterators as _iterators
from ..dates import (Period, Span, )
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
    max_lag: int
    max_lead: int
    databox_names: Iterable[str]
    databox_validators: dict[str, Callable]
    fallbacks: dict[str, Real]
    overwrites: dict[str, Real]
    output_names: Iterable[str]
    qid_to_logly: dict[int, bool | None]


class Dataslate(
    _has_variants.HasVariantsMixin,
):
    """
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    def __init__(self, /, ) -> None:
        self._invariant = None
        self._variants = []

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
    def nan_from_names_periods(
        klass,
        names: Iterable[str],
        periods: Iterable[Period] | string,
        /,
        num_variants: int = 1,
        **kwargs,
    ) -> Iterator[Self]:
        """
        """
        names = tuple(names or databox.keys())
        periods = _dates.ensure_period_tuple(periods, )
        num_names = len(names)
        num_periods = len(periods)
        self = klass()
        self._invariant = _invariants.Invariant(names, periods, **kwargs, )
        self._variants = [
            _variants.Variant.nan_data_array(self._invariant, )
            for _ in range(num_variants, )
        ]
        return self

    @classmethod
    def nan_from_template(
        klass,
        other: Self,
        /,
        num_variants: int = 1,
    ) -> Self:
        """
        """
        self = klass.skeleton(other, )
        self._variants = [
            _variants.Variant.nan_data_array(self._invariant, )
            for _ in range(num_variants, )
        ]
        return self

    @classmethod
    def from_databox(
        klass,
        databox: Databox | dict,
        names: Iterable[str],
        periods: Iterable[Period] | string,
        /,
        num_variants: int = 1,
        fallbacks: dict[str, Number] | None = None,
        overwrites: dict[str, Number] | None = None,
        clip_data_to_base_span: bool = False,
        validators: dict[str, Callable] | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        names = tuple(names or databox.keys())
        periods = _dates.ensure_period_tuple(periods, )
        #
        if validators:
            Databox.validate(databox, validators, )
        #
        from_to = periods[0], periods[-1] if periods else ()
        item_iterator = \
            _ft.partial(_slate_value_variant_iterator, from_to=from_to, )
        #
        databox_variant_iterator = \
            Databox.iter_variants(databox, item_iterator=item_iterator, names=names, )
        #
        fallbacks_variant_iterator = (
            Databox.iter_variants(fallbacks, item_iterator=item_iterator, names=names, )
            if fallbacks else _it.repeat(None, )
        )
        #
        overwrites_variant_iterator = (
            Databox.iter_variants(overwrites, item_iterator=item_iterator, names=names, )
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
        self._invariant = _invariants.Invariant(names, periods, **kwargs, )
        self._variants = [
            _variants.Variant.from_databox_variant(
                databox_v, self._invariant,
                fallbacks=fallbacks_v,
                overwrites=overwrites_v,
                clip_data_to_base_span=clip_data_to_base_span,
            )
            for vid, databox_v, fallbacks_v, overwrites_v in zipped
        ]
        return self

    @classmethod
    def from_databox_for_slatable(
        klass,
        slatable: SlatableProtocol,
        databox: Databox | dict,
        base_span: Iterable[Period],
        /,
        extra_databox_names: Iterable[str] | None = None,
        prepend_initial: bool = True,
        append_terminal: bool = True,
        clip_data_to_base_span: bool = False,
        **kwargs,
    ) -> Self:
        """
        """
        names = tuple(slatable.databox_names, ) if slatable.databox_names else ()
        if extra_databox_names:
            names = names + tuple(i for i in extra_databox_names if i not in names)
        periods, base_columns, *min_max_shift = _get_extended_span(
            slatable, base_span,
            prepend_initial=prepend_initial,
            append_terminal=append_terminal,
        )
        #
        return klass.from_databox(
            databox, names, periods,
            base_columns=base_columns,
            min_max_shift=min_max_shift,
            qid_to_logly=slatable.qid_to_logly,
            fallbacks=slatable.fallbacks,
            overwrites=slatable.overwrites,
            clip_data_to_base_span=clip_data_to_base_span,
            output_names=slatable.output_names,
            validators=slatable.databox_validators,
            **kwargs,
        )

    @property
    def descriptions(self, /, ) -> tuple[str, ...] | None:
        return self._invariant.descriptions

    @property
    def logly_indexes(self, /, ) -> tuple[int, ...] | None:
        return self._invariant.logly_indexes

    @property
    def periods(self, /, ) -> tuple[Period]:
        """
        """
        return self._invariant.periods

    @property
    def start(self, /, ) -> Period:
        """
        """
        return self._invariant.periods[0]

    first_period = start

    @property
    def end(self, /, ) -> Period:
        """
        """
        return self._invariant.periods[-1]

    last_period = end

    @property
    def base_periods(self, /, ) -> tuple[Period]:
        """
        """
        return self._invariant.base_periods

    @base_periods.setter
    def base_periods(self, base_periods: Iterable[Period], /, ) -> None:
        """
        """
        self._invariant.base_periods = base_periods

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

    @property
    def output_names(self, /, ) -> tuple[str, ...]:
        """
        """
        return tuple(self.names[i] for i in self._invariant.output_qids)

    def copy(
        self, 
        invariant=True,
        variants=True,
    ) -> Self:
        """
        """
        new = type(self)()
        if invariant:
            new._invariant = self._invariant.copy()
        else:
            new._invariant = self._invariant
        #
        if variants:
            new._variants = [i.copy() for i in self._variants]
        else:
            new._variants = [i for i in self._variants]
        #
        return new

    def rename(self, old_name_to_new_name: dict[str, str], /, ) -> None:
        """
        """
        self._invariant.names = tuple(
            old_name_to_new_name.get(i, i)
            for i in self._invariant.names
        )

    def extend(
        self: Self,
        other: Self,
        /,
    ) -> None:
        """
        """
        # TODO: Check if the other is compatible
        self._variants.extend(other._variants, )

    def nan_copy(self, /, ) -> Self:
        """
        """
        return self.nan_from_template(self, num_variants=self.num_variants, )

    def update_columns_from(
        self,
        other: Self,
        columns: Iterable[int],
        /,
    ) -> None:
        """
        """
        for self_v, other_v in zip(self._variants, other._variants, ):
            self_v.update_columns_from(other_v, columns, )

    def remove_initial_data(self, /, ) -> None:
        """
        """
        if self.num_initials:
            self.data = self.data[:, self.num_initials:]

    def to_databox(
        self,
        /,
        target_db: Databox | None = None,
    ) -> Databox:
        """
        Add data from a dataslate to a new or existing databox
        """
        #[
        if target_db is None:
            target_db = Databox()
        num_names = self.num_names
        num_variants = self.num_variants
        start = self._invariant.periods[0]
        for qid in self._invariant.output_qids:
            name = self._invariant.names[qid]
            description = self._invariant.descriptions[qid]
            values = _np.vstack([
                v.data[qid, :] for v in self._variants
            ]).T
            target_db[name] = Series._guaranteed(
                start=start,
                values=values,
                description=description,
            )
        return target_db
        #]

    for n in ["num_periods", "from_to", "num_row", "base_slice", "base_columns", "nonbase_columns", ]:
        exec(f"{n} = property(lambda self: self._invariant.{n})", )

    def get_data_variant(
        self,
        vid: int = 0,
    ) -> _np.ndarray:
        """
        """
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

    def rescale_data(self, *args, **kwargs, ) -> None:
        """
        Rescale data in all variants by a common factor
        """
        for v in self._variants:
            v.rescale_data(*args, **kwargs, )

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

    def logarithmize(self, /, ) -> None:
        """
        """
        logly_indexes = tuple(self.logly_indexes)
        if not logly_indexes:
            return
        for v in self._variants:
            v.logarithmize(logly_indexes, )

    def delogarithmize(self, /, ) -> None:
        """
        """
        logly_indexes = tuple(self.logly_indexes)
        if not logly_indexes:
            return
        for v in self._variants:
            v.delogarithmize(logly_indexes, )
    #]


def _get_extended_span(
    slatable: SlatableProtocol,
    base_span: Iterable[Period],
    /,
    prepend_initial: bool,
    append_terminal: bool,
) -> tuple[Iterable[Period], tuple[int, ...]]:
    """
    """
    base_span = tuple(t for t in base_span)
    num_base_periods = len(base_span)
    min_shift = slatable.max_lag if prepend_initial else 0
    max_shift = slatable.max_lead if append_terminal else 0
    min_base_date = min(base_span)
    max_base_date = max(base_span)
    start_date = min_base_date + min_shift
    end_date = max_base_date + max_shift
    base_columns = tuple(_dates.date_index(base_span, start_date))
    extended_dates = tuple(Span(start_date, end_date))
    return extended_dates, base_columns, min_shift, max_shift


def _slate_value_variant_iterator(
    value: Any,
    /,
    from_to: tuple[Period, Period],
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
    column_zero: int,
    /,
) -> _np.ndarray:
    """
    """
    rows, columns = _incidences.rows_and_columns_from_tokens(tokens, column_zero, )
    return data[rows, columns]


def store_vector_in_data_array(
    vector: _np.ndarray,
    data: _np.ndarray,
    tokens: Iterable[_incidences.Incidence],
    column_zero: int,
) -> None:
    """
    """
    rows, columns = _incidences.rows_and_columns_from_tokens(tokens, column_zero, )
    data[rows, columns] = vector

