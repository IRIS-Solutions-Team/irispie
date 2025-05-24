"""
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )
import numpy as _np

from . import _invariants as _invariants

if TYPE_CHECKING:
    from typing import (Any, Self, Iterable, )
    from numbers import (Real, )
    from ..databoxes.main import (Databox, )
#]


class Variant:
    """
    """
    #[

    __slots__ = (
        "data",
    )

    def __init__(self, ) -> None:
        """
        """
        self.data = None

    @classmethod
    def from_databox_variant(
        klass,
        databox_v: Databox | dict,
        invariant: _invariants.Invariant,
        fallbacks: dict[str, Real] | None = None,
        overwrites: dict[str, Real] | None = None,
        clip_data_to_base_span: bool = False,
    ) -> None:
        """
        """
        def _create_nan_vector() -> _np.ndarray:
            return _np.full((invariant.num_periods, ), _np.nan, dtype=_np.float64, )
        self = klass()
        data_list = []
        for n in invariant.names:
            new_data = _create_nan_vector()
            if n in databox_v:
                new_data[:] = databox_v[n]
            data_list.append(new_data, )
        #
        self.data = _np.vstack(data_list, )
        nonbase_columns = invariant.nonbase_columns
        if clip_data_to_base_span and nonbase_columns:
            self.data[:, nonbase_columns] = _np.nan
        self._apply_fallbacks(fallbacks, invariant, )
        self._apply_overwrites(overwrites, invariant, )
        return self

    @classmethod
    def nan_data_array(
        klass,
        invariant: _invariants.Invariant,
    ) -> Self:
        """
        """
        self = klass()
        self.data = _np.full((invariant.num_names, invariant.num_periods, ), _np.nan, dtype=_np.float64, )
        return self

    def copy(self, ) -> Self:
        """
        """
        new = type(self)()
        new.data = self.data.copy()
        return new

    def update_columns_from(
        self,
        other: Self,
        columns: Iterable[int],
    ) -> None:
        """
        """
        columns = tuple(columns, )
        self.data[:, columns] = other.data[:, columns]

    def remove_periods_from_start(
        self,
        num_periods_to_remove: int,
    ) -> None:
        """
        """
        if num_periods_to_remove > 0:
            self.data = self.data[:, num_periods_to_remove:]

    def remove_periods_from_end(
        self,
        num_periods_to_remove: int,
    ) -> None:
        """
        """
        if num_periods_to_remove > 0:
            self.data = self.data[:, :-num_periods_to_remove]

    def add_periods_to_end(
        self,
        num_periods_to_add: int,
    ) -> None:
        if num_periods_to_add > 0:
            padding = ((0, 0), (0, num_periods_to_add))
            self.data = _np.pad(
                self.data,
                padding,
                mode="constant",
                constant_values=_np.nan,
            )

    def retrieve_record(
        self,
        record_id: int,
        #
        columns: slice | Iterable[int] | None = None,
    ) -> _np.ndarray:
        """
        """
        if columns is None:
            columns = ...
        return self.data[record_id, columns]

    def iter_data(self, ) -> _np.ndarray:
        """
        """
        return iter(self.data)

    def store_record(
        self,
        values: _np.ndarray,
        record_id: int,
        columns: slice | Iterable[int] | None = None,
    ) -> None:
        """
        """
        if columns is None:
            self.data[record_id, :] = values
        else:
            self.data[record_id, columns] = values

    def _apply_fallbacks(
        self,
        fallbacks: dict[int, Real] | None,
        invariant: _invariants.Invariant,
    ) -> None:
        """
        """
        if not fallbacks:
            return
        for record_id, name in enumerate(invariant.names, ):
            if name not in fallbacks:
                continue
            values = self.retrieve_record(record_id, )
            index_nan = _np.isnan(values)
            values[index_nan] = _np.float64(fallbacks[name])
            self.store_record(values, record_id, )

    def _apply_overwrites(
        self,
        overwrites: dict[str, Real] | None,
        invariant: _invariants.Invariant,
    ) -> None:
        """
        """
        if not overwrites:
            return
        for record_id, name in enumerate(invariant.names, ):
            if name not in overwrites:
                continue
            values = self.retrieve_record(record_id, )
            values[:] = _np.float64(overwrites[name])
            self.store_record(values, record_id, )

    def logarithmize(self, logly_indexes: tuple[int], ) -> None:
        r"""
        Logarithmize data flagged as logarithmic
        """
        if logly_indexes:
            self.data[logly_indexes, :] = _np.log(self.data[logly_indexes, :])

    def delogarithmize(self, logly_indexes: tuple[int], ) -> None:
        r"""
        Delogarithmize data flagged as logarithmic
        """
        if logly_indexes:
            self.data[logly_indexes, :] = _np.exp(self.data[logly_indexes, :])

    def rescale_data(self, factor: Real, ) -> None:
        r"""
        Rescale data by a common factor
        """
        self.data *= factor

    #]

