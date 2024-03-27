"""
"""


#[
from __future__ import annotations

from typing import (Any, Self, Iterable, )
import numpy as _np

from .. import dates as _dates
from ..databoxes import main as _databoxes
from . import _invariants as _invariants
#]


class Variant:
    """
    """
    #[

    __slots__ = (
        "data",
        "tag_alongs",
    )

    def __init__(
        self,
        /,
        tag_alongs: dict[str, Any] | None = None,
    ) -> None:
        """
        """
        self.data = None
        self.tag_alongs = tag_alongs or {}

    @classmethod
    def from_databox_variant(
        klass,
        databox_v: _databoxes.Databox | dict,
        invariant: _invariants.Invariant,
        /,
        fallbacks: dict[str, Number] | None = None,
        overwrites: dict[str, Number] | None = None,
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
        self.data = _np.vstack(data_list, )
        self._fill_fallbacks(fallbacks, invariant, )
        self._fill_overwrites(overwrites, invariant, )
        return self

    @classmethod
    def nan_data_array(
        klass,
        invariant: _invariants.Invariant,
        /,
    ) -> Self:
        """
        """
        self = klass()
        self.data = _np.full((invariant.num_names, invariant.num_periods, ), _np.nan, dtype=_np.float64, )
        return self

    def copy(self, /) -> Self:
        """
        """
        return type(self).from_data_array(self.data.copy(), )

    def remove_periods_from_start(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove:
            self.data = self.data[:, remove:]

    def remove_periods_from_end(
        self,
        remove: int,
        /,
    ) -> None:
        """
        """
        if remove:
            self.data = self.data[:, :-remove]

    def retrieve_record(
        self,
        record_id: int,
        /,
        columns: slice | Iterable[int] | None = None,
    ) -> _np.ndarray:
        """
        """
        columns = columns if columns is not None else ...
        return self.data[record_id, columns]

    def iter_data(self, /, ) -> _np.ndarray:
        """
        """
        return iter(self.data)

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

    def _fill_fallbacks(
        self,
        fallbacks: dict[str, Number] | None,
        invariant: _invariants.Invariant,
        /,
    ) -> None:
        """
        """
        if not fallbacks:
            return
        for record_id, name in enumerate(invariant.names, ):
            if name in fallbacks:
                values = self.retrieve_record(record_id, )
                index_nan = _np.isnan(values)
                values[index_nan] = _np.float64(fallbacks[name])
                self.store_record(values, record_id, )

    def _fill_overwrites(
        self,
        overwrites: dict[str, Number] | None,
        invariant: _invariants.Invariant,
        /,
    ) -> None:
        """
        """
        if not overwrites:
            return
        for record_id, name in enumerate(invariant.names, ):
            if name in overwrites:
                values = self.retrieve_record(record_id, )
                values[:] = _np.float64(overwrites[name])
                self.store_record(values, record_id, )

