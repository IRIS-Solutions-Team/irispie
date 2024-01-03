"""
"""


#[
from __future__ import annotations

import dataclasses as _dc
import numpy as _np

from .. import dates as _dates
from ..databoxes import main as _databoxes
from . import _invariants as _invariants
#]


@_dc.dataclass(slots=True, )
class Variant:
    """
    """
    #[

    data: _np.ndarray | None = None

    def __init__(
        self,
        databox: _databoxes.Databox | dict,
        invariant: _invariants.Invariant,
        /,
        fallbacks: dict[str, Number] | None = None,
        overwrites: dict[str, Number] | None = None,
    ) -> None:
        """
        """
        def _create_nan_vector() -> _np.ndarray:
            return _np.full((invariant.num_periods, ), _np.nan, dtype=_np.float64, )
        data = []
        for n in invariant.names:
            new_data = _create_nan_vector()
            if n in databox:
                new_data[:] = databox[n]
            data.append(new_data, )
        self.data = _np.vstack(data, )
        self._fill_fallbacks(fallbacks, invariant, )
        self._fill_overwrites(overwrites, invariant, )

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

