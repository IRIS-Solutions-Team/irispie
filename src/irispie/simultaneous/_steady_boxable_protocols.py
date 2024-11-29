"""
Implement SteadyDataboxableProtocol
"""


#[

from __future__ import annotations

import numpy as _np

from .. import has_variants as _has_variants
from ..quantities import QuantityKind
from ..series.main import Series

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable
    from numbers import Real
    from ..dates import Period

#]


"""
Quantities that are time series in model databoxes
"""
_TIME_SERIES_QUANTITY = QuantityKind.ANY_VARIABLE | QuantityKind.ANY_SHOCK


# REFACTOR:
# Create a SteadyBoxable class
class Inlay:
    def generate_steady_items(
        self,
        start: Period,
        end: Period,
        /,
        deviation: bool = False,
        unpack_singleton: bool = True,
    ) -> Iterable[tuple[str, Series | Real]]:
        """
        """
        num_periods = int(end - start + 1)
        shift_in_first_column = start.get_distance_from_origin()
        qid_to_name = self.create_qid_to_name()
        qid_to_description = self.create_qid_to_description()
        #
        tuple_of_arrays = tuple(
            self.create_some_array(
            variant=variant,
            deviation=deviation,
            num_columns=num_periods,
            shift_in_first_column=shift_in_first_column,
        ) for variant in self._variants)
        array = _np.stack(arrays=tuple_of_arrays, axis=2, )
        #
        num_rows = array.shape[0]
        qid_to_kind = self.create_qid_to_kind()
        for qid in qid_to_name.keys():
            name = qid_to_name[qid]
            array_slice = array[qid, :, :]
            is_time_series = qid_to_kind[qid] in _TIME_SERIES_QUANTITY
            if is_time_series:
                value = Series(
                    start=start,
                    values=array_slice,
                    description=qid_to_description[qid],
                )
            else:
                value = _has_variants.unpack_singleton(
                    array_slice[0, :].tolist(), self.is_singleton,
                    unpack_singleton=unpack_singleton,
                )
            yield name, value,

