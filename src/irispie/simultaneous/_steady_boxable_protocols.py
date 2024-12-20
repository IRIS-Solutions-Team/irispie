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
    r"""
    ................................................................................
    ==Class: Inlay==

    Facilitates the generation of steady-state items for model databoxes. This class 
    handles the creation and processing of time series and non-time-series quantities 
    for specified periods and variants.

    Attributes:
        - `_variants`: The list of model variants managed by the instance.
        - `is_singleton`: Indicates whether there is only a single variant.
    ................................................................................
    """
    def generate_steady_items(
        self,
        start: Period,
        end: Period,
        /,
        deviation: bool = False,
        unpack_singleton: bool = True,
    ) -> Iterable[tuple[str, Series | Real]]:
        r"""
        ................................................................................
        ==Method: generate_steady_items==

        Generates steady-state items (time series or scalar values) for each quantity 
        in the model databox over the specified time span.

        ### Input arguments ###
        ???+ input "start: Period"
            The start period for the steady-state data.
        ???+ input "end: Period"
            The end period for the steady-state data.
        ???+ input "deviation: bool = False"
            Whether to generate deviations from the steady-state values.
        ???+ input "unpack_singleton: bool = True"
            If `True`, unpacks results when there is only a single variant.

        ### Returns ###
        ???+ returns "Iterable[tuple[str, Series | Real]]"
            An iterable of tuples, where each tuple contains:
            - A string representing the quantity name.
            - A `Series` object or scalar value for the quantity.

        ### Example ###
        ```python
            steady_items = obj.generate_steady_items(start=period1, end=period10)
            for name, value in steady_items:
                print(f"{name}: {value}")
        ```
        ................................................................................
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

