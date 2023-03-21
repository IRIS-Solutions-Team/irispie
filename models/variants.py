"""
"""

#[
from __future__ import annotations
from IPython import embed
import warnings
import numpy
from numbers import Number
from typing import Self, NoReturn, TypeAlias, Literal

from ..quantities import get_max_qid
#]


class Variant:
    """
    Container for parameter variant specific attributes of a model
    """
    _missing = numpy.nan
    __slots__ = ["levels", "changes", "solution"]
    #[
    def __init__(self, quantities:Quantities, /, ) -> NoReturn:
        self._initilize_values(quantities)
        self.solution = None

    def _initilize_values(self, quantities:Quantities, /, ) -> NoReturn:
        max_qid = get_max_qid(quantities, )
        size_array = max_qid + 1
        self.levels = numpy.full((size_array,), self._missing, dtype=float, )
        self.changes = numpy.full((size_array,), self._missing, dtype=float, )

    def update_values_from_dict(self, update: dict, /, ) -> NoReturn:
        self.levels = update_levels_from_dict(self.levels, update, )
        self.changes = update_changes_from_dict(self.changes, update, )

    def update_levels_from_array(self, levels: numpy.ndarray, qids: Iterable[int], /, ) -> NoReturn:
        self.levels = update_from_array(self.levels, levels, qids, )

    def update_changes_from_array(self, changes: numpy.ndarray, qids: Iterable[int], /, ) -> NoReturn:
        self.changes = update_from_array(self.changes, changes, qids, )

    def retrieve_values(
        self,
        attr: Literal["levels"] | Literal["changes"],
        qids: Iterable[int]|None = None,
        /,
    ) -> numpy.ndarray:
        values = numpy.copy(getattr(self, attr)).reshape(-1, 1)
        return values[qids] if qids is not None else values

    def create_steady_array(
        self,
        qid_to_logly: dict[int, bool],
        /,
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> numpy.ndarray:
        """
        """
        levels = numpy.copy(self.levels).reshape(-1, 1)
        changes = numpy.copy(self.changes).reshape(-1, 1)
        #
        if num_columns==1 and shift_in_first_column==0:
            return levels
        #
        logly = numpy.array([
            qid_to_logly[i] is True
            for i in range(levels.shape[0])
        ])
        #
        shift_vec = numpy.array(range(shift_in_first_column, num_columns))
        #
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        levels[logly] = numpy.log(levels[logly])
        changes[logly] = numpy.log(changes[logly])
        warnings.filterwarnings(action="default", category=RuntimeWarning)
        #
        levels[numpy.isnan(levels) | numpy.isinf(levels)] = numpy.nan
        changes[numpy.isnan(changes) | numpy.isinf(changes)] = 0
        #
        steady_array = levels + changes * shift_vec
        #
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        steady_array[logly, :] = numpy.exp(steady_array[logly, :])
        warnings.filterwarnings(action="default", category=RuntimeWarning)
        #
        return steady_array


    def create_zero_array(
        self,
        qid_to_logly: dict[int, bool],
        /,
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> numpy.ndarray:
        """
        """
        levels = numpy.copy(self.levels).reshape(-1, 1)
        inx_set_to_0 = [ qid for qid, logly in qid_to_logly.items() if logly is False ]
        inx_set_to_1 = [ qid for qid, logly in qid_to_logly.items() if logly is True ]
        levels[inx_set_to_0] = 0
        levels[inx_set_to_1] = 1
        return numpy.tile(levels, (1, num_columns))
    #]


def update_from_array(
    values: numpy.ndarray,
    updated_values: numpy.ndarray,
    qids: list[int],
    /,
) -> numpy.ndarray:
    """
    Update variant levels or changes from an array and a list of qids
    """
    if updated_values is not None:
        values[qids] = updated_values.flat
    return values


def update_levels_from_dict(
    levels: numpy.ndarray,
    update: dict[int, Number|tuple],
    /,
) -> numpy.ndarray:
    """
    Update variant levels from a dictionary
    """
    for qid, new_value in update.items():
        new_value = new_value if isinstance(new_value, Number) else new_value[0]
        levels[qid] = new_value if new_value is not ... else levels[qid]
    return levels


def update_changes_from_dict(
    changes: numpy.ndarray,
    update: dict[int, Number|tuple],
    /,
) -> numpy.ndarray:
    """
    Update variant changes from a dictionary
    """
    for qid, new_value in update.items():
        new_value = ... if isinstance(new_value, Number) else new_value[1]
        changes[qid] = new_value if new_value is not ... else changes[qid]
    return changes


