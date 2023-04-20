"""
"""


#[
from __future__ import annotations
# from IPython import embed

import warnings
import numpy as np_
import operator as op_
from numbers import Number
from typing import (Self, NoReturn, TypeAlias, Literal, Callable, )

from ..quantities import get_max_qid
#]


class Variant:
    """
    Container for parameter variant specific attributes of a model
    """
    __slots__ = (
        "levels", "changes", "solution",
    )
    _missing = np_.nan
    #[
    def __init__(self, quantities:Quantities, /, ) -> NoReturn:
        self._initilize_values(quantities)
        self.solution = None

    def _initilize_values(self, quantities:Quantities, /, ) -> NoReturn:
        max_qid = get_max_qid(quantities, )
        size_array = max_qid + 1
        self.levels = np_.full((size_array,), self._missing, dtype=float, )
        self.changes = np_.full((size_array,), self._missing, dtype=float, )

    def update_values_from_dict(self, update: dict, /, ) -> NoReturn:
        self.levels = update_something_from_dict(self.levels, update, op_.itemgetter(0), lambda x: x, )
        self.changes = update_something_from_dict(self.changes, update, op_.itemgetter(1), lambda x: ..., )

    def update_levels_from_array(self, levels: np_.ndarray, qids: Iterable[int], /, ) -> NoReturn:
        self.levels = update_from_array(self.levels, levels, qids, )

    def update_changes_from_array(self, changes: np_.ndarray, qids: Iterable[int], /, ) -> NoReturn:
        self.changes = update_from_array(self.changes, changes, qids, )

    def retrieve_values(
        self,
        attr: Literal["levels"] | Literal["changes"],
        qids: Iterable[int]|None = None,
        /,
    ) -> np_.ndarray:
        values = np_.copy(getattr(self, attr)).reshape(-1, 1)
        return values[qids] if qids is not None else values

    def create_steady_array(
        self,
        qid_to_logly: dict[int, bool],
        /,
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> np_.ndarray:
        """
        """
        levels = np_.copy(self.levels).reshape(-1, 1)
        changes = np_.copy(self.changes).reshape(-1, 1)
        #
        if num_columns==1 and shift_in_first_column==0:
            return levels
        #
        logly = np_.array([
            qid_to_logly[i] is True
            for i in range(levels.shape[0])
        ])
        #
        shift_vec = np_.array(range(shift_in_first_column, shift_in_first_column+num_columns))
        #
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        levels[logly] = np_.log(levels[logly])
        changes[logly] = np_.log(changes[logly])
        warnings.filterwarnings(action="default", category=RuntimeWarning)
        #
        levels[np_.isnan(levels) | np_.isinf(levels)] = np_.nan
        changes[np_.isnan(changes) | np_.isinf(changes)] = 0
        #
        steady_array = levels + changes * shift_vec
        #
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        steady_array[logly, :] = np_.exp(steady_array[logly, :])
        warnings.filterwarnings(action="default", category=RuntimeWarning)
        #
        return steady_array

    def create_zero_array(
        self,
        qid_to_logly: dict[int, bool],
        /,
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> np_.ndarray:
        """
        """
        levels = np_.copy(self.levels).reshape(-1, 1)
        inx_set_to_0 = [ qid for qid, logly in qid_to_logly.items() if logly is False ]
        inx_set_to_1 = [ qid for qid, logly in qid_to_logly.items() if logly is True ]
        levels[inx_set_to_0] = 0
        levels[inx_set_to_1] = 1
        return np_.tile(levels, (1, num_columns))
    #]


def update_from_array(
    values: np_.ndarray,
    updated_values: np_.ndarray,
    qids: list[int],
    /,
) -> np_.ndarray:
    """
    Update variant levels or changes from an array and a list of qids
    """
    if updated_values is not None:
        values[qids] = updated_values.flat
    return values


def update_something_from_dict(
    something: np_.ndarray,
    update: dict[int, Number|tuple],
    when_tuple: Callable,
    when_not_tuple: Callable,
    /,
) -> np_.ndarray:
    """
    Update variant levels or changes from a dictionary
    """
    for qid, value in update.items():
        new_value = when_tuple(value) if isinstance(value, tuple) else when_not_tuple(value)
        something[qid] = new_value if new_value is not ... else something[qid]
    return something

