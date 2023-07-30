"""
"""


#[
import warnings
from numbers import Number
from typing import (Self, Literal, Callable, )
import numpy as _np
import operator as _op
from collections.abc import (Iterable, )

from .. import (quantities as _qu, )
#]


class Variant:
    """
    Container for parameter variant specific attributes of a model
    """
    #[
    __slots__ = (
        "levels", "changes", "solution",
    )

    _missing = _np.nan

    def __init__(
        self,
        quantities: _qu.Quantities, 
        /,
        **kwargs,
    ) -> None:
        self._initilize_values(quantities, )
        self.solution = None

    def _initilize_values(
        self,
        quantities: _qu.Quantities, 
        /,
    ) -> None:
        max_qid = _qu.get_max_qid(quantities, )
        size_array = max_qid + 1
        init_value = self._missing
        self.levels = _np.full((size_array,), init_value, dtype=float, )
        self.changes = _np.full((size_array,), init_value, dtype=float, )

    def update_values_from_dict(self, update: dict, /, ) -> None:
        self.levels = _update_from_dict(self.levels, update, _op.itemgetter(0), lambda x: x, )
        self.changes = _update_from_dict(self.changes, update, _op.itemgetter(1), lambda x: ..., )

    def update_levels_from_array(self, levels: _np.ndarray, qids: Iterable[int], /, ) -> None:
        self.levels = _update_from_array(self.levels, levels, qids, )

    def update_changes_from_array(self, changes: _np.ndarray, qids: Iterable[int], /, ) -> None:
        self.changes = _update_from_array(self.changes, changes, qids, )

    def retrieve_values(
        self,
        attr: Literal["levels"] | Literal["changes"],
        qids: Iterable[int]|None = None,
        /,
    ) -> _np.ndarray:
        values = _np.copy(getattr(self, attr)).reshape(-1, 1)
        return values[qids] if qids is not None else values

    def retrieve_maybelog_values_for_qids(
        self,
        qids: Iterable[int],
        qid_to_logly: dict[int, bool],
    ) -> tuple[_np.ndarray, _np.ndarray, ]:
        """
        """
        qids = list(qids)
        index_logly = list(_qu.generate_index_logly(qids, qid_to_logly, ))
        #
        # Extract initial guesses for levels and changes
        maybelog_levels = self.levels[qids].flatten()
        maybelog_changes = self.changes[qids].flatten()
        #
        # Logarithmize
        maybelog_levels[index_logly] = _np.log(maybelog_levels[index_logly])
        maybelog_changes[index_logly] = _np.log(maybelog_changes[index_logly])
        #
        return maybelog_levels, maybelog_changes

    def reset_changes(
        self,
        qid_to_logly: dict[int, bool],
        /,
    ) -> None:
        index_logly = self._generate_index_logly(qid_to_logly, )
        self.changes[:] = 0
        self.changes[index_logly] = 1

    def create_steady_array(
        self,
        qid_to_logly: dict[int, bool],
        /,
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> _np.ndarray:
        """
        """
        levels = _np.copy(self.levels).reshape(-1, 1)
        changes = _np.copy(self.changes).reshape(-1, 1)
        #
        if num_columns==1 and shift_in_first_column==0:
            return levels
        #
        index_logly = list(self._generate_index_logly(qid_to_logly))
        #
        shift_vec = _np.array(range(shift_in_first_column, shift_in_first_column+num_columns))
        #
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        levels[index_logly] = _np.log(levels[index_logly])
        changes[index_logly] = _np.log(changes[index_logly])
        warnings.filterwarnings(action="default", category=RuntimeWarning)
        #
        levels[_np.isnan(levels) | _np.isinf(levels)] = _np.nan
        changes[_np.isnan(changes) | _np.isinf(changes)] = 0
        #
        steady_array = levels + changes * shift_vec
        #
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        steady_array[index_logly, :] = _np.exp(steady_array[index_logly, :])
        warnings.filterwarnings(action="default", category=RuntimeWarning)
        #
        return steady_array

    def _generate_index_logly(
        self,
        qid_to_logly: dict[int, bool],
        /,
    ) -> list[int]:
        """
        """
        qids = range(self.levels.shape[0])
        return _qu.generate_index_logly(qids, qid_to_logly, )

    def create_zero_array(
        self,
        qid_to_logly: dict[int, bool],
        /,
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> _np.ndarray:
        """
        """
        levels = _np.copy(self.levels).reshape(-1, 1)
        inx_set_to_0 = [ qid for qid, logly in qid_to_logly.items() if logly is False ]
        inx_set_to_1 = [ qid for qid, logly in qid_to_logly.items() if logly is True ]
        levels[inx_set_to_0] = 0
        levels[inx_set_to_1] = 1
        return _np.tile(levels, (1, num_columns))
    #]


def _update_from_array(
    values: _np.ndarray,
    updated_values: _np.ndarray,
    qids: Iterable[int],
    /,
) -> _np.ndarray:
    """
    Update levels or changes from an array and a list of qids
    """
    #[
    if updated_values is not None:
        values[list(qids)] = updated_values.flat
    return values
    #]


def _update_from_dict(
    what_to_update: _np.ndarray,
    update: dict[int, Number|tuple],
    when_tuple: Callable,
    when_not_tuple: Callable,
    /,
) -> _np.ndarray:
    """
    Update levels or changes from a dictionary
    """
    #[
    for qid, value in update.items():
        new_value = when_tuple(value) if isinstance(value, tuple) else when_not_tuple(value)
        what_to_update[qid] = new_value if new_value is not ... else what_to_update[qid]
    return what_to_update
    #]


