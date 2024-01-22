"""
"""


#[
import warnings
from numbers import Number
from typing import (Self, Literal, Callable, )
import copy as _co
import numpy as _np
import operator as _op
from collections.abc import (Iterable, )

from ..conveniences import copies as _copies
from .. import quantities as _quantities
#]


class Variant:
    """
    Container for parameter variant specific attributes of a model
    """
    #[

    __slots__ = ("levels", "changes", "solution", "_max_qid", )

    _missing = _np.nan

    def __init__(
        self,
        quantities: Iterable[_quantities.Quantity],
        is_flat: bool,
        /,
        **kwargs,
    ) -> None:
        """
        """
        max_qid = _quantities.get_max_qid(quantities, )
        self._max_qid = max_qid
        self.solution = None
        self._initilize_values()
        if is_flat:
            qid_to_logly = _quantities.create_qid_to_logly(quantities, )
            self.zero_changes(qid_to_logly, )

    def copy(self, /, ) -> Self:
        """
        """
        return _co.deepcopy(self, )

    def _initilize_values(self, /, ) -> None:
        """
        """
        self._reset_levels()
        self._reset_changes()

    def _reset_levels(self, /, ) -> None:
        """
        """
        self.levels = _np.full((self._max_qid+1, ), self._missing, dtype=float, )

    def _reset_changes(self, /, ) -> None:
        """
        """
        self.changes = _np.full((self._max_qid+1, ), self._missing, dtype=float, )

    def update_values_from_dict(self, update: dict, /, ) -> None:
        self.levels = _update_from_dict(self.levels, update, _op.itemgetter(0), lambda x: x, )
        self.changes = _update_from_dict(self.changes, update, _op.itemgetter(1), lambda x: ..., )

    def update_levels_from_array(self, levels: _np.ndarray, qids: Iterable[int], /, ) -> None:
        self.levels = _update_from_array(self.levels, levels, qids, )

    def update_changes_from_array(self, changes: _np.ndarray, qids: Iterable[int], /, ) -> None:
        self.changes = _update_from_array(self.changes, changes, qids, )

    def retrieve_values(
        self,
        attr: Literal["levels", "changes"],
        qids: Iterable[int] | None = None,
        /,
    ) -> _np.ndarray:
        # values = _np.copy(getattr(self, attr)).reshape(-1, 1)
        values = _np.copy(getattr(self, attr, ), )
        qids = list(qids, )
        return values[qids] if qids is not None else values

    def retrieve_maybelog_values_for_qids(
        self,
        qids: Iterable[int],
        qid_to_logly: dict[int, bool],
    ) -> tuple[_np.ndarray, ..., ]:
        """
        """
        qids = list(qids)
        where_logly = list(_quantities.generate_where_logly(qids, qid_to_logly, ))
        #
        # Extract initial guesses for levels and changes
        maybelog_levels = self.levels[qids].flatten()
        maybelog_changes = self.changes[qids].flatten()
        #
        # Logarithmize
        maybelog_levels[where_logly] = _np.log(maybelog_levels[where_logly])
        maybelog_changes[where_logly] = _np.log(maybelog_changes[where_logly])
        #
        return maybelog_levels, maybelog_changes

    def zero_changes(
        self,
        qid_to_logly: dict[int, bool | None],
        /,
    ) -> None:
        """
        Reset all quantities to flat
        """
        self._reset_changes()
        for qid, logly in qid_to_logly.items():
            if logly is None:
                continue
            self.changes[qid] = 1 if logly else 0

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
        qids = list(range(self._max_qid+1))
        where_logly = list(_quantities.generate_where_logly(qids, qid_to_logly))
        #
        shift_vec = _np.array(range(shift_in_first_column, shift_in_first_column+num_columns))
        #
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        levels[where_logly] = _np.log(levels[where_logly])
        changes[where_logly] = _np.log(changes[where_logly])
        warnings.filterwarnings(action="default", category=RuntimeWarning)
        #
        levels[_np.isnan(levels) | _np.isinf(levels)] = _np.nan
        changes[_np.isnan(changes) | _np.isinf(changes)] = 0
        #
        steady_array = levels + changes * shift_vec
        #
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        steady_array[where_logly, :] = _np.exp(steady_array[where_logly, :])
        warnings.filterwarnings(action="default", category=RuntimeWarning)
        #
        return steady_array

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

