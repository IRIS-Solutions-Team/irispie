r"""
"""


#[

from __future__ import annotations

import warnings as _wa
import copy as _co
import numpy as _np
import operator as _op

from ..conveniences import copies as _copies
from .. import quantities as _quantities

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numbers import Real
    from typing import Self, Literal, Callable
    from collections.abc import Iterable

#]


class Variant:
    """
    Container for parameter variant specific attributes of a model
    """
    #[

    __slots__ = (
        "levels",
        "changes",
        "solution",
    )

    def __init__(self, **kwargs, ) -> None:
        """
        """
        for n in self.__slots__:
            setattr(self, n, None, )

    @classmethod
    def from_source(
        klass,
        quantities: Iterable[_quantities.Quantity],
        is_flat: bool,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass()
        max_qid = _quantities.get_max_qid(quantities, )
        qid_range = range(max_qid+1, )
        self._initilize_values(qid_range, )
        if is_flat:
            qid_to_logly = _quantities.create_qid_to_logly(quantities, )
            self.zero_changes(qid_to_logly, )
        return self

    @property
    def _all_qids(self, ) -> Iterable[int]:
        """
        """
        return self.levels.keys()

    def copy(self, ) -> Self:
        """
        """
        new = type(self)()
        for i in ("levels", "changes", "solution", ):
            attr = getattr(self, i, )
            if attr is not None:
                setattr(new, i, attr.copy(), )
        return new

    def _initilize_values(self, qid_range, ) -> None:
        """
        """
        self.levels = { qid: None for qid in qid_range }
        self.changes = { qid: None for qid in qid_range }

    def update_values_from_dict(self, update: dict, ) -> None:
        _update_from_dict(self.levels, update, _op.itemgetter(0), lambda x: x, )
        _update_from_dict(self.changes, update, _op.itemgetter(1), lambda x: ..., )

    def update_levels_from_array(self, levels: _np.ndarray, qids: Iterable[int], ) -> None:
        _update_from_array(self.levels, levels, qids, )

    def update_changes_from_array(self, changes: _np.ndarray, qids: Iterable[int], ) -> None:
        _update_from_array(self.changes, changes, qids, )

    def retrieve_values_as_array(
        self,
        attr: Literal["levels", "changes"],
        qids: Iterable[int] | None = None,
    ) -> _np.ndarray:
        values = getattr(self, attr, )
        qids = qids if qids is not None else values.keys()
        return _np.array(tuple(values.get(qid, None, ) for qid in qids), dtype=float, )

    def rescale_values(
        self,
        attr: Literal["levels", "changes"],
        factor: Real,
        qids: Iterable[int] | None = None,
    ) -> None:
        """
        Rescale values by a common factor
        """
        values = getattr(self, attr, )
        qids = qids if qids is not None else values.keys()
        for qid in qids:
            values[qid] *= factor if values[qid] is not None else None

    def retrieve_levels_as_dict(
        self,
        qids: Iterable[int],
    ) -> dict[int, Real | None]:
        r"""
        """
        return { i: self.levels[i] for i in qids }

    def retrieve_changes_as_dict(
        self,
        qids: Iterable[int],
    ) -> dict[int, Real | None]:
        r"""
        """
        return { i: self.changes[i] for i in qids }

    def retrieve_maybelog_values_for_qids(
        self,
        qids: Iterable[int],
        qid_to_logly: dict[int, bool],
    ) -> tuple[_np.ndarray, _np.ndarray, ]:
        """
        """
        #
        # Extract levels and changes as arrays
        qids = tuple(qids)
        maybelog_levels = self.retrieve_values_as_array("levels", qids, )
        maybelog_changes = self.retrieve_values_as_array("changes", qids, )
        #
        # Logarithmize
        where_logly = list(_quantities.generate_where_logly(qids, qid_to_logly, ))
        maybelog_levels[where_logly] = _np.log(maybelog_levels[where_logly])
        maybelog_changes[where_logly] = _np.log(maybelog_changes[where_logly])
        #
        return maybelog_levels, maybelog_changes,

    def zero_changes(
        self,
        qid_to_logly: dict[int, bool | None],
    ) -> None:
        """
        Reset all quantities to flat
        """
        for qid in self.changes.keys():
            logly = qid_to_logly.get(qid, None, )
            self.changes[qid] = float(logly) if logly is not None else None

    def create_steady_array(
        self,
        qid_to_logly: dict[int, bool | None],
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> _np.ndarray:
        """
        """
        levels = self.retrieve_values_as_array("levels", ).reshape(-1, 1)
        if num_columns==1 and shift_in_first_column==0:
            return levels
        changes = self.retrieve_values_as_array("changes", ).reshape(-1, 1)
        #
        where_logly = list(_quantities.generate_where_logly(self._all_qids, qid_to_logly))
        #
        shift_vec = _np.array(range(shift_in_first_column, shift_in_first_column+num_columns))
        #
        _wa.filterwarnings(action="ignore", category=RuntimeWarning)
        levels[where_logly] = _np.log(levels[where_logly])
        changes[where_logly] = _np.log(changes[where_logly])
        _wa.filterwarnings(action="default", category=RuntimeWarning)
        #
        levels[_np.isnan(levels) | _np.isinf(levels)] = _np.nan
        changes[_np.isnan(changes) | _np.isinf(changes)] = 0
        #
        steady_array = levels + changes * shift_vec
        #
        _wa.filterwarnings(action="ignore", category=RuntimeWarning)
        steady_array[where_logly, :] = _np.exp(steady_array[where_logly, :])
        _wa.filterwarnings(action="default", category=RuntimeWarning)
        #
        return steady_array

    def create_zero_array(
        self,
        qid_to_logly: dict[int, bool],
        #
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> _np.ndarray:
        """
        """
        levels = self.retrieve_values_as_array("levels", ).reshape(-1, 1)
        inx_set_to_0 = [ qid for qid, logly in qid_to_logly.items() if logly is False ]
        inx_set_to_1 = [ qid for qid, logly in qid_to_logly.items() if logly is True ]
        levels[inx_set_to_0] = 0
        levels[inx_set_to_1] = 1
        return _np.tile(levels, (1, num_columns, ))

    def to_portable(self, qid_to_name, ) -> dict[str, Any]:
        """
        """
        return {
            qid_to_name[qid]: (level, self.changes[qid], )
            for qid, level in self.levels.items()
        }


def _update_from_array(
    what_to_update: dict[int, Real | None],
    updated_values: _np.ndarray | Iterable[Real] | None,
    qids: Iterable[int],
) -> None:
    """
    Update levels or changes from an array and a list of qids
    """
    #[
    if updated_values is None:
        return
    if hasattr(updated_values, "flatten"):
        updated_values = updated_values.flatten().tolist()
    for qid, value in zip(qids, updated_values):
        what_to_update[qid] = value if not _is_nan(value) else None
    #]


def _is_nan(x: Real | None, ) -> bool:
    return x != x


def _update_from_dict(
    what_to_update: dict[int, Real | None],
    update: dict[int, Real | tuple[Real]],
    when_tuple: Callable,
    when_not_tuple: Callable,
) -> _np.ndarray:
    """
    Update levels or changes from a dictionary
    """
    #[
    for qid, value in update.items():
        new_value = when_tuple(value) if isinstance(value, tuple) else when_not_tuple(value)
        new_value = new_value if new_value is not ... else what_to_update[qid]
        what_to_update[qid] = new_value if not _is_nan(new_value) else None
    #]

