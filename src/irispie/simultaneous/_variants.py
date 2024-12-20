"""
Defines the `Variant` class and helper functions for managing variant-specific 
attributes of a model, such as levels and changes.
"""


#[

from __future__ import annotations

import warnings as _wa
import copy as _co
import numpy as _np
import operator as _op

from ..conveniences import copies as _copies
from .. import quantities as _quantities
from ..sources import LOGGABLE_VARIABLE

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numbers import Real
    from typing import Self, Literal, Callable
    from collections.abc import Iterable

#]


class Variant:
    r"""
    ................................................................................
    ==Class: Variant==

    Represents a model variant, encapsulating attributes such as levels and changes 
    for quantities in the model. Provides methods for initialization, copying, updating, 
    and retrieving values for steady-state and dynamic calculations.

    Attributes:
        - `levels`: A dictionary mapping quantity IDs (QIDs) to their levels.
        - `changes`: A dictionary mapping QIDs to their changes.
        - `solution`: Solution-related data specific to the variant.
    ................................................................................
    """
    #[

    __slots__ = (
        "levels",
        "changes",
        "solution",
    )

    def __init__(self, **kwargs, ) -> None:
        r"""
        ................................................................................
        ==Method: __init__==

        Initializes a `Variant` instance with `levels`, `changes`, and `solution` set 
        to `None`. This constructor ensures that all slots are defined.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            variant = Variant()
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Class Method: from_source==

        Creates a `Variant` instance from a source of quantities. Initializes values 
        for levels and changes based on the model configuration.

        ### Input arguments ###
        ???+ input "quantities: Iterable[_quantities.Quantity]"
            The collection of quantities for the model.
        ???+ input "is_flat: bool"
            Indicates whether the model operates in flat mode.
        ???+ input "**kwargs"
            Additional parameters for initialization.

        ### Returns ###
        ???+ returns "Self"
            A newly initialized `Variant` instance.

        ### Example ###
        ```python
            variant = Variant.from_source(quantities, is_flat=True)
        ```
        ................................................................................
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
    def _all_qids(self, /, ) -> Iterable[int]:
        r"""
        ................................................................................
        ==Property: _all_qids==

        Retrieves all QIDs managed by the variant.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "Iterable[int]"
            An iterable of all QIDs.

        ### Example ###
        ```python
            qids = variant._all_qids
        ```
        ................................................................................
        """
        return self.levels.keys()

    def copy(self, /, ) -> Self:
        r"""
        ................................................................................
        ==Method: copy==

        Creates a deep copy of the `Variant` instance. All attributes (`levels`, 
        `changes`, and `solution`) are duplicated to ensure independence.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "Self"
            A deep copy of the `Variant` instance.

        ### Example ###
        ```python
            variant_copy = variant.copy()
        ```
        ................................................................................
        """
        new = type(self)()
        for i in ("levels", "changes", "solution", ):
            attr = getattr(self, i, )
            if attr is not None:
                setattr(new, i, attr.copy(), )
        return new

    def _initilize_values(self, qid_range, /, ) -> None:
        r"""
        ................................................................................
        ==Method: _initilize_values==

        Initializes the `levels` and `changes` dictionaries for the variant, setting 
        all entries to `None`.

        ### Input arguments ###
        ???+ input "qid_range: Iterable[int]"
            A range of QIDs to initialize.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            variant._initilize_values(range(10))
        ```
        ................................................................................
        """
        self.levels = { qid: None for qid in qid_range }
        self.changes = { qid: None for qid in qid_range }

    def update_values_from_dict(self, update: dict, ) -> None:
        r"""
        ................................................................................
        ==Method: update_values_from_dict==

        Updates the `levels` and `changes` dictionaries using data from a provided 
        dictionary. Supports tuple-based updates for compound attributes.

        ### Input arguments ###
        ???+ input "update: dict"
            A dictionary containing update values for levels and changes.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            variant.update_values_from_dict({1: (100, 0.01), 2: (200, 0.02)})
        ```
        ................................................................................
        """
        _update_from_dict(self.levels, update, _op.itemgetter(0), lambda x: x, )
        _update_from_dict(self.changes, update, _op.itemgetter(1), lambda x: ..., )

    def update_levels_from_array(self, levels: _np.ndarray, qids: Iterable[int], ) -> None:
        r"""
        ................................................................................
        ==Method: update_levels_from_array==

        Updates the `levels` dictionary using values from a numpy array. Each value 
        in the array is mapped to a corresponding QID.

        ### Input arguments ###
        ???+ input "levels: _np.ndarray"
            A numpy array containing level values to update.
        ???+ input "qids: Iterable[int]"
            A list of QIDs corresponding to the levels.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            variant.update_levels_from_array(level_values, qids=[1, 2, 3])
        ```
        ................................................................................
        """
        _update_from_array(self.levels, levels, qids, )

    def update_changes_from_array(self, changes: _np.ndarray, qids: Iterable[int], ) -> None:
        r"""
        ................................................................................
        ==Method: update_changes_from_array==

        Updates the `changes` dictionary using values from a numpy array. Each value 
        in the array is mapped to a corresponding QID.

        ### Input arguments ###
        ???+ input "changes: _np.ndarray"
            A numpy array containing change values to update.
        ???+ input "qids: Iterable[int]"
            A list of QIDs corresponding to the changes.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            variant.update_changes_from_array(change_values, qids=[1, 2, 3])
        ```
        ................................................................................
        """
        _update_from_array(self.changes, changes, qids, )

    def retrieve_values_as_array(
        self,
        attr: Literal["levels", "changes"],
        qids: Iterable[int] | None = None,
    ) -> _np.ndarray:
        r"""
        ................................................................................
        ==Method: retrieve_values_as_array==

        Retrieves values for the specified attribute (`levels` or `changes`) as a numpy 
        array. Missing values are replaced with `None`.

        ### Input arguments ###
        ???+ input "qids: Iterable[int]"
            A list of QIDs to retrieve values for.
        ???+ input "attribute: Literal['levels', 'changes']"
            The attribute to retrieve (`levels` or `changes`).

        ### Returns ###
        ???+ returns "_np.ndarray"
            A numpy array containing the retrieved values.

        ### Example ###
        ```python
            level_values = variant.retrieve_values_as_array(qids=[1, 2, 3], attribute="levels")
        ```
        ................................................................................
        """
        values = getattr(self, attr, )
        qids = qids if qids is not None else values.keys()
        return _np.array(tuple(values.get(qid, None, ) for qid in qids), dtype=float, )

    def rescale_values(
        self,
        attr: Literal["levels", "changes"],
        factor: Real,
        qids: Iterable[int] | None = None,
    ) -> None:
        r"""
        ................................................................................
        ==Method: rescale_values==

        Rescales level and change values in the variant by applying specified scale 
        factors to corresponding QIDs.

        ### Input arguments ###
        ???+ input "scale_factors: dict[int, Real]"
            A dictionary mapping QIDs to their respective scale factors.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            variant.rescale_values({1: 2.0, 2: 0.5})
        ```
        ................................................................................
        """
        values = getattr(self, attr, )
        qids = qids if qids is not None else values.keys()
        for qid in qids:
            values[qid] *= factor if values[qid] is not None else None

    def retrieve_maybelog_values_for_qids(
        self,
        qids: Iterable[int],
        qid_to_logly: dict[int, bool],
    ) -> tuple[_np.ndarray, ..., ]:
        r"""
        ................................................................................
        ==Method: retrievemaybelog_values_for_qids==

        Retrieves the specified attribute (`levels` or `changes`) for given QIDs, 
        applying logarithmic transformation if required.

        ### Input arguments ###
        ???+ input "qids: Iterable[int]"
            A list of QIDs to retrieve values for.
        ???+ input "attribute: Literal['levels', 'changes']"
            The attribute to retrieve (`levels` or `changes`).
        ???+ input "qid_to_logly: dict[int, bool]"
            A mapping of QIDs to their logarithmic status (`True` if logarithmic).

        ### Returns ###
        ???+ returns "_np.ndarray"
            A numpy array containing the retrieved values, with transformations applied.

        ### Example ###
        ```python
            values = variant.retrievemaybelog_values_for_qids(
                qids=[1, 2, 3], attribute="levels", qid_to_logly={1: True, 2: False}
            )
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Method: zero_changes==

        Sets the `changes` dictionary to zero for all QIDs. Logarithmic QIDs are set 
        to `1` instead, ensuring consistency with log scaling.

        ### Input arguments ###
        ???+ input "qid_to_logly: dict[int, bool]"
            A mapping of QIDs to their logarithmic status (`True` if logarithmic).

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            variant.zero_changes(qid_to_logly={1: True, 2: False})
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Method: create_steady_array==

        Constructs a steady-state array for the model variant. Includes level and 
        change values with adjustments for logarithmic variables.

        ### Input arguments ###
        ???+ input "qid_to_logly: dict[int, bool]"
            A mapping of QIDs to their logarithmic status (`True` if logarithmic).
        ???+ input "num_columns: int"
            The number of columns in the resulting array.
        ???+ input "shift_in_first_column: int"
            The shift value corresponding to the first column.

        ### Returns ###
        ???+ returns "_np.ndarray"
            A 2D numpy array representing the steady-state values.

        ### Example ###
        ```python
            steady_array = variant.create_steady_array(
                qid_to_logly={1: True, 2: False},
                num_columns=5,
                shift_in_first_column=-2,
            )
        ```
        ................................................................................
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
        /,
        num_columns: int = 1,
        shift_in_first_column: int = 0,
    ) -> _np.ndarray:
        r"""
        ................................................................................
        ==Method: create_zero_array==

        Constructs a zero-filled array with the specified dimensions.

        ### Input arguments ###
        ???+ input "num_rows: int"
            The number of rows in the array.
        ???+ input "num_columns: int"
            The number of columns in the array.

        ### Returns ###
        ???+ returns "_np.ndarray"
            A 2D numpy array filled with zeros.

        ### Example ###
        ```python
            zero_array = variant.create_zero_array(num_rows=5, num_columns=10)
        ```
        ................................................................................
        """
        levels = self.retrieve_values_as_array("levels", ).reshape(-1, 1)
        inx_set_to_0 = [ qid for qid, logly in qid_to_logly.items() if logly is False ]
        inx_set_to_1 = [ qid for qid, logly in qid_to_logly.items() if logly is True ]
        levels[inx_set_to_0] = 0
        levels[inx_set_to_1] = 1
        return _np.tile(levels, (1, num_columns, ))

    def _serialize_to_portable(self, qid_to_name, /, ) -> dict[str, Any]:
        r"""
        ................................................................................
        ==Method: serialize_to_portable==

        Serializes the `Variant` instance into a portable dictionary format. This 
        is useful for exporting or saving variant data.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "dict"
            A dictionary containing serialized variant data.

        ### Example ###
        ```python
            serialized_data = variant.serialize_to_portable()
        ```
        ................................................................................
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
    r"""
    ................................................................................
    ==Function: _update_from_array==

    Updates a dictionary with values from a numpy array. Each value in the array 
    is mapped to the corresponding QID in the dictionary.

    ### Input arguments ###
    ???+ input "destination: dict[int, Any]"
        The dictionary to update.
    ???+ input "values: _np.ndarray"
        A numpy array containing the new values.
    ???+ input "qids: Iterable[int]"
        A list of QIDs corresponding to the values.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _update_from_array(levels, level_values, qids=[1, 2, 3])
    ```
    ................................................................................
    """
    #[
    if updated_values is None:
        return
    if hasattr(updated_values, "flatten"):
        updated_values = updated_values.flatten().tolist()
    for qid, value in zip(qids, updated_values):
        what_to_update[qid] = value if not _is_nan(value) else None
    #]


def _is_nan(x: Real | None, /, ) -> bool:
    return x != x


def _update_from_dict(
    what_to_update: dict[int, Real | None],
    update: dict[int, Real | tuple[Real]],
    when_tuple: Callable,
    when_not_tuple: Callable,
    /,
) -> _np.ndarray:
    r"""
    ................................................................................
    ==Function: _update_from_dict==

    Updates a dictionary with values from another dictionary. Each value is processed 
    using a getter and factory function.

    ### Input arguments ###
    ???+ input "destination: dict[int, Any]"
        The dictionary to update.
    ???+ input "source: dict"
        The source dictionary containing update data.
    ???+ input "getter: Callable[[Any], Any]"
        A function to extract the relevant part of the source data.
    ???+ input "factory: Callable[[Any], Any]"
        A function to transform the extracted data.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _update_from_dict(destination, source, operator.itemgetter(0), lambda x: x)
    ```
    ................................................................................
    """
    #[
    for qid, value in update.items():
        new_value = when_tuple(value) if isinstance(value, tuple) else when_not_tuple(value)
        new_value = new_value if new_value is not ... else what_to_update[qid]
        what_to_update[qid] = new_value if not _is_nan(new_value) else None
    #]

