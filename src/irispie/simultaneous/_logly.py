"""
Handles operations related to logarithm transformations for numerical arrays.
"""


#[
from __future__ import annotations

from typing import (Callable, Iterable, )
import numpy as _np

from .. import quantities as _quantities
#]


class Inlay:
    r"""
    ................................................................................
    ==Class: Inlay==

    Provides functionality to manage logarithmic transformations of numerical arrays 
    and log-status mappings of quantities. This class is essential for systems that 
    deal with both logarithmic and linear scales, offering methods for seamless 
    transitions between the two.

    Attributes:
        - `_invariant.quantities`: The collection of quantities managed by the instance.
    ................................................................................
    """
    #[

    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        r"""
        ................................................................................
        ==Method: create_qid_to_logly==

        Creates a mapping from quantity IDs to their log-status. The log-status 
        indicates whether a quantity is stored in logarithmic scale.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "dict[int, bool]"
            A dictionary mapping each quantity ID to a boolean value representing 
            its log-status (`True` for logarithmic scale, `False` otherwise).

        ### Example ###
        ```python
            logly_map = obj.create_qid_to_logly()
        ```
        ................................................................................
        """
        return _quantities.create_qid_to_logly(self._invariant.quantities)

    def get_logly_indexes(self, /) -> tuple[int, ...]:
        r"""
        ................................................................................
        ==Method: get_logly_indexes==

        Generates a tuple of indexes corresponding to quantities stored in logarithmic 
        scale. This is useful for applying transformations to specific elements in 
        numerical arrays.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "tuple[int, ...]"
            A tuple of indexes representing quantities in logarithmic scale.

        ### Example ###
        ```python
            logly_indexes = obj.get_logly_indexes()
        ```
        ................................................................................
        """
        return tuple(_quantities.generate_logly_indexes(self._invariant.quantities))

    def logarithmize(
        self,
        *arrays: tuple[_np.ndarray],
    ) -> None:
        r"""
        ................................................................................
        ==Method: logarithmize==

        Applies the natural logarithm transformation to the specified arrays for 
        elements corresponding to quantities in logarithmic scale.

        ### Input arguments ###
        ???+ input "*arrays: tuple[_np.ndarray]"
            A tuple of numerical arrays to transform. Only elements corresponding to 
            logarithmic quantities will be transformed.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            obj.logarithmize(array1, array2)
        ```
        ................................................................................
        """
        _apply(self, _np.log, *arrays, )

    def delogarithmize(
        self,
        *arrays: tuple[_np.ndarray],
    ) -> None:
        r"""
        ................................................................................
        ==Method: delogarithmize==

        Applies the exponential transformation to the specified arrays for elements 
        corresponding to quantities in logarithmic scale. This reverses the 
        logarithmize operation.

        ### Input arguments ###
        ???+ input "*arrays: tuple[_np.ndarray]"
            A tuple of numerical arrays to transform. Only elements corresponding to 
            logarithmic quantities will be transformed.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            obj.delogarithmize(array1, array2)
        ```
        ................................................................................
        """
        _apply(self, _np.exp, *arrays, )

    #]


def _apply(
    self,
    func: Callable,
    *arrays: tuple[_np.ndarray],
) -> None:
    r"""
    ................................................................................
    ==Function: _apply==

    Internal function to apply a specified transformation function (e.g., logarithm 
    or exponential) to numerical arrays for elements corresponding to quantities in 
    logarithmic scale.

    ### Input arguments ###
    ???+ input "self"
        The instance of the class calling this function.
    ???+ input "func: Callable"
        The transformation function to apply (e.g., `np.log` or `np.exp`).
    ???+ input "*arrays: tuple[_np.ndarray]"
        A tuple of numerical arrays to transform.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _apply(self, np.log, array1, array2)
    ```
    ................................................................................
    """
    #[
    logly_indexes = self.get_logly_indexes()
    if not logly_indexes:
        return
    for array in arrays:
        array[logly_indexes, ...] = func(array[logly_indexes, ...])
    #]

