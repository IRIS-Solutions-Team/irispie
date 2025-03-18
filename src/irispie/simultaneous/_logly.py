"""
"""


#[
from __future__ import annotations

from typing import (Callable, Iterable, )
import numpy as _np

from .. import quantities as _quantities
#]


class Inlay:
    """
    """
    #[

    def logarithmize(
        self,
        *arrays: tuple[_np.ndarray],
    ) -> None:
        """
        """
        _apply(self, _np.log, *arrays, )

    def delogarithmize(
        self,
        *arrays: tuple[_np.ndarray],
    ) -> None:
        """
        """
        _apply(self, _np.exp, *arrays, )

    #]


def _apply(
    self,
    func: Callable,
    *arrays: tuple[_np.ndarray],
) -> None:
    """
    """
    #[
    logly_indexes = self.get_logly_indexes()
    if not logly_indexes:
        return
    for array in arrays:
        array[logly_indexes, ...] = func(array[logly_indexes, ...])
    #]

