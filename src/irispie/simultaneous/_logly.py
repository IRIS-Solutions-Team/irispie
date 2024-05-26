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

    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        """
        Create a dictionary mapping from quantity id to quantity log-status
        """
        return _quantities.create_qid_to_logly(self._invariant.quantities)

    def get_logly_indexes(self, /) -> tuple[int, ...]:
        """
        """
        return tuple(_quantities.generate_logly_indexes(self._invariant.quantities))

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

