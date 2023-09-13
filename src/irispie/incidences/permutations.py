"""
"""


#[
from __future__ import annotations

import numpy as _np

from .. import wrongdoings as _wrongdoings
#]


def sequentialize(
    incident_matrix: _np.ndarray,
    /,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    """
    #[
    MAX_ITERATIONS = 100
    im = _np.copy(incident_matrix)
    row_index = _np.arange(im.shape[0])
    column_index = _np.arange(im.shape[1])
    #
    count = 0
    while not is_sequential(im):
        #
        # Check if too many iterations
        if count >= MAX_ITERATIONS:
            _wrongdoings.throw(
                "error",
                f"Could not find sequential ordering after {MAX_ITERATIONS} iterations.",
            )
        #
        count += 1
        #
        # Reorder columns given rows
        column_reordering = _get_column_reordering(im)
        column_index = column_index[column_reordering]
        im = im[:, column_reordering]
        #
        # Reorder rows given columns
        row_reordering = _get_row_reordering(im)
        im = im[row_reordering, :]
        row_index = row_index[row_reordering]

    info = {"iterations": count}
    return (row_index, column_index), im, info
    #]


def _get_column_reordering(
    im: _np.ndarray,
    /,
) -> _np.ndarray:
    """
    """
    #[
    num_incidences_in_columns = _np.sum(im, axis=0, )
    return _np.flip(_np.argsort(num_incidences_in_columns))
    #]


def _get_row_reordering(
    im: _np.ndarray,
    /,
) -> _np.ndarray:
    """
    """
    #[
    num_incidences_in_rows = _np.sum(im, axis=1, )
    return _np.argsort(num_incidences_in_rows)
    #]


def is_sequential(
    im: _np.ndarray,
    /,
    order: int = 1,
) -> bool:
    """
    """
    return _np.all(~_np.triu(im, order))

