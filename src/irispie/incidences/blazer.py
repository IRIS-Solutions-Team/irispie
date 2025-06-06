"""
"""


#[
from __future__ import annotations

from collections.abc import (Generator, Iterable, )
from typing import (Any, )
import numpy as _np

from .. import wrongdoings as _wrongdoings
from ..equations import (Equation, )
from ..quantities import (Quantity, )

# from ._dulmage_mendelsohn import dulmage_mendelsohn_reordering
#]


class HumanBlock:
    """
    """
    #[

    __slots__ = (
        "equations",
        "quantities",
    )

    def __init__(
        self,
        block: Block,
        equations: Iterable[Equation],
        quantities: Iterable[Quantity],
    ) -> None:
        """
        """
        self
        self.equations = tuple(equations[eid].human for eid in sorted(block.eids))
        self.quantities = tuple(quantities[qid].human for qid in sorted(block.qids))

    @property
    def num_equations(self, ) -> int:
        r"""
        """
        return len(self.equations)

    @property
    def num_quantities(self, ) -> int:
        r"""
        """
        return len(self.quantities)

    def __repr__(self) -> str:
        """
        """
        return f"HumanBlock[{self.num_equations}×{self.num_quantities}]"

    #]


class Block:
    """
    """
    #[

    __slots__ = (
        "eids",
        "qids",
    )

    def __init__(
        self, eids: Iterable[int] | None = None,
        qids: Iterable[int] | None = None,
    ) -> None:
        """
        """
        self.eids = tuple(sorted(eids))
        self.qids = tuple(sorted(qids))

    @property
    def num_equations(self, ) -> int:
        return len(self.eids)

    @property
    def num_quantities(self, ) -> int:
        return len(self.qids)

    def get_qids_after_removing_fixed(
        self,
        fixed_level_qids: Iterable[int] | None = None,
        fixed_change_qids: Iterable[int] | None = None,
    ) -> tuple[int, ...]:
        """
        """

    #]


def blaze(
    im: _np.ndarray,
    /,
    eids: Iterable[int] = None,
    qids: Iterable[int] = None,
    #
    return_info: bool = False,
) -> tuple[Block, ...] | tuple[tuple[Block, ...], dict[str, Any]]:
    """
    Complete sequential block analysis of incidence matrix
    """
    #[
    if eids is None:
        eids = range(im.shape[0])
    if qids is None:
        qids = range(im.shape[1])
    eids = tuple(eids)
    qids = tuple(qids)
    #
    # Step 1: Prefetch quantities/equations that can be ordered first
    # individually (an equation only has one quantity), and ordered last
    # individually (a quantity only occurs in one equation)
    #
    # dm_eids, dm_qids, dm_im, = dulmage_mendelsohn_reordering(im, )
    eids_first, qids_first, eids_last, qids_last, eids_inner, qids_inner, im_inner \
        = prefetch(im, eids=eids, qids=qids, )
    #
    # Step 2: If there is a remaining core of interdependent equations, run
    # a naive triangularization algorithm on it
    #
    if im_inner.size:
        eids_inner, qids_inner, im_inner, *_ \
            = triangularize_inner_block(im_inner, eids=eids_inner, qids=qids_inner, )

        # from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn as dm
        # import scipy as _sp
        # im_inner = im
        # rr, cc = im_inner.nonzero()
        # z = _sp.sparse.coo_matrix((im_inner[rr, cc], (rr, cc, )), shape=im_inner.shape, )
        # row_part, col_part, = dm(z)
        # row_perm = _np.concatenate((
        #     row_part.unmatched,
        #     row_part.underconstrained,
        #     row_part.square,
        #     row_part.overconstrained
        #     )).astype(int)
        # col_perm = _np.concatenate((
        #     col_part.unmatched,
        #     col_part.underconstrained,
        #     col_part.square,
        #     col_part.overconstrained
        #     )).astype(int)
        # im_inner = im_inner[row_perm, :]
        # im_inner = im_inner[:, col_perm]
        # eids_inner = tuple(_np.array(eids_inner, dtype=int)[row_perm])
        # qids_inner = tuple(_np.array(qids_inner, dtype=int)[col_perm])
    #
    # Combine the results
    #
    eids = eids_first + eids_inner + eids_last
    qids = qids_first + qids_inner + qids_last
    #
    # Step 3: Create a tuple with a Block object for each block of
    # equations and equantities
    #
    first_blocks = tuple(Block((eid, ), (qid, ), ) for eid, qid in zip(eids_first, qids_first, ))
    inner_blocks = tuple(_generate_inner_blocks(im_inner, eids=eids_inner, qids=qids_inner, ))
    last_blocks = tuple(Block((eid, ), (qid, ), ) for eid, qid in zip(eids_last, qids_last, ))
    #
    out_blocks = first_blocks + inner_blocks + last_blocks
    if return_info:
        info = {
            "eids_first": eids_first,
            "qids_first": qids_first,
            "eids_last": eids_last,
            "qids_last": qids_last,
            "eids_inner": eids_inner,
            "qids_inner": qids_inner,
            "im_inner": im_inner,
        }
        return out_blocks, info
    else:
        return out_blocks
    #]


def _generate_inner_blocks(
    im: _np.ndarray,
    eids: Iterable[int],
    qids: Iterable[int],
) -> Generator[Block]:
    """
    """
    #[
    eids = tuple(eids)
    qids = tuple(qids)
    while im.size:
        block_size = next(i for i in range(1, im.shape[0] + 1) if not im[:i, i:].any())
        yield Block(eids[:block_size], qids[:block_size], )
        eids = eids[block_size:]
        qids = qids[block_size:]
        im = im[block_size:, block_size:]
    #]


def prefetch(
    im: _np.ndarray,
    /,
    eids: Iterable[int] = None,
    qids: Iterable[int] = None,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    _np.ndarray,
]:
    """
    Recursively prefetch singleton equations and quantities that can be ordered first and last
    """
    #[
    initial_size = im.size
    if eids is None:
        eids = tuple(range(im.shape[0]))
    if qids is None:
        qids = tuple(range(im.shape[1]))
    eids_first, qids_first, eids, qids, im = _prefetch_first(im, eids, qids, )
    eids_last, qids_last, eids, qids, im = _prefetch_last(im, eids, qids, )
    if im.size < initial_size:
        eids_first_next, qids_first_next, \
            eids_last_next, qids_last_next, \
            eids, qids, im, \
            = prefetch(im, eids=eids, qids=qids, )
        #
        eids_first = eids_first + eids_first_next
        qids_first = qids_first + qids_first_next
        #
        eids_last = eids_last_next + eids_last
        qids_last = qids_last_next + qids_last
        #
    return \
        eids_first, qids_first, \
        eids_last, qids_last, \
        eids, qids, im,
    #]


def sequentialize_strictly(
    im: _np.ndarray,
    /,
    eids: Iterable[int] | None = None,
    qids: Iterable[int] | None = None,
) -> tuple[int, ...]:
        """
        """
        #[
        if eids is None:
            eids = range(im.shape[0])
        if qids is None:
            qids = range(im.shape[1])
        eids = tuple(eids)
        qids = tuple(qids)
        #
        eids_first, qids_first, \
            eids_last, qids_last, \
            eids_rem, qids_rem, im_rem, \
            = prefetch(im, eids=eids, qids=qids, )
        #
        fail = \
            im_rem.size or eids_rem or qids_rem \
            or eids_first != qids_first \
            or eids_last != qids_last
        if fail:
            _wrongdoings.IrisPieError("Cannot find strict sequential reordering", )
        return eids_first + eids_last
        #]


def _prefetch_first(im, eids, qids, /, ):
    """
    Equations with a single incidence of quantities are reordered first
    """
    sum_in_rows = im.sum(axis=1, )
    index_rows = tuple(_np.where(sum_in_rows == 1)[0])
    index_columns = [
        _np.where(im[i, :] == 1)[0][0]
        for i in index_rows
    ]
    im = _np.delete(im, index_rows, axis=0, )
    im = _np.delete(im, index_columns, axis=1, )
    eids_first, eids_rem = _split_ids(eids, index_rows, )
    qids_first, qids_rem = _split_ids(qids, index_columns, )
    return eids_first, qids_first, eids_rem, qids_rem, im


def _prefetch_last(im, eids, qids, /, ):
    """
    Quantities with a single incidence in equations are reordered last
    """
    sum_in_columns = im.sum(axis=0, )
    index_columns = tuple(_np.where(sum_in_columns == 1)[0])
    index_rows = [
        _np.where(im[:, j] == 1)[0][0]
        for j in index_columns
    ]
    im = _np.delete(im, index_columns, axis=1, )
    im = _np.delete(im, index_rows, axis=0, )
    eids_last, eids_rem = _split_ids(eids, index_rows, )
    qids_last, qids_rem = _split_ids(qids, index_columns, )
    return eids_last, qids_last, eids_rem, qids_rem, im


def _split_ids(ids, index):
    extracted = tuple(ids[i] for i in index)
    remaining = tuple(id_ for i, id_ in enumerate(ids) if i not in index)
    return extracted, remaining


def triangularize_inner_block(
    im: _np.ndarray,
    /,
    eids: Iterable[int] = None,
    qids: Iterable[int] = None,
    max_iterations: int = 100,
) -> tuple[tuple[int, ...], tuple[int, ...], _np.ndarray, dict[str, Any]]:
    """
    Naive block triangularization
    """
    #[
    im = im.copy()
    if eids is None:
        eids = tuple(_np.arange(im.shape[0]))
    if qids is None:
        qids = tuple(_np.arange(im.shape[1]))
    #
    eids = _np.array(eids, dtype=int, )
    qids = _np.array(qids, dtype=int, )
    count = 0
    while True:
        #
        # Check if too many iterations
        if count >= max_iterations:
            break
        #
        count += 1
        prev_eids = eids
        prev_qids = qids
        #
        # Reorder columns (quantities) given rows (equations)
        column_reordering = _get_column_reordering(im)
        im = im[:, column_reordering]
        qids = qids[column_reordering]
        #
        # Reorder rows (equations) given columns (quantities)
        row_reordering = _get_row_reordering(im)
        im = im[row_reordering, :]
        eids = eids[row_reordering]
        #
        # Check if no change
        if _np.array_equal(prev_eids, eids) and _np.array_equal(prev_qids, qids):
            break
        #
    eids_reordered = tuple(eids)
    qids_reordered = tuple(qids)
    info = {"iterations": count}
    return eids_reordered, qids_reordered, im, info
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


def human_blocks_from_blocks(
    blocks: Iterable[Block],
    equations: Iterable[Equation],
    quantities: Iterable[Quantity],
) -> tuple[HumanBlock, ...]:
    """
    """
    return tuple(
        HumanBlock(block, equations, quantities)
        for block in blocks
    )

