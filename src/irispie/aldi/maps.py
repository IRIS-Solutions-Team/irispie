"""
Create a map from one column of a stacked Jacobian to a full (dense or
sparse) Jacobian matrix
"""


#[
from __future__ import annotations

from typing import (Any, )
from collections.abc import (Iterable, Collection, )
import itertools as it_
import dataclasses as dc_
#]


class ArrayMap:
    """
    """
    #[

    __slots__ = ('lhs', 'rhs', )

    def __init__(
        self,
        eids: list[int],
        eid_to_wrt_tokens: dict[int, Any],
        tokens_in_columns_on_lhs: list[Any],
        eid_to_rhs_offset: dict[int, int],
        /,
        **kwargs,
    ) -> None:
        """
        """
        self.lhs = ([], [], )
        self.rhs = ([], [], )
        #
        token_to_lhs_column = {
            t: i
            for i, t in enumerate(tokens_in_columns_on_lhs, )
        }
        #
        for lhs_row, eid in enumerate(eids):
            self.add_lhs_rhs(*_get_raw_map_for_single_equation(
                eid_to_wrt_tokens[eid],
                token_to_lhs_column,
                eid_to_rhs_offset[eid],
                lhs_row,
                **kwargs,
            ))


    def __len__(self, /, ) -> int:
        return len(self.lhs[0])

    def append(
        self,
        lhs: tuple[list[int], list[int]],
        rhs: tuple[list[int], list[int]],
        /
    ) -> None:
        """
        """
        self.lhs[0].append(lhs[0])
        self.lhs[1].append(lhs[1])
        self.rhs[0].append(rhs[0])
        self.rhs[1].append(rhs[1])

    def add_lhs_rhs(
        self,
        lhs_rows: Iterable[int],
        lhs_columns: Iterable[int],
        rhs_rows: Iterable[int],
        rhs_columns: Iterable[int],
        /,
    ) -> None:
        """
        """
        self.lhs = (self.lhs[0]+list(lhs_rows), self.lhs[1]+list(lhs_columns), )
        self.rhs = (self.rhs[0]+list(rhs_rows), self.rhs[1]+list(rhs_columns), )

    def remove_nones(self) -> None:
        """
        Remove any map entry that has a None for the row index on the LHS
        """
        if not self.lhs[0]:
            return
        zipped_pruned = [
            i for i in zip(self.lhs[0], self.lhs[1], self.rhs[0], self.rhs[1])
            if i[0] is not None
        ]
        unzipped_pruned = list(zip(*zipped_pruned))
        self.lhs = (list(unzipped_pruned[0]), list(unzipped_pruned[1]))
        self.rhs = (list(unzipped_pruned[2]), list(unzipped_pruned[3]))

class VectorMap:
    """
    """
    #[

    __slots__ = ('lhs', 'rhs', )

    def __init__(
        self,
        eids: Collection[int],
        /,
    ) -> None:
        """
        """
        num_equations = len(eids)
        self.lhs = (list(range(num_equations)), )
        self.rhs = (list(eids), )

    def __len__(self, /, ) -> int:
        return len(self.lhs[0])

    def append(
        self,
        lhs: tuple[list[int]],
        rhs: tuple[list[int]],
        /
    ) -> None:
        """
        """
        self.lhs[0].append(lhs[0])
        self.rhs[0].append(rhs[0])

    #]


def create_eid_to_rhs_offset(
    eids: tuple[int],
    eid_to_wrt_something: dict[int, Any],
    /,
) -> dict[int, int]:
    """
    Cumulative sum of number of tokens in individual equations, starting with 0
    """
    rhs_offsets = list(it_.accumulate(
        len(eid_to_wrt_something[i]) 
        for i in eids
    ))
    # Offset starts from 0 for the first equations
    rhs_offsets.pop()
    rhs_offsets.insert(0, 0)
    return dict(zip(eids, rhs_offsets))


def _get_raw_map_for_single_equation(
    tokens_in_equation_on_rhs: in_.Tokens,
    token_to_lhs_column: dict[Any, int],
    rhs_offset: int,
    lhs_row: int,
    /,
    rhs_column: int,
    lhs_column_offset: int,
) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
    """
    """
    raw_map = (
        (lhs_row, lhs_column_offset + token_to_lhs_column[t], rhs_row, rhs_column, )
        for rhs_row, t in enumerate(tokens_in_equation_on_rhs, start=rhs_offset)
        if t in token_to_lhs_column.keys()
    )
    # Collect all lhr_rows, all lhs_columns, all rhs_rows, all rhs_columns
    raw_map = zip(*raw_map)
    try:
        lhs_rows = next(raw_map, )
        lhs_columns = next(raw_map, )
        rhs_rows = next(raw_map, )
        rhs_columns = next(raw_map, )
    except:
        lhs_rows = ()
        lhs_columns = ()
        rhs_rows = ()
        rhs_columns = ()
    return lhs_rows, lhs_columns, rhs_rows, rhs_columns

