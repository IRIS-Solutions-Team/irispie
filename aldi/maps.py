

#[
from __future__ import annotations

from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )
import itertools as it_
import dataclasses as dc_

#]


@dc_.dataclass
class ArrayMap:
    """
    """
    lhs: tuple[list[int], list[int]] | None = None
    rhs: tuple[list[int], list[int]] | None = None
    #[
    def __init__(self, /, ) -> NoReturn:
        self.lhs = ([], [])
        self.rhs = ([], [])

    def __len__(self, /, ) -> int:
        return len(self.lhs[0])

    def append(
        self,
        lhs: tuple[int, int], 
        rhs: tuple[int, int],
        /
    ) -> NoReturn:
        """
        """
        self.lhs[0].append(lhs[0])
        self.lhs[1].append(lhs[1])
        self.rhs[0].append(rhs[0])
        self.rhs[1].append(rhs[1])

    def merge_with(
        self,
        other: Self,
    ) -> NoReturn:
        """
        """
        self.lhs = (self.lhs[0]+other.lhs[0], self.lhs[1]+other.lhs[1])
        self.rhs = (self.rhs[0]+other.rhs[0], self.rhs[1]+other.rhs[1])

    def offset(
        self,
        lhs_row: int, 
        rhs_row_offset: int,
    ) -> NoReturn:
        """
        """
        self.lhs = ([lhs_row if i is not None else None for i in self.lhs[0]], self.lhs[1])
        self.rhs = ([i+rhs_row_offset if i is not None else None for i in self.rhs[0]], self.rhs[1])

    def _remove_nones(self) -> NoReturn:
        """
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


    @classmethod
    def for_equation(
        cls,
        tokens_in_equation_on_rhs: in_.Tokens,
        tokens_in_columns_on_lhs: in_.Tokens,
        rhs_offset: int,
        lhs_row: int,
    ) -> Self:
        """
        """
        index = tokens_in_columns_on_lhs.index
        raw_map = (
            (lhs_row, index(t), rhs_row, 0) 
            for rhs_row, t in enumerate(tokens_in_equation_on_rhs, start=rhs_offset)
            if t in tokens_in_columns_on_lhs
        )
        # Collect all lhr_rows, all lhs_columns, all rhs_rows, all rhs_columns
        raw_map = zip(*raw_map)
        self = cls()
        try:
            self.lhs = (list(next(raw_map)), list(next(raw_map)))
            self.rhs = (list(next(raw_map)), list(next(raw_map)))
        except:
            self.lhs = ([], [])
            self.rhs = ([], [])
        # Equivalent to:
        # self = cls()
        # for rhs_row, t in enumerate(tokens_in_equation_on_rhs, start=rhs_offset):
            # if t in tokens_in_columns_on_lhs:
                # lhs_column = tokens_in_columns_on_lhs.index(t)
                # self.append((lhs_row, lhs_column), (rhs_row, 0))
        return self

    @classmethod
    def constant_vector(
        cls,
        eids: Iterable[int],
    ) -> Self:
        """
        """
        num_equations = len(eids)
        self = cls()
        self.lhs = (list(range(num_equations)), [0]*num_equations)
        self.rhs = (list(eids), [0]*num_equations)
        return self
    #]


def vstack_array_maps(maps: Iterable[ArrayMap]) -> ArrayMap:
    """
    """
    #[
    stacked_map = ArrayMap()
    for m in maps:
        stacked_map.merge_with(m)
    return stacked_map
    #]


def create_eid_to_rhs_offset(
    eids: Iterable[int],
    eid_to_wrt_tokens: dict[int, in_.Tokens],
) -> dict[int, int]:
    """
    Cumulative sum of number of tokens in individual equations, starting with 0
    """
    rhs_offsets = list(it_.accumulate(
        len(eid_to_wrt_tokens[i]) 
        for i in eids
    ))
    # Offset starts from 0 for the first equations
    rhs_offsets.pop()
    rhs_offsets.insert(0, 0)
    return dict(zip(eids, rhs_offsets))


