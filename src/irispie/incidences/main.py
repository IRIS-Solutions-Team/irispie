"""
Incidence tokens and wrt tokens
"""


#[
from collections.abc import (Iterable, )
from numbers import (Number, )
from typing import (
    NamedTuple, Callable, Self, 
    Protocol, TypeAlias,
)
import operator as _op
import itertools as _it
import numpy as _np

from .. import quantities as _quantities
#]


_PRINT_TOKEN = "x[({qid},t{shift:+g})]"
_PRINT_TOKEN_ZERO_SHIFT = "x[({qid},t)]"


class Token(NamedTuple):
    """
    Incidence
    """
    #[
    qid: int
    shift: int

    def shifted(
        self: Self,
        by: int,
    ) -> Self:
        return Token(self.qid, self.shift+by)

    def print(
        self,
        qid_to_name: dict[int, str], 
    ) -> str:
        s = qid_to_name[self.qid]
        if self.shift:
            s += f"{{{self.shift:+g}}}"
        return s

    def print_xtring(
        self,
    ) -> str:
        return (
            _PRINT_TOKEN.format(qid=self.qid, shift=self.shift)
            if self.shift else _PRINT_TOKEN_ZERO_SHIFT.format(qid=self.qid)
        )
    #]


def get_max_shift(tokens: Iterable[Token]) -> int:
    return max(tok.shift for tok in tokens) if tokens else None


def get_min_shift(tokens: Iterable[Token]) -> int:
    return min(tok.shift for tok in tokens) if tokens else None


def get_max_qid(tokens: Iterable[Token]) -> int:
    return max(tok.qid for tok in tokens) if tokens else None


def generate_qids_from_tokens(tokens: Iterable[Token]) -> Iterable[int]:
    return (tok.qid for tok in tokens)


def get_some_shift_by_quantities(tokens: Iterable[Token], something: Callable) -> dict:
    key = _op.attrgetter("qid")
    sorted_tokens = sorted(tokens, key=key)
    return { k: something(t.shift for t in tokens) for k, tokens in _it.groupby(sorted_tokens, key=key) }


def generate_tokens_of_kinds(tokens: Iterable[Token], qid_to_kind: dict, kinds: _quantities.QuantityKind) -> Iterable[Token]:
    return (tok for tok in tokens if qid_to_kind[tok.qid] in kinds)


def sort_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
    """
    Sort tokens by shift and id
    """
    return sorted(tokens, key=lambda x: (-x.shift, x.qid))


def is_qid_in_tokens(tokens: Iterable[Token], qid: int) -> bool:
    return any(tok.qid == qid for tok in tokens)


def print_tokens(
    tokens: Iterable[Token],
    qid_to_name: dict[int, str]
) -> Iterable[str]:
    """
    Create list of printed tokens
    """
    return [ t.print(qid_to_name) for t in tokens ]


def rows_and_columns_from_tokens_in_horizontal(
    tokens: Iterable[Token],
    column_zero: int,
) -> tuple[Iterable[int], Iterable[int]]:
    """
    """
    rows, columns = zip(*((t.qid, column_zero+t.shift) for t in tokens))
    return tuple(rows), tuple(columns)

