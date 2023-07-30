"""
Incidence tokens and wrt tokens
"""

#[
from collections.abc import (Iterable, )
from numbers import (Number, )
import operator as _op
import itertools as _it
import numpy as _np

from typing import (
    NamedTuple, Callable, Self, 
    Protocol, TypeAlias,
)

from . import (quantities as _qu, )
#]


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
    #]


"""
"""
Tokens: TypeAlias = Iterable[Token]


def get_max_shift(tokens: Tokens) -> int:
    return max(tok.shift for tok in tokens) if tokens else None


def get_min_shift(tokens: Tokens) -> int:
    return min(tok.shift for tok in tokens) if tokens else None


def get_max_qid(tokens: Tokens) -> int:
    return max(tok.qid for tok in tokens) if tokens else None


def generate_qids_from_tokens(tokens: Tokens) -> Iterable[int]:
    return (tok.qid for tok in tokens)


def get_some_shift_by_quantities(tokens: Tokens, something: Callable) -> dict:
    key = _op.attrgetter("qid")
    sorted_tokens = sorted(tokens, key=key)
    return { k: something(t.shift for t in tokens) for k, tokens in _it.groupby(sorted_tokens, key=key) }


def generate_tokens_of_kinds(tokens: Tokens, qid_to_kind: dict, kinds: _qu.QuantityKind) -> Tokens:
    return (tok for tok in tokens if qid_to_kind[tok.qid] in kinds)


def sort_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
    """
    Sort tokens by shift and id
    """
    return sorted(tokens, key=lambda x: (-x.shift, x.qid))


def is_qid_in_tokens(tokens: Tokens, qid: int) -> bool:
    return any(tok.qid == qid for tok in tokens)


def print_tokens(
    tokens: Tokens,
    qid_to_name: dict[int, str]
) -> Iterable[str]:
    """
    Create list of printed tokens
    """
    return [ t.print(qid_to_name) for t in tokens ]


