
#---{ External imports

from typing import (
    Iterable, NamedTuple, Callable, Self,
    TypeAlias,
)

from itertools import (
    chain,
)

#---}


#---{ Internal imports

from .quantities import (
    Quantity, QuantityKind, create_qid_to_kind
)

#---}


class Token(NamedTuple):
    """
    """
    qid: int
    shift: int

    def lag(self: Self) -> Self:
        return Token(self.qid, self.shift-1)

    def lead(self: Self) -> Self:
        return Token(self.qid, self.shift+1)

    def print(self: Self, qid_to_name: dict[int, str]) -> str:
        s = qid_to_name[self.qid]
        if self.shift:
            s += f"{{{self.shift:+g}}}"
        return s


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


def get_some_shifts_by_quantities(tokens: Tokens, some: Callable) -> dict[int, int]:
    tokens = set(tokens)
    unique_qids = set(generate_qids_from_tokens(tokens))
    return {
        qid: _get_some_shift_for_quantity(tokens, qid, some)
        for qid in unique_qids
    }


def generate_tokens_of_kinds(tokens: Tokens, qid_to_kind: dict, kinds: QuantityKind) -> Tokens:
    return (tok for tok in tokens if qid_to_kind[tok.qid] in kinds)


def sort_tokens(tokens: Tokens) -> Tokens:
    """
    Sort tokens by shift and id
    """
    return sorted(tokens, key=lambda x: (-x.shift, x.qid))


def print_tokens(
    tokens: Tokens,
    id_to_name: dict[int, str]
) -> Iterable[str]:
    """
    Create list of printed tokens
    """
    return [ t.print(id_to_name) for t in tokens ]


def _get_some_shift_for_quantity(tokens: Tokens, qid: int, some: Callable) -> int:
    return some(tok.shift for tok in tokens if tok.qid==qid)


