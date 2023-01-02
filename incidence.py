
from numpy import(
    zeros, vstack
)

from typing import (
    Iterable, NamedTuple, Callable, Self
)

from itertools import (
    chain,
)


from .quantities import (
    Quantity, QuantityKind, create_id_to_kind
)



class Token(NamedTuple):
    quantity_id: int
    shift: int

    def lag(self: Self) -> Self:
        return Token(self.quantity_id, self.shift-1)

    def lead(self: Self) -> Self:
        return Token(self.quantity_id, self.shift+1)

    def print(self: Self, id_to_name: dict[int, str]) -> str:
        s = id_to_name[self.quantity_id]
        if self.shift:
            s += f"{{{self.shift:+g}}}"
        return s


Tokens = Iterable[Token]


def get_max_shift(tokens: Tokens) -> int:
    return max(tok.shift in tokens) if tokens else None


def get_min_shift(tokens: Tokens) -> int:
    return min(tok.shift in tokens) if tokens else None


def get_max_quantity_id(tokens: Tokens) -> int:
    return max(tok.quantity_id in tokens) if tokens else None


def generate_quantity_ids_from_tokens(tokens: Tokens) -> Iterable[int]:
    return (tok.quantity_id for tok in tokens)


def _get_some_shift_for_quantity(tokens: Tokens, quantity_id: int, some: Callable) -> int:
    return some(tok.shift for tok in tokens if tok.quantity_id==quantity_id)


def get_some_shifts_by_quantities(tokens: Tokens, some: Callable) -> dict[int, int]:
    tokens = set(tokens)
    unique_quantity_ids = set(generate_quantity_ids_from_tokens(tokens))
    return {
        quantity_id: _get_some_shift_for_quantity(tokens, quantity_id, some)
        for quantity_id in unique_quantity_ids
    }


def generate_tokens_of_kind(tokens: Tokens, id_to_kind: dict, kind: QuantityKind) -> Tokens:
    return (tok for tok in tokens if id_to_kind[tok.quantity_id] is kind)


