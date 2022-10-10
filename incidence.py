
from __future__ import annotations
from typing import NamedTuple, Union, Iterable, Optional, Callable


def get_max_shift(inputs: Union[set[Incidence], set[Token]], **kwargs) -> int:
    return max(generate_shifts(inputs, **kwargs))


def get_min_shift(inputs: Union[set[Incidence], set[Token]], **kwargs) -> int:
    return min(generate_shifts(inputs, **kwargs)) if inputs else None


def get_max_quantity_id(inputs: Union[set[Incidence], set[Token]], **kwargs) -> int:
    return max(generate_quantity_ids(inputs, **kwargs)) if inputs else None


def generate_quantity_ids(inputs: Union[set[Incidence], set[Token]], **kwargs) -> Iterable[int]:
    return (qid for i in inputs if (qid := i.get_quantity_id(**kwargs)) is not None)


def generate_shifts(inputs: Union[set[Incidence], set[Token]], **kwargs) -> Iterable[int]:
    return (sh for i in inputs if (sh := i.get_shift(**kwargs)) is not None)


def generate_tokens(incidences: Iterable[Incidence]) -> Iterable[Token]:
    return (inc.token for inc in incidences)


def generate_equation_ids(incidences: Iterable[Incidence]) -> Iterable[int]:
    return (inc.equation_id for inc in incidences)


class Token(NamedTuple):
    #(
    quantity_id: int
    shift: int

    def get_quantity_id(self, shift_test: Callable=lambda x: True) -> Optional[int]:
        return self.quantity_id if shift_test(self.shift) else None

    def get_shift(self, quantity_test: Callable=lambda x: True) -> Optional[int]:
        return self.shift if quantity_test(self.quantity_id) else None
    #)


class Incidence(NamedTuple):
    #(
    equation_id: Optional[int]
    token: Token

    def get_shift(self, equation_test: Callable=lambda x: True, **kwargs) -> Optional[int]:
        return self.token.get_shift(**kwargs) if equation_test(self.equation_id) else None

    def get_quantity_id(self, equation_test: Callable=lambda x: True, **kwargs) -> Optional[int]:
        return self.token.get_quantity_id(**kwargs) if equation_test(self.equation_id) else None
    #)

