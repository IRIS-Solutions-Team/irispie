
from __future__ import annotations

from typing import NamedTuple, Union, Generator, Optional, Callable


def get_max_shift(inputs: Union[set[Incidence], set[Token]], **kwargs) -> int:
    return max(collect_shifts(inputs, **kwargs))


def get_min_shift(inputs: Union[set[Incidence], set[Token]], **kwargs) -> int:
    return min(collect_shifts(inputs, **kwargs)) if inputs else None


def get_max_quantity_id(inputs: Union[set[Incidence], set[Token]], **kwargs) -> int:
    return max(collect_quantity_ids(inputs, **kwargs)) if inputs else None


def collect_quantity_ids(inputs: Union[set[Incidence], set[Token]], **kwargs) -> Generator[int, None, None]:
    return (sh for i in inputs if (sh := i.get_quantity_id(**kwargs)) is not None)


def collect_shifts(inputs: Union[set[Incidence], set[Token]], **kwargs) -> Generator[int, None, None]:
    return (sh for i in inputs if (sh := i.get_shift(**kwargs)) is not None)


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

