"""
"""


#[
from __future__ import annotations

from typing import (Any, Self, Iterable, Iterator, TypeVar, Protocol, )
from numbers import (Number, )
import numpy as _np
import functools as _ft

from .conveniences import iterators as _iterators
#]


_T = TypeVar("T")


class HasVariantsProtocol:
    """
    """
    #[

    _invariant: Any | None
    _variants: list | None

    #]


class Mixin:
    """
    Mixin for handling multiple alternative variants of a single model object
    """
    #[

    @classmethod
    def skeleton(
        klass,
        other,
    ) -> Self:
        r"""
        """
        return klass(invariant=other._invariant, )

    def select_variants(
        self,
        selectors: Iterable[bool] | None = None,
        indices: Iterable[int] | None = None,
    ) -> None:
        if (selectors is None) + (indices is None) != 1:
            raise Exception("Either selectors or indices must be specified, but not both.")
        if selectors is not None:
            new_variants = [ v for v, s in zip(self._variants, selectors) if s ]
        if indices is not None:
            new_variants = [ self._variants[i] for i in indices ]
        self._variants = new_variants

    @property
    def num_variants(self, ) -> int:
        """
        Number of alternative variants within this model
        """
        return len(self._variants)

    @property
    def is_singleton(self, ) -> bool:
        """
        True for models with only one variant
        """
        return is_singleton(self.num_variants, )

    def new_with_shallow_variants(self, ) -> Self:
        """
        Create a new model with pointers to invariant and variants of this model
        """
        new = type(self)()
        new._invariant = self._invariant
        new._variants = [ v for v in self._variants ]
        return new

    def alter_num_variants(
        self,
        new_num: int,
        /,
    ) -> None:
        """
        Alter (expand, shrink) the number of alternative parameter variants in this model object
        """
        if new_num < self.num_variants:
            self.shrink_num_variants(new_num, )
        elif new_num > self.num_variants:
            self.expand_num_variants(new_num, )

    def shrink_num_variants(self, new_num: int, ) -> None:
        r"""
        """
        if new_num < 1:
            raise Exception('Number of variants must be one or more')
        self._variants = (
            self._variants[0:new_num]
            if new_num < self.num_variants
            else self._variants
        )

    def expand_num_variants(self, new_num: int, ) -> None:
        r"""
        """
        if new_num < 1:
            raise Exception('Number of variants must be one or more')
        for i in range(self.num_variants, new_num):
            self._variants.append(self._variants[-1].copy(), )

    def resolve_num_variants_in_context(self, custom_num_variants: int | None, ) -> int:
        """
        """
        if custom_num_variants is None:
            return self.num_variants
        else:
            return custom_num_variants

    def broadcast_variants(self, other: Self, ) -> None:
        """
        """
        if self.num_variants < other.num_variants:
            self.expand_num_variants(other.num_variants, )

    def get_variant(
        self,
        vids: Iterable[int] | int | slice | EllipsisType,
    ) -> Self:
        new = self.skeleton(self, )
        new._variants = [ self._variants[i] for i in _resolve_vids(self, vids, ) ]
        return new

    def iter_variants(self, ) -> Iterator[Self]:
        r"""
        """
        return _iterators.exhaust_then_last(self, )

    def iter_extract_variants(self, ) -> Iterator:
        """
        """
        return _iterators.exhaust_then_last(self._variants, )

    def iter_own_variants(self, ) -> Iterator[Self]:
        r"""
        Iterate over alternative variants of this object
        """
        for v in self._variants:
            new = self.skeleton(self, )
            new._variants = [v, ]
            yield new

    def __iter__(self, ) -> Iterator[Self]:
        r"""
        Iterate over alternative variants of this object
        """
        return self.iter_own_variants()

    def repack_singleton(
        self: HasVariantsProtocol,
        anything: list[_T] | _T,
        /,
    ) -> list[_T]:
        """
        """
        return [anything, ] if not isinstance(anything, list) else anything

    def unpack_singleton(
        self: HasVariantsProtocol,
        anything: list[_T] | _T,
        **kwargs,
    ) -> list[_T] | _T:
        """
        """
        return unpack_singleton(
            anything,
            self.is_singleton,
            **kwargs,
        )

    def unpack_singleton_in_dict(
        self: HasVariantsProtocol,
        anything_in_dict: dict[str, list[_T] | _T],
        **kwargs,
    ) -> dict[str, _T | list[_T]]:
        """
        """
        return unpack_singleton_in_dict(
            anything_in_dict,
            self.is_singleton,
            **kwargs,
        )
    #]


def _resolve_vids(
    self,
    vids: Iterable[int] | int | slice | EllipsisType,
    /,
) -> Iterable[int]:
    """
    """
    #[
    if isinstance(vids, Number):
        return (int(vids), )
    elif vids is ...:
        return range(self.num_variants, )
    elif isinstance(vids, slice):
        return range(*vids.indices(self.num_variants, ), )
    else:
        return tuple(int(i) for i in vids)
    #]


def iter_variants(anything: _T, ) -> Iterator[_T]:
    """
    """
    if hasattr(anything, "iter_variants", ):
        return anything.iter_variants()
    elif isinstance(anything, _np.ndarray, ) and anything.ndim:
        return _iterators.exhaust_then_last(anything.reshape(anything.shape[0], -1, ).T)
    elif isinstance(anything, list, ):
        return _iterators.exhaust_then_last(anything, )
    else:
        return _iterators.exhaust_then_last([anything, ], )


def iter_own_variants(anything: _T, ) -> Iterator[_T]:
    """
    """
    if hasattr(anything, "iter_own_variants", ):
        return anything.iter_own_variants()
    elif isinstance(anything, _np.ndarray, ) and anything.ndim:
        return anything.reshape(anything.shape[0], -1, ).T
    elif isinstance(anything, list, ):
        return anything
    else:
        return [anything, ]


def unpack_singleton(
    anything: list[_T],
    is_singleton: bool | None = True,
    unpack_singleton: bool = True,
) -> _T | list[_T]:
    """
    """
    if not unpack_singleton:
        return anything
    if is_singleton is None:
        is_singleton = len(anything) == 1
    return anything[0] if is_singleton else anything


shadow_unpack_singleton = unpack_singleton


def unpack_singleton_in_dict(
    anything_in_dict: dict[str, list[_T] | _T],
    is_singleton: bool,
    /,
    unpack_singleton: bool = True,
) -> dict[str, _T | list[_T]]:
    """
    """
    if is_singleton and unpack_singleton:
        return {
            k: shadow_unpack_singleton(v, )
            for k, v in anything_in_dict.items()
        }
    else:
        return anything_in_dict


def is_singleton(num_variants: int) -> bool:
    """
    """
    return num_variants == 1


def unpack_singleton_decorator(func: Callable, ):
    """
    """
    #[
    @_ft.wraps(func, )
    def _wrapper(self, *args, **kwargs, ):
        unpack_singleton = kwargs.pop("unpack_singleton", True)
        output = func(self, *args, **kwargs)
        return self.unpack_singleton(output, unpack_singleton=unpack_singleton, )
    return _wrapper
    #]

