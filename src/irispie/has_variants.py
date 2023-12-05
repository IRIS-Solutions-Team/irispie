"""
"""


#[
from __future__ import annotations

from typing import (Self, Iterable, Iterator, TypeVar, )
from numbers import (Number, )
import copy as _co
import numpy as _np

from .conveniences import iterators as _iterators
#]


_T = TypeVar("T")


class HasVariantsMixin:
    """
    Mixin for handling multiple alternative variants of a single model object
    """
    #[

    @property
    def num_variants(self, /, ) -> int:
        """
        Number of alternative variants within this model
        """
        return len(self._variants)

    @property
    def is_singleton(self, /, ) -> bool:
        """
        True for models with only one variant
        """
        return self.num_variants == 1

    def new_with_shallow_variants(self, /, ) -> Self:
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

    def shrink_num_variants(self, new_num: int, /, ) -> None:
        """
        """
        if new_num<1:
            raise Exception('Number of variants must be one or more')
        self._variants = (
            self._variants[0:new_num]
            if new_num < self.num_variants
            else self._variants
        )

    def expand_num_variants(self, new_num: int, /, ) -> None:
        """
        """
        for i in range(self.num_variants, new_num):
            self._variants.append(self._variants[-1].copy())

    def broadcast_variants(self, other: Self, /, ) -> None:
        """
        """
        if self.num_variants < other.num_variants:
            self.expand_num_variants(other.num_variants, )

    def get_variant(
        self,
        vids: Iterable[int] | int | slice | EllipsisType,
        /,
    ) -> Self:
        new = type(self).skeleton()
        new._invariant = self._invariant
        variant_iter = _resolve_vids(self, vids, )
        new._variants = [ self._variants[i] for i in variant_iter ]
        return new

    def iter_variants(
        self,
        /,
    ) -> Iterator[Self]:
        """
        """
        return _iterators.exhaust_then_last(self, )

    def iter_extract_variants(self, /, ) -> Iterator:
        """
        """
        return _iterators.exhaust_then_last(self._variants, )

    def _new_with_single_variant(self: _T, variant, /, ) -> _T:
        """
        """
        new = type(self).skeleton()
        new._invariant = self._invariant
        new._variants = [ variant ]
        return new

    def __iter__(
        self,
        /,
    ) -> Iterator[Self]:
        """
        Iterate over alternative variants of this object
        """
        for v in self._variants:
            yield self._new_with_single_variant(v, )

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

