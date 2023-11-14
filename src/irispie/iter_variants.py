"""
"""


#[
from __future__ import annotations

from numbers import (Number, )
import copy as _co

from .conveniences import iterators as _iterators
#]


class IterVariantsMixin:
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
    ) -> Self:
        """
        Alter (expand, shrink) the number of alternative parameter variants in this model object
        """
        if new_num < self.num_variants:
            self.shrink_num_variants(new_num, )
        elif new_num > self.num_variants:
            self.expand_num_variants(new_num, )
        return self

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
            self._variants.append(_co.deepcopy(self._variants[-1]))

    def get_variant(
        self,
        vids: Iterable[int] | int | slice | EllipsisType,
        /,
    ) -> Self:
        new = self.new_with_shallow_variants()
        variant_iter = _resolve_vids(new, vids, )
        new._variants = [ new._variants[i] for i in variant_iter ]
        return new

    def iter_variants(
        self,
        /,
    ) -> Iterator[Self]:
        """
        """
        yield from _iterators.exhaust_then_last(self, )

    def _new_with_single_variant(
        self,
        variant: _variants.Variant,
        /,
    ) -> Self:
        new = self.new_with_shallow_variants()
        new._variants = [ variant ]
        return new

    def __iter__(
        self,
        /,
    ) -> Iterator[Self]:
        """
        Iterate over alternative variants of this model object
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

