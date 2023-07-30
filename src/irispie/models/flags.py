"""
"""


#[
from __future__ import (annotations, )

import enum as en_
#]


class Flags(en_.IntFlag, ):
    """
    """
    #[
    DEFAULT = 0
    LINEAR = en_.auto()
    FLAT = en_.auto()

    @property
    def is_linear(self, /, ) -> bool:
        return Flags.LINEAR in self

    @property
    def is_nonlinear(self, /, ) -> bool:
        return not self.is_linear

    @property
    def is_flat(self, /, ) -> bool:
        return Flags.FLAT in self

    @property
    def is_nonflat(self, /, ) -> bool:
        return not self.is_flat

    def update_from_kwargs(self, /, **kwargs) -> Self:
        linear = kwargs.get("linear") if kwargs.get("linear") is not None else self.is_linear
        flat = kwargs.get("flat") if kwargs.get("flat") is not None else self.is_flat
        return type(self).from_kwargs(linear=linear, flat=flat)

    @classmethod
    def from_kwargs(cls: type, **kwargs, ) -> Self:
        self = cls.DEFAULT
        if kwargs.get("linear"):
            self |= cls.LINEAR
        if kwargs.get("flat"):
            self |= cls.FLAT
        return self
    #]


