"""
"""


#[
from __future__ import annotations

import enum as _en
#]


class Flags(_en.IntFlag, ):
    """
    """
    #[

    DEFAULT = 0
    LINEAR = _en.auto()
    FLAT = _en.auto()
    DETERMINISTIC = _en.auto()

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

    @property
    def is_deterministic(self, /, ) -> bool:
        return Flags.DETERMINISTIC in self

    @property
    def is_stochastic(self, /, ) -> bool:
        return not self.is_deterministic

    def update_from_kwargs(self, /, **kwargs) -> Self:
        linear = kwargs.get("linear") if kwargs.get("linear") is not None else self.is_linear
        flat = kwargs.get("flat") if kwargs.get("flat") is not None else self.is_flat
        deterministic = kwargs.get("deterministic") if kwargs.get("deterministic") is not None else self.is_deterministic
        return type(self).from_kwargs(linear=linear, flat=flat, deterministic=deterministic, )

    @classmethod
    def from_kwargs(cls: type, **kwargs, ) -> Self:
        self = cls.DEFAULT
        if kwargs.get("linear") or kwargs.get("is_linear"):
            self |= cls.LINEAR
        if kwargs.get("flat") or kwargs.get("is_flat"):
            self |= cls.FLAT
        if kwargs.get("deterministic") or kwargs.get("is_deterministic"):
            self |= cls.DETERMINISTIC
        return self

    def to_portable(self, /, ) -> dict[str, bool]:
        """
        """
        return {
            "is_linear": self.is_linear,
            "is_flat": self.is_flat,
            "is_deterministic": self.is_deterministic,
        }

    @classmethod
    def from_portable(klass, portable: dict[str, bool], ) -> Self:
        return klass.from_kwargs(**portable, )

    #]

