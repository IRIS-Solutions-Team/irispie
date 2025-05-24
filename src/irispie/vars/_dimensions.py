r"""
"""


#[

from __future__ import annotations

from typing import NamedTuple

#]


class Dimensions(NamedTuple, ):
    r"""
    """
    #[

    num_endogenous: int | None = None
    order: int | None = None
    has_intercept: bool = True
    num_exogenous: int = 0

    @property
    def num_nonendogenous(self, ) -> int:
        return self.num_exogenous + int(self.has_intercept)

    @property
    def num_lagged_endogenous(self, ) -> int:
        return self.num_endogenous * self.order

    @property
    def num_rhs(self, ) -> int:
        return self.num_lagged_endogenous + self.num_nonendogenous

    @property
    def has_exogenous(self, ) -> bool:
        return self.num_exogenous > 0

    #]


