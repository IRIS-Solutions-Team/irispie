r"""
Prior dummy observations
"""


#[

from __future__ import annotations

import numpy as _np
from numbers import Real

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import NoReturn

#]


__all__ = (
    "LittermanDummies",
)


class _VecAutoregDimsMixin:
    r"""
    """
    #[

    @property
    def num_endogenous(self, ) -> int:
        return self.dims[0]

    @property
    def num_exogenous(self, ) -> int:
        return self.dims[1]

    @property
    def has_intercept(self, ) -> bool:
        return bool(self.dims[2])

    @property
    def num_nonendogenous(self, ) -> int:
        return self.num_exogenous + int(self.has_intercept)

    @property
    def order(self, ) -> int:
        return self.dims[3]

    #]


class LittermanDummies(
    _VecAutoregDimsMixin,
):
    r"""
    """
    #[

    def __init__(
        self,
        dims: tuple[int, int, bool, int],
        rho: Real | _np.ndarray,
        mu: Real | _np.ndarray,
        kappa: Real,
    ) -> None:
        r"""
        """
        self.dims = dims
        self.rho = _ensure_array(rho, self.num_endogenous, )
        self.mu = _ensure_array(mu, self.num_endogenous, )
        self.kappa = kappa
        #
        self.num_dummies = self.num_endogenous * self.order
        self._populate_y0()
        self._populate_y1()
        self._populate_x()

    def _populate_y0(self, ) -> None:
        r"""
        """
        num_endogenous = self.num_endogenous
        order = self.order
        self.y0 = _np.hstack([
            _np.diag(self.mu * self.rho),
            _np.zeros((num_endogenous, num_endogenous * (order - 1)), ),
        ], dtype=float)

    def _populate_y1(self, ) -> None:
        r"""
        """
        order_multipliers = ( (i+1)**self.kappa for i in range(self.order) )
        self.y1 = _np.diag(_np.hstack([ self.mu * i for i in order_multipliers ]))

    def _populate_x(self, ) -> None:
        r"""
        """
        shape = (self.num_nonendogenous, self.num_dummies, )
        self.x = _np.zeros(shape, dtype=float, )

    #]


def _ensure_array(
    x: Real | _np.ndarray,
    num: int,
) -> _np.ndarray | NoReturn:
    r"""
    """
    #[
    shape = (num, )
    if isinstance(x, Real) or x.size == 1:
        x = _np.full(shape, x)
    if x.shape != shape:
        raise ValueError(f"Expected {shape} but got {x.shape}", )
    return x.astype(float, )
    #]


