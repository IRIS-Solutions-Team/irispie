r"""
Prior observations
"""


#[

from __future__ import annotations

import numpy as _np
import math as _mt
from numbers import Real
from typing import Protocol

from ._dimensions import Dimensions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import NoReturn, Self

#]


__all__ = (
    "MinnesotaPriorObs",
    "MeanPriorObs",
)


class PriorObs(Protocol, ):
    r"""
    """
    #[

    def get_num_obs(self, dimensions: Dimensions, ) -> int:
        ...

    def generate_y0(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        ...

    def generate_y1(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        ...

    def generate_x(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        ...

    def generate_k(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        ...

    def __iter__(self, ) -> Iterable[Self, ]:
        yield self

    def generate_lhs(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        r"""
        """
        return self.generate_y0(dimensions, )

    def generate_rhs(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        r"""
        """
        return _np.vstack([
            self.generate_y1(dimensions, ),
            self.generate_x(dimensions, ),
            self.generate_k(dimensions, ),
        ])

    def _populate_mu(
        self,
        mu: Real | None = None,
        mu2: Real | None = None,
    ) -> None:
        r"""
        """
        if (mu is not None) + (mu2 is not None) != 1:
            raise ValueError("Exactly one of mu and mu2 must be specified.")
        self.mu = mu if mu is not None else _mt.sqrt(mu2)

    #]


class MinnesotaPriorObs(PriorObs, ):
    r"""
    """
    #[

    def __init__(
        self,
        rho: Real | _np.ndarray = 0,
        mu: Real | None = None,
        mu2: Real | None = None,
        kappa: Real = 0,
    ) -> None:
        r"""
        """
        self.rho = rho
        self._populate_mu(mu, mu2, )
        self.kappa = kappa

    def generate_y0(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        r"""
        """
        dims = Dimensions(*dimensions)
        num_endogenous, order, = dims.num_endogenous, dims.order,
        try:
            rho = _ensure_array(self.rho, num_endogenous, )
        except ValueError:
            raise ValueError("Invalid dimensions of MinnesotaPriorObs parameter rho.", )
        y_std = _ensure_array(y_std, num_endogenous, )
        mu_scaled = self.mu * y_std
        return _np.hstack([
            _np.diag(mu_scaled * rho),
            _np.zeros((num_endogenous, num_endogenous * (order - 1)), ),
        ], dtype=float, )

    def generate_y1(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        r"""
        """
        dims = Dimensions(*dimensions)
        num_endogenous, order, = dims.num_endogenous, dims.order,
        y_std = _ensure_array(y_std, num_endogenous, )
        mu_scaled = self.mu * y_std
        order_multipliers = ( (i+1)**self.kappa for i in range(order) )
        y1_diag = _np.hstack([ mu_scaled * i for i in order_multipliers ], dtype=float, )
        return _np.diag(y1_diag, )

    def generate_x(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> None:
        r"""
        """
        dims = Dimensions(*dimensions)
        num_exogenous = dims.num_exogenous
        num_obs = self.get_num_obs(dimensions, )
        shape = (num_exogenous, num_obs, )
        return _np.zeros(shape, dtype=float, )

    def generate_k(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> None:
        r"""
        """
        dims = Dimensions(*dimensions)
        num_obs = self.get_num_obs(dimensions, )
        shape = (int(dims.has_intercept), num_obs, )
        return _np.zeros(shape, dtype=float, )

    def get_num_obs(self, dimensions: Dimensions, ) -> int:
        r"""
        """
        dims = Dimensions(*dimensions)
        num_endogenous, order, = dims.num_endogenous, dims.order,
        return num_endogenous * order

    #]


class MeanPriorObs(PriorObs, ):
    r"""
    """
    #[

    def __init__(
        self,
        mean: Real | _np.ndarray = 0,
        mu: Real | None = None,
        mu2: Real | None = None,
    ) -> None:
        r"""
        """
        self._populate_mu(mu, mu2, )
        self.mean = mean

    def generate_y0(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        r"""
        """
        dims = Dimensions(*dimensions)
        num_endogenous, has_intercept, = dims.num_endogenous, dims.has_intercept,
        try:
            mean = _ensure_array(self.mean, dims.num_endogenous, )
        except ValueError:
            raise ValueError("Invalid dimensions of MeanPriorObs parameter mean.", )
        shape = (dims.num_endogenous, int(dims.has_intercept), )
        return _np.ones(shape, dtype=float, ) * mean.reshape((-1, 1)) * self.mu

    def generate_y1(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> _np.ndarray:
        r"""
        """
        y0 = self.generate_y0(dimensions, y_std, )
        dims = Dimensions(*dimensions)
        return _np.tile(y0, (dims.order, 1, ))

    def generate_x(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> None:
        r"""
        """
        dims = Dimensions(*dimensions)
        shape = (dims.num_exogenous, self.get_num_obs(dimensions, ), )
        return _np.zeros(shape, dtype=float, )

    def generate_k(
        self,
        dimensions: Dimensions,
        y_std: _np.ndarray | Real = 1,
    ) -> None:
        r"""
        """
        num_obs = self.get_num_obs(dimensions, )
        dims = Dimensions(*dimensions)
        shape = (int(dims.has_intercept), num_obs, )
        return _np.ones(shape, dtype=float, ) * self.mu

    def get_num_obs(self, dimensions: Dimensions, ) -> int:
        r"""
        """
        dims = Dimensions(*dimensions)
        return int(dims.has_intercept)

    #]


def _ensure_array(
    x: Real | _np.ndarray,
    num: int,
) -> _np.ndarray | NoReturn:
    r"""
    """
    #[
    shape = (num, )
    if not isinstance(x, _np.ndarray) or x.size == 1:
        x = _np.full(shape, x)
    if x.shape != shape:
        raise ValueError(f"Expected shape {shape} but got {x.shape}", )
    return x.astype(float, )
    #]


def arrays_from_prior_obs(
    prior_obs: PriorObs | Iterable[PriorObs, ] | None,
    dimensions: Dimensions,
    y_std: _np.ndarray | Real = 1,
) -> tuple[_np.ndarray, _np.ndarray, ] | None:
    r"""
    """
    #[
    if prior_obs is None:
        return None
    lhs, rhs, = zip(*(
        (i.generate_lhs(dimensions, y_std, ), i.generate_rhs(dimensions, y_std, ), )
        for i in prior_obs
    ))
    lhs = _np.hstack(lhs, )
    rhs = _np.hstack(rhs, )
    return lhs, rhs,
    #]

