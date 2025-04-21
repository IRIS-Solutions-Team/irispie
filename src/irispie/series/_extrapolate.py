"""
Autoregressive extrapolation
"""


#[

from __future__ import annotations

from typing import (TYPE_CHECKING, )
from numbers import (Number, )
import numpy as _np
import scipy as _sp
import documark as _dm

from .. import dates as _dates
from ._functionalize import FUNC_STRING

if TYPE_CHECKING:
    from typing import (Iterable, )
    from numbers import (Real, )
    from ..dates import (Period, )

#]


__all__ = []


class Inlay:
    """
    """
    #[

    @_dm.reference(category="homogenizing", )
    def extrapolate(
        self: Self,
        ar_coeffs: Iterable[Real],
        span: Iterable[Period],
        *,
        intercept: Real = 0,
        log: bool = False,
    ) -> None:
        r"""
......................................................................

==Extrapolate time series using autoregressive process==


### Functional forms creating a new time `Series` object ###


    new = extrapolate(
        self,
        ar_coeffs,
        span,
        *,
        intercept=0,
        log=False,
    )


### Class method changing an existing time `Series` object in-place ###


    self.extrapolate(
        ar_coeffs,
        span,
        *,
        intercept=0,
        log=False,
    )


### Input arguments ###


???+ input "self"
    The time `Series` object to be extrapolated by an autoregressive process.

???+ input "ar_coeffs"
    The autoregressive coefficients to be used in the extrapolation, entered as
    a tuple of AR_1, AR_2, ..., AR_p coefficients as if on the RHS of the AR
    process definition; see Details below.

???+ input "span"
    The time span on which the time series will be extrapolated.

???+ input "intercept"
    The intercept in the autorergressive process.

???+ input "log"
    If `log=True`, the time series will be logarithmized before the
    extrapolation and then delogarithmized back.


### Returns ###


???+ returns "self"
    The existing time `Series` object with its values replaced in-place.

???+ returns "new"
    A new time `Series` object.


### Details ###


The new extrapolated observations are created using this $p$-th order
autoregressive process defined recursively as:

$$
x_t = \rho_1 \, x_{t-1} + \cdots + \rho_p \, x_{t-p} + c, \qquad t = 1, \dots, T
$$

where

* the initial condion $x_{0},\ x_{-1}, \, \dots,\ x_{-p+1}$ are taken from
the existing observations in the input series `self`;

* the autoregressive coefficents
$\rho_1,\ \rho_2,\ \dots,\ \rho_p$ given by the input argument `ar_coeff`

* $c$ is the `intercept`.

......................................................................
        """
        if self.start is None:
            return
        span = self.resolve_periods(span, )
        if not span:
            return
        num_periods = len(span)
        ar_coeffs = (ar_coeffs, ) if isinstance(ar_coeffs, Number) else ar_coeffs
        order = len(ar_coeffs)
        ar = (1, ) + tuple(-i for i in ar_coeffs)
        start = span[0]
        initial_from_until = (start-order, start-1, )
        iter_initial_data = self.iter_own_data_variants_from_until(initial_from_until, )
        new_data = _np.hstack(tuple(
            _extrapolate_data(
                initial_data, num_periods, ar,
                intercept=intercept,
                log=log,
            )
            for initial_data in iter_initial_data
        ))
        self.set_data(span, new_data, )

    #]


attributes = (n for n in dir(Inlay) if not n.startswith("_"))
for n in attributes:
    code = FUNC_STRING.format(n=n, )
    exec(code, globals(), locals(), )
    __all__.append(n)


def _extrapolate_data(
    initial: _np.ndarray,
    num_periods: int,
    ar: Iterable[Real],
    *,
    intercept: Real = 0,
    log: bool = False,
) -> None:
    """
    """
    #[
    initial = _np.flip(initial.flatten(), )
    if log:
        initial = _np.log(initial)
    innovation = _np.full((num_periods, ), intercept, dtype=_np.float64, )
    ma = (1, )
    zi = _sp.signal.lfiltic(ma, ar, initial, )
    out, *_ = _sp.signal.lfilter(ma, ar, innovation, zi=zi, axis=0, )
    if log:
        out = _np.exp(out)
    return out.reshape(-1, 1, )
    #]


