r"""
"""


#[

from __future__ import annotations

from typing import Literal
import functools as _ft
import numpy as _np
import daqp as _qp
import ctypes as _ct

from ..dates import Period
from .main import Series

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable
    from numbers import Real
    from typing import Any, Callable

#]


__all__ = ("lonf", )


OrderType = Literal[1, 2]


def lonf(
    input_series,
    order: OrderType,
    smooth: Real,
    #
    span: Iterable[Period] | None = None,
) -> tuple[Any, Any]:
    r"""
................................................................................

==L1 norm based filter==

................................................................................
    """

    # Resolve periods
    periods = input_series.resolve_periods(span, )
    start_period = periods[0]
    end_period = periods[-1]
    from_until = (start_period, end_period, )
    num_periods = len(periods)

    # Set up solver function
    matrix_setup_func = _MATRIX_SETUP_DISPATCH[order]
    d, D, = matrix_setup_func(num_periods, )
    H = D @ D.T;
    A = _np.eye(num_periods-order, dtype=_ct.c_double, )
    bounds = _np.full((num_periods-order, ), smooth, dtype=_ct.c_double, );
    sense = _np.full((num_periods-order, ), 0, dtype=_ct.c_int, )
    solve = _ft.partial(_qp.solve, H=H, A=A, bupper=bounds, blower=-bounds, sense=sense, )

    # Iterate over data variants and solve the QP problem
    trend_data_variants = []
    gap_data_variants = []
    for data in input_series.iter_own_data_variants_from_until(from_until, ):
        trend_data, gap_data = _lonf_for_variant(solve, data, D, )
        trend_data_variants.append(trend_data, )
        gap_data_variants.append(gap_data, )

    # Create output series
    trend_series = Series(start=start_period, values=trend_data_variants, )
    gap_series = Series(start=start_period, values=gap_data_variants, )

    return trend_series, gap_series,


def _lonf_for_variant(
    solve: Callable,
    data: _np.ndarray,
    D: _np.ndarray,
) -> tuple[tuple[Real, ...], tuple[Real, ...]]:
    r"""
    """
    data = data.flatten()
    data.dtype = _ct.c_double
    f = -D @ data;
    x, *_ = solve(f=f, )
    gap_data = D.T @ x;
    trend_data = data - gap_data;
    trend_data = tuple(trend_data.flatten().tolist())
    gap_data = tuple(gap_data.flatten().tolist())
    return trend_data, gap_data,

#]


def _first_order_matrix_setup(
    num_periods: int,
) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    order = 1
    d = _np.eye(num_periods-order, num_periods, dtype=_ct.c_double, )
    D = _np.eye(num_periods-order, num_periods, dtype=_ct.c_double, )
    D[:, 1:] = D[:, 1:] - d[:, :-1];
    return d, D,


def _second_order_matrix_setup(
    num_periods: int,
) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    order = 2
    d = _np.eye(num_periods-order, num_periods, dtype=_ct.c_double, )
    D = _np.eye(num_periods-order, num_periods, dtype=_ct.c_double, )
    D[:, 1:] = D[:, 1:] - 2 * d[:, :-1] 
    D[:, 2:] = D[:, 2:] + d[:, :-2]
    return d, D,


_MATRIX_SETUP_DISPATCH = {
    1: _first_order_matrix_setup,
    2: _second_order_matrix_setup,
}

