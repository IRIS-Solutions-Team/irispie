r"""
Multivariate linear least squares estimators
"""


#[

from __future__ import annotations

import numpy as _np
from numbers import Real

#]


__all__ = (
    "ordinary_least_squares",
    "destandardize_lstsq",
    "standardize",
    "destandardize",
)


def ordinary_least_squares(
    lhs: _np.ndarray,
    rhs: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    #[
    Mx = rhs @ rhs.T
    My = rhs @ lhs.T
    return _np.linalg.solve(Mx, My).T
    #]


def standardized_ordinary_least_squares(
    lhs: _np.ndarray,
    rhs: _np.ndarray,
    intercept: bool = True,
    dummy_obs: tuple[_np.ndarray, _np.ndarray, ] | None = None,
) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    #[
    if intercept:
        lhs_mean = None
        rhs_mean = None
    else:
        lhs_mean = 0
        rhs_mean = 0
    #
    # Standardize the data first, add dummy observations then
    lhs_standardized, lhs_mean_std, = standardize(lhs, mean=lhs_mean, )
    rhs_standardized, rhs_mean_std, = standardize(rhs, mean=rhs_mean, )
    #
    # Add dummy observations
    if dummy_obs is not None:
        lhs_dummy_obs, rhs_dummy_obs = dummy_obs
        lhs_standardized = _np.hstack((lhs_standardized, lhs_dummy_obs, ))
        rhs_standardized = _np.hstack((rhs_standardized, rhs_dummy_obs, ))
    #
    # Estimate the coefficients on standardized data
    b_standardized = ordinary_least_squares(
        lhs_standardized,
        rhs_standardized,
        intercept=False,
    )
    #
    # Destandardize the coefficients, calculate the intercept if needed
    b, c, = destandardize_lstsq(
        b_standardized,
        lhs_mean_std,
        rhs_mean_std,
        intercept=intercept,
    )
    #
    return b, c,
    #]


def destandardize_lstsq(
    bb,
    lhs_mean_std,
    rhs_mean_std,
) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    #[
    mean_lhs, std_lhs = lhs_mean_std
    mean_rhs, std_rhs = rhs_mean_std
    sigma_lhs = _np.diag(std_lhs)
    inv_sigma_rhs = _np.diag(1 / std_rhs)
    b = sigma_lhs @ bb @ inv_sigma_rhs
    c = None
    if intercept:
        c = mean_lhs - b @ mean_rhs
    return b, c,
    #]


def standardize(
    x: _np.ndarray,
    mean: _np.ndarray | Real | None = None,
) -> tuple[_np.ndarray, tuple[_np.ndarray, _np.ndarray]]:
    r"""
    """
    #[
    if mean is None:
        mean = _np.mean(x, axis=1, )
    elif isinstance(mean, Real):
        mean = _np.full((x.shape[0], ), mean, dtype=x.dtype, )
    std = _np.std(x, axis=1, mean=mean.reshape(-1, 1, ), )
    x_standardized = (x - mean.reshape(-1, 1, )) / std.reshape(-1, 1, )
    x_mean_std = (mean, std, )
    return x_standardized, x_mean_std,
    #]


def destandardize(
    x: _np.ndarray,
    mean_std : tuple[_np.ndarray, _np.ndarray],
) -> _np.ndarray:
    r"""
    """
    mean, std = mean_std
    return x * std.reshape(-1, 1, ) + mean.reshape(-1, 1, ),


def add_intercept_dummy(x: _np.ndarray, ) -> _np.ndarray:
    r"""
    """
    return _np.pad(
        x,
        ((0, 1), (0, 0), ),
        mode="constant",
        constant_values=1,
    )

