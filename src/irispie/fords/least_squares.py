r"""
Multivariate linear least squares estimators
"""

#[

from __future__ import annotations

import numpy as _np

#]


__all__ = (
    "lstsq",
    "destandardize_lstsq",
    "standardize",
    "destandardize",
)


def lstsq(
    lhs: _np.ndarray,
    rhs: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    Mx = rhs @ rhs.T
    My = rhs @ lhs.T
    beta = _np.linalg.solve(Mx, My).T
    res = lhs - beta @ rhs
    return beta, res,


def destandardize_lstsq(bb, lhs, rhs, ) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    mean_lhs, std_lhs = lhs
    mean_rhs, std_rhs = rhs
    inv_sigma_lhs = _np.diag(1 / std_lhs)
    sigma_rhs = _np.diag(std_rhs)
    b = inv_sigma_lhs @ bb @ sigma_rhs
    c = mean_lhs - b @ mean_rhs
    return b, c,


def standardize(
    x: _np.ndarray,
) -> tuple[_np.ndarray, tuple[_np.ndarray, _np.ndarray]]:
    r"""
    """
    mean = _np.mean(x, axis=1, )
    std = _np.std(x, axis=1, mean=mean.reshape(-1, 1, ), )
    standardized = (x - mean.reshape(-1, 1, )) / std.reshape(-1, 1, )
    mean_std = (mean, std, )
    return standardized, mean_std


def destandardize(
    x: _np.ndarray,
    mean_std : tuple[_np.ndarray, _np.ndarray],
) -> _np.ndarray:
    r"""
    """
    mean, std = mean_std
    return x * std.reshape(-1, 1, ) + mean.reshape(-1, 1, ),

