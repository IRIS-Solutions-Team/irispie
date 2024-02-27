"""
Initialize mean and MSE matrix for alpha vector
"""


#[
from __future__ import annotations

import numpy as _np
from numbers import (Real, )

from . import solutions as _solutions
from . import covariances as _covariances
#]


_DEFAULT_DIFFUSE_SCALE = 1e7


def initialize(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
    mse_method: Literal["fixed", "approx_diffuse", ] = "approx_diffuse",
    diffuse_scale: Real | None = None,
) -> tuple[np_.ndarray, np_.ndarray]:
    """
    """
    diffuse_scale = (
        diffuse_scale if diffuse_scale is not None
        else _DEFAULT_DIFFUSE_SCALE
    )
    if mse_method == "fixed":
        diffuse_scale = 0.0
    return (
        _initialize_mean(solution, ),
        _initialize_mse(solution, cov_u, diffuse_scale, ),
        diffuse_scale,
    )


def _initialize_mean(
    solution: _solutions.Solution,
    /,
) -> _np.ndarray:
    """
    Solve alpha_stable = Ta_stable @ alpha_stable + Ka_stable for alpha_stable
    and return alpha with 0s for unstable elements
    """
    #[
    num_alpha = solution.num_alpha
    num_unit_roots = solution.num_unit_roots
    num_stable = solution.num_stable
    Ta_stable = solution.Ta_stable
    Ka_stable = solution.Ka_stable
    init_mean = _np.zeros((num_alpha, ), dtype=float, )
    #
    T = _np.eye(num_stable, dtype=float, ) - Ta_stable
    init_mean[num_unit_roots:] = _solutions.left_div(T, Ka_stable, )
    return init_mean
    #]


def _initialize_mse(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
    diffuse_scale: Real | None = None,
    /,
) -> _np.ndarray:
    """
    """
    #[
    init_mse = _covariances.get_cov_alpha_00(solution, cov_u, )
    if diffuse_scale:
        num_unit_roots = solution.num_unit_roots
        init_mse[:num_unit_roots, :num_unit_roots] = \
            _initialize_mse_unstable_approx_diffuse(solution, cov_u, init_mse, diffuse_scale, )
    return init_mse
    #]


def _initialize_mse_unstable_approx_diffuse(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
    init_mse: _np.ndarray,
    diffuse_scale: Real | None = None,
    /,
) -> _np.ndarray:
    """
    """
    #[
    num_unit_roots = solution.num_unit_roots
    num_alpha = solution.num_alpha
    base_cov = (
        init_mse[num_unit_roots:, num_unit_roots:] if num_alpha
        else (cov_u if cov_u.size else _np.ones((1, 1, ), dtype=float, ))
    )
    scale = _mean_of_diag(base_cov, ) * diffuse_scale
    return scale * _np.eye(num_unit_roots, dtype=float, )
    #]


def _mean_of_diag(
    x: _np.ndarray,
    /,
) -> Real:
    """
    """
    return _np.mean(_np.diag(x, ), )

