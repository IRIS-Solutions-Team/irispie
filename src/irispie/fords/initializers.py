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


_DIFFUSE_SCALE = 1e8


def initialize(
    *args,
    method: Literal["fixed", "approx_diffuse", ] = "approx_diffuse",
    **kwargs,
) -> tuple[np_.ndarray, np_.ndarray]:
    """
    """
    return _RESOLVE_METHOD[method](*args, **kwargs, )


def _initialize_fixed(*args, **kwargs, ) -> tuple[_np.ndarray, _np.ndarray]:
    """
    """
    return (
        _initialize_mean(*args, **kwargs, ),
        _initialize_mse_fixed(*args, **kwargs, ),
    )


def _initialize_approx_diffuse(
    *args,
    **kwargs,
    ) -> tuple[_np.ndarray, _np.ndarray]:
    """
    """
    return (
        _initialize_mean(*args, **kwargs, ),
        _initialize_mse_approx_diffuse(*args, **kwargs, ),
    )


def _initialize_mean(
    solution: _solutions.Solution,
    *args,
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
    init_mean = _np.zeros((num_alpha, 1), dtype=float, )
    #
    T = _np.eye(num_stable, dtype=float, ) - Ta_stable
    init_mean[num_unit_roots:, :] = _solutions.left_div(T, Ka_stable, )
    return init_mean
    #]


def _initialize_mse_fixed(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
) -> _np.ndarray:
    """
    """
    return _covariances.get_cov_alpha_00(solution, cov_u, )


def _initialize_mse_approx_diffuse(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
    /,
) -> _np.ndarray:
    """
    """
    #[
    num_unit_roots = solution.num_unit_roots
    num_alpha = solution.num_alpha
    init_mse = _covariances.get_cov_alpha_00(solution, cov_u, )
    init_mse[:num_unit_roots, :num_unit_roots] = \
        _initialize_mse_unstable_approx_diffuse(solution, cov_u, init_mse, )
    return init_mse
    #]


def _initialize_mse_unstable_approx_diffuse(
    solution,
    cov_u: _np.ndarray,
    init_mse: _np.ndarray,
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
    scale = _mean_of_diag(base_cov, ) * _DIFFUSE_SCALE
    return scale * _np.eye(num_unit_roots, dtype=float, )
    #]


def _mean_of_diag(x: _np.ndarray, /, ) -> Real:
    """
    """
    return _np.mean(_np.diag(x, ), )


_RESOLVE_METHOD = {
    "fixed": _initialize_fixed,
    "approx_diffuse": _initialize_approx_diffuse,
}

