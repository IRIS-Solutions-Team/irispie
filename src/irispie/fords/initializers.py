"""
Initialize median and MSE matrix for alpha vector
"""


#[
from __future__ import annotations

import numpy as _np
from numbers import (Real, )

from .solutions import (Solution, left_div, )
from . import covariances as _covariances
#]


_DEFAULT_DIFFUSE_SCALE = 1e8


def _approx_diffuse(
        solution: Solution,
        custom_diffuse_scale: Real | None = None,
) -> tuple[Real, _np.ndarray | None]:
    """
    """
    #[
    diffuse_scale = custom_diffuse_scale or _DEFAULT_DIFFUSE_SCALE
    unknown_init_impact = None
    return diffuse_scale, unknown_init_impact
    #]


def _fixed_unknown(
        solution: Solution,
        custom_diffuse_scale: Real | None = None,
) -> tuple[Real, _np.ndarray | None]:
    """
    """
    #[
    diffuse_scale = 0
    unknown_init_impact = (
        _np.eye(solution.num_xi, solution.num_unit_roots, )
        if solution.num_unit_roots else None
    )
    return diffuse_scale, unknown_init_impact,
    #]


def _fixed_zero(
        solution: Solution,
        custom_diffuse_scale: Real | None = None,
) -> tuple[Real, _np.ndarray | None]:
    """
    """
    #[
    diffuse_scale = 0
    unknown_init_impact = None
    return diffuse_scale, unknown_init_impact,
    #]


_RESOLVE_DIFFUSE = {
    "approx_diffuse": _approx_diffuse,
    "fixed_unknown": _fixed_unknown,
    "fixed_zero": _fixed_zero,
}


def initialize(
    solution: Solution,
    cov_u: _np.ndarray,
    *,
    diffuse_method: Literal["approx_diffuse", "fixed_unknown", "fixed_zero", ] = "fixed_unknown",
    diffuse_scale: Real | None = None,
) -> tuple[np_.ndarray, np_.ndarray, _np.ndarray, ]:
    r"""
    Return median and MSE matrix for initial alpha, and the impact of fixed
    unknowns on initial alpha
    """
    #[
    diffuse_func = _RESOLVE_DIFFUSE[diffuse_method]
    diffuse_scale, unknown_init_impact = diffuse_func(solution, diffuse_scale, )
    init_med = _initialize_med(solution, )
    init_mse = _initialize_mse(solution, cov_u, diffuse_scale, )
    return init_med, init_mse, unknown_init_impact,
    #]


def _initialize_med(solution: Solution, ) -> _np.ndarray:
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
    init_med = _np.zeros((num_alpha, ), dtype=float, )
    #
    T = _np.eye(num_stable, dtype=float, ) - Ta_stable
    init_med_stable = left_div(T, Ka_stable, )
    init_med[num_unit_roots:] = init_med_stable
    return init_med
    #]


def _initialize_mse(
    solution: Solution,
    cov_u: _np.ndarray,
    diffuse_scale: Real | None = None,
) -> _np.ndarray:
    """
    """
    #[
    init_mse = _covariances.get_cov_alpha_00(solution, cov_u, )
    if diffuse_scale:
        num_unit_roots = solution.num_unit_roots
        init_mse[:num_unit_roots, :num_unit_roots] = \
            _initialize_mse_unstable_approx_diffuse(solution, cov_u, init_mse, diffuse_scale, )
    init_mse = _covariances.symmetrize(init_mse, )
    return init_mse
    #]


def _initialize_mse_unstable_approx_diffuse(
    solution: Solution,
    cov_u: _np.ndarray,
    init_mse: _np.ndarray,
    diffuse_scale: Real | None = None,
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


def _mean_of_diag(x: _np.ndarray, ) -> Real:
    """
    """
    return _np.mean(_np.diag(x, ), )

