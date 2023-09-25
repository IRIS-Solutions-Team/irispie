"""
"""


#[
from __future__ import annotations

import numpy as _np
import scipy as _sp

from . import solutions as _solutions
#]


def get_autocov_square(
    solution: _solutions.Solution,
    cov_v: _np.ndarray,
    cov_w: _np.ndarray,
    order: int,
    /,
) -> _np.ndarray:
    """
    Autocovariance of [xi; y] with nans for unstable elements
    """
    #[
    boolex_stable = _np.hstack((
        solution.boolex_stable_transition_vector,
        solution.boolex_stable_measurement_vector,
    ))
    def fill_nans(cov: _np.ndarray, /, ) -> _np.ndarray:
        cov[~boolex_stable, :] = _np.nan
        cov[:, ~boolex_stable] = _np.nan
        return cov
    return tuple(
        fill_nans(cov, )
        for cov in get_autocov_square_00(solution, cov_v, cov_w, order, )
    )
    #]


def get_autocov_square_00(
    solution: _solutions.Solution,
    cov_v: _np.ndarray,
    cov_w: _np.ndarray,
    order: int,
    /,
) -> _np.ndarray:
    """
    Covariance of [xi; y] based on stable alpha
    """
    #[
    num_alpha = solution.num_alpha
    Ua = solution.Ua
    def transform_cov(cov: _np.ndarray, /, ) -> _np.ndarray:
        cov[:num_alpha, :] = Ua @ cov[:num_alpha, :]
        cov[:, :num_alpha] = cov[:, :num_alpha] @ Ua.T
        return cov
    return tuple(
        transform_cov(cov, )
        for cov in get_autocov_triangular_00(solution, cov_v, cov_w, order, )
    )
    #]


def get_autocov_triangular_00(
    solution: _solutions.Solution,
    cov_v: _np.ndarray,
    cov_w: _np.ndarray,
    order: int,
    /,
) -> tuple[_np.ndarray, ...]:
    """
    Autocovariance of [alpha; y] based on stable alpha
    """
    #[
    cov_triangular_00 = get_cov_triangular_00(solution, cov_v, cov_w, )
    #
    num_unit_roots = solution.num_unit_roots
    num_alpha = solution.num_alpha
    num_y = solution.num_y
    Ta_00 = solution.Ta.copy()
    Ta_00[:num_unit_roots, :] = 0
    Ta_00[:, :num_unit_roots] = 0
    Za = solution.Za
    A = _np.block([
        [Ta_00, _np.zeros((num_alpha, num_y), dtype=float, )],
        [Za @ Ta_00, _np.zeros((num_y, num_y), dtype=float, )],
    ])
    #
    autocov_triangular_00 = [None, ] * (order + 1)
    autocov_triangular_00[0] = cov_triangular_00
    for i in range(order):
        autocov_triangular_00[i+1] = A @ autocov_triangular_00[i]
    return tuple(autocov_triangular_00)
    #]


def get_cov_triangular_00(
    solution: _solutions.Solution,
    cov_v: _np.ndarray,
    cov_w: _np.ndarray,
    /,
) -> _np.ndarray:
    """
    Covariance of [alpha; y] based on stable alpha
    """
    #[
    # Triangular transition equations
    #
    cov_alpha_stable, cov_alpha_00 = get_cov_alpha_stable(solution, cov_v, )
    #
    # Measurement equations
    #
    Za_stable = solution.Za_stable
    H = solution.H
    sigma_w = H @ cov_w @ H.T
    cov_y_00 = Za_stable @ cov_alpha_stable @ Za_stable.T + sigma_w
    #
    # Cross-covariance between transition and measurement equations
    #
    Za = solution.Za
    cov_alpha_y_00 = cov_alpha_00 @ Za.T
    #
    # Full covariance matrix based on stable alpha
    #
    return _np.block([
        [cov_alpha_00, cov_alpha_y_00],
        [cov_alpha_y_00.T, cov_y_00],
    ])
    #]


def get_cov_alpha_stable(
    solution: _solutions.Solution,
    cov_v: _np.ndarray,
    /,
) -> _np.ndarray:
    """
    Covariance of stable alpha, and full alpha based on stable alpha
    """
    #[
    num_unit_roots = solution.num_unit_roots
    Ta_stable = solution.Ta_stable
    Ra_stable = solution.Ra_stable
    sigma_v = Ra_stable @ cov_v @ Ra_stable.T
    cov_alpha_stable = \
        _sp.linalg.solve_discrete_lyapunov(Ta_stable, sigma_v, )
    cov_alpha_00 = _np.zeros(solution.Ta.shape, dtype=float, )
    cov_alpha_00[num_unit_roots:, num_unit_roots:] = cov_alpha_stable
    return cov_alpha_stable, cov_alpha_00
    #]


def acorr_from_acov(
    acov_by_order: tuple[_np.ndarray, ...],
    /,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    acorr_by_order = [None, ] * len(acov_by_order)
    is_singleton = _is_singleton(acov_by_order[0], )
    if is_singleton:
        acov_by_order = _add_dim_to_singleton(acov_by_order, )
    #
    # Scale matrix has inv_std[i]*inv_std[j]
    #
    scale_matrix = _np.dstack(tuple(
        _get_scale_matrix(acov_by_order[0][:, :, d], )
        for d in range(acov_by_order[0].shape[2])
    ))
    #
    # Rescale autocovariance matrices to autocorrelation matrices order by
    # order
    #
    acorr_by_order = tuple(
        cov * scale_matrix
        for cov in acov_by_order
    )
    #
    if is_singleton:
        acorr_by_order = _remove_dim_from_singleton(acorr_by_order, )
    return acorr_by_order
    #]


def _get_scale_matrix(
    cov: _np.ndarray,
    /,
    tolerance: float = 1e-12,
) -> _np.ndarray:
    """
    """
    #[
    inv_std = _np.diag(cov).copy()
    index_positive = _np.abs(inv_std) > 0
    inv_std[index_positive] = 1 / _np.sqrt(inv_std[index_positive])
    inv_std[~index_positive] = 0
    return inv_std.reshape(-1, 1, ) @ inv_std.reshape(1, -1, )
    #]


def _is_singleton(
    array: _np.ndarray,
    /,
) -> bool:
    """
    """
    return array.ndim == 2


def _add_dim_to_singleton(
    array_by_order: tuple[_np.ndarray, ...],
    /,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    return tuple(
        array[:, :, _np.newaxis]
        for array in array_by_order
    )
    #]


def _remove_dim_from_singleton(
    array_by_order: tuple[_np.ndarray, ...],
    /,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    return tuple(
        array[:, :, 0]
        for array in array_by_order
    )
    #]

