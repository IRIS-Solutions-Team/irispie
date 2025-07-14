r"""
"""


#[

from __future__ import annotations

import numpy as _np
import scipy as _sp
import warnings as _wa
from typing import Callable
from numbers import Real

from . import solutions as _solutions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

#]


__all__ = (
    "std_from_cov",
    "cov_from_std",
    "CovarianceSimulator",
)


def get_autocov_square(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
    cov_w: _np.ndarray,
    order: int,
) -> _np.ndarray:
    """
    Autocovariance of [xi; y] with nans for unstable elements
    """
    #[
    boolex_stable = _np.hstack((
        solution.boolex_stable_transition_vector,
        solution.boolex_stable_measurement_vector,
    ))
    def fill_nans(cov: _np.ndarray, ) -> _np.ndarray:
        cov[~boolex_stable, :] = _np.nan
        cov[:, ~boolex_stable] = _np.nan
        return cov
    return tuple(
        fill_nans(cov, )
        for cov in get_autocov_square_00(solution, cov_u, cov_w, order, )
    )
    #]


def get_autocov_square_00(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
    cov_w: _np.ndarray,
    order: int,
) -> tuple[_np.ndarray, ...]:
    """
    Autocovariance (up to order) of [xi; y] based on stable alpha and a zero cov assumption for unstable alpha
    """
    #[
    num_alpha = solution.num_alpha
    Ua = solution.Ua
    #
    def _transform_cov_triangular_to_square(cov: _np.ndarray, ) -> _np.ndarray:
        cov[:num_alpha, :] = Ua @ cov[:num_alpha, :]
        cov[:, :num_alpha] = cov[:, :num_alpha] @ Ua.T
        return cov
    #
    return tuple(
        _transform_cov_triangular_to_square(cov, )
        for cov in get_autocov_triangular_00(solution, cov_u, cov_w, order, )
    )
    #]


def get_autocov_triangular_00(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
    cov_w: _np.ndarray,
    order: int,
) -> tuple[_np.ndarray, ...]:
    """
    Autocovariance (up to order) of [alpha; y] based on stable alpha
    """
    #[
    cov_triangular_00 = get_cov_triangular_00(solution, cov_u, cov_w, )
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
    cov_u: _np.ndarray,
    cov_w: _np.ndarray,
) -> _np.ndarray:
    """
    Covariance of [alpha; y] based on stable alpha
    """
    #[
    # Triangular transition equations
    cov_alpha_00 = get_cov_alpha_00(solution, cov_u, )
    #
    # Measurement equations
    num_unit_roots = solution.num_unit_roots
    cov_alpha_stable = cov_alpha_00[num_unit_roots:, num_unit_roots:]
    Za_stable = solution.Za_stable
    H = solution.H
    sigma_w = H @ cov_w @ H.T
    cov_y_00 = Za_stable @ cov_alpha_stable @ Za_stable.T + sigma_w
    #
    # Cross-covariance between transition and measurement equations
    Za = solution.Za
    cov_alpha_y_00 = cov_alpha_00 @ Za.T
    #
    # Full covariance matrix based on stable alpha
    return _np.block([
        [cov_alpha_00, cov_alpha_y_00],
        [cov_alpha_y_00.T, cov_y_00],
    ])
    #]


def get_cov_alpha_00(
    solution: _solutions.Solution,
    cov_u: _np.ndarray,
) -> _np.ndarray:
    """
    Covariance of stable alpha, and full alpha based on stable alpha
    """
    #[
    num_unit_roots = solution.num_unit_roots
    Ta_stable = solution.Ta_stable
    Pa_stable = solution.Pa_stable
    sigma_u = Pa_stable @ cov_u @ Pa_stable.T if Pa_stable is not None else cov_u
    cov_alpha_stable = _sp.linalg.solve_discrete_lyapunov(Ta_stable, sigma_u, )
    cov_alpha_00 = _np.zeros(solution.Ta.shape, dtype=float, )
    cov_alpha_00[num_unit_roots:, num_unit_roots:] = cov_alpha_stable
    return cov_alpha_00
    #]


def acorr_from_acov(
    acov_by_order: tuple[_np.ndarray, ...],
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    # Scale matrix has inv_std[i]*inv_std[j]
    scale_matrix = _get_scale_matrix(acov_by_order[0], )
    #
    # Rescale autocovariance matrices to autocorrelation matrices order by
    # order
    acorr_by_order = tuple(
        i * scale_matrix
        for i in acov_by_order
    )
    #
    return acorr_by_order
    #]


def _get_scale_matrix(
    cov: _np.ndarray,
    tolerance: float = 1e-12,
) -> _np.ndarray:
    """
    """
    #[
    inv_std = _np.diag(cov).copy()
    where_positive = inv_std > 0
    inv_std[where_positive] = 1 / _np.sqrt(inv_std[where_positive])
    inv_std[~where_positive] = 0
    return inv_std.reshape(-1, 1, ) @ inv_std.reshape(1, -1, )
    #]


def _is_singleton(
    array: _np.ndarray,
) -> bool:
    """
    """
    return array.ndim == 2


def _add_dim_to_singleton(
    array_by_order: tuple[_np.ndarray, ...],
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
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    return tuple(
        array[:, :, 0]
        for array in array_by_order
    )
    #]


def _lyapunov(
    T: _np.ndarray,
    Sigma: _np.ndarray,
) -> _np.ndarray:
    """
    """

    def left_div(A: _np.ndarray, B: _np.ndarray) -> _np.ndarray:
        """
        Solve A / B
        """
        return _np.linalg.solve(A, B, )

    def right_div(B: _np.ndarray, A: _np.ndarray) -> _np.ndarray:
        """
        Solve B / A
        """
        return _np.linalg.solve(A.T, B.T, ).T

    C = _np.zeros_like(T)
    i = T.shape[0]
    Tt = T.T

    while i >= 1:
        if i == 1 or T[i-1, i-2] == 0:
            # 1x1 block with a real eigenvalue.
            C[i-1, i:] = C[i:, i-1].T
            c = right_div(
                Sigma[i-1, :i] + T[i-1, i-1] * C[i-1, i:] @ Tt[i:, :i] + T[i-1, i:] @ C[i:, :] @ Tt[:, :i],
                _np.eye(i) - T[i-1, i-1] * Tt[:i, :i],
            )
            C[i-1, :i] = c
            i -= 1
        else:
            # 2x2 block corresponding to a pair of complex eigenvalues.
            C[i-2:i, i:] = C[i:, i-2:i].T
            X = T[i-2:i, i-2:i] @ C[i-2:i, i:] @ Tt[i:, :i] + T[i-2:i, i:] @ C[i:, :] @ Tt[:, :i] + Sigma[i-2:i, :i]
            # Solve
            #     c = T(i-1:i, i-1:i)*c*Tt(1:i, 1:i)
            # Transpose the equation first
            #     c' = Tt'*c'*T' + X'
            # so that the below kronecker product becomes faster to evaluate,
            # then vectorize
            #     vec(c') = kron(T, Tt')*vec(c') + vec(X').
            Xt = X.T
            U = Tt[:i, :i].T
            k = _np.block([
                [ T[i-2, i-2]*U, T[i-2, i]*U ],
                [ T[i-1, i-2]*U, T[i-1, i]*U ],
            ])
            ct = left_div(_np.eye(2*i) - k, Xt.ravel(), )
            C[i-2:i, :i] = ct.reshape(i, 2).T
            i -= 2

    return C


def symmetrize(X: _np.ndarray, ) -> _np.ndarray:
    """
    """
    return (X + X.T) / 2


def std_from_cov(
    cov: _np.ndarray,
    trim_negative: bool = True,
) -> _np.ndarray:
    r"""
    """
    #[
    diag = _np.diag(cov, )
    if trim_negative:
        sqrt = sqrt_positive
    else:
        sqrt = _np.sqrt
    return sqrt(diag, )
    #]


def sqrt_positive(x):
    return _np.sqrt(_np.maximum(x, 0.0, ))


def cov_from_std(
    std: _np.ndarray,
    trim_negative: bool = True,
) -> _np.ndarray:
    r"""
    """
    #[
    if trim_negative:
        std = _np.maximum(std, 0)
    return _np.diag(std**2, )
    #]


def corr_std_from_cov(
    cov: _np.ndarray,
    trim_negative: bool = True,
) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    #[
    std = std_from_cov(cov, trim_negative=trim_negative, )
    corr = cov / std.reshape(-1, 1, ) / std.reshape(1, -1, )
    return corr, std,
    #]


def cov_from_corr_std(
    corr: _np.ndarray,
    std: _np.ndarray,
    trim_negative: bool = True,
) -> _np.ndarray:
    r"""
    """
    #[
    if trim_negative:
        std = _np.maximum(std, 0)
    return corr * std.reshape(-1, 1, ) * std.reshape(1, -1, )
    #]


def autocorr_std_from_autocov(
    autocov: Iterable[_np.ndarray, ...],
    trim_negative: bool = True,
) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    """
    #[
    std = std_from_cov(autocov[0], trim_negative=trim_negative, )
    corr = tuple(
        i / std.reshape(-1, 1, ) / std.reshape(1, -1, )
        for i in autocov
    )
    return autocorr, std,
    #]


def get_acorr_by_variant(
    self,
    acov: tuple[_np.ndarray, ...] | list[tuple[_np.ndarray, ...]] | None = None,
    up_to_order: int = 0,
    unpack_singleton: bool = True,
) -> tuple[_np.ndarray, ...] | list[tuple[_np.ndarray, ...]]:
    r"""
    """
    if acov is None:
        acov_by_variant = self.get_acov(
            up_to_order=up_to_order,
            unpack_singleton=False,
        )
    else:
        acov_by_variant = self.repack_singleton(acov, )
    #
    acorr_by_variant = [
        acorr_from_acov(i)
        for i in acov_by_variant
    ]
    acorr_by_variant = self.unpack_singleton(
        acorr_by_variant,
        unpack_singleton=unpack_singleton,
    )
    return acorr_by_variant


_DEFAULT_NUMPY_RNG = _np.random.default_rng()
_DEFAULT_GENERATOR = _DEFAULT_NUMPY_RNG.standard_normal


class CovarianceSimulator:
    r"""
    """
    #[

    def __init__(
        self,
        cov: _np.ndarray,
        mean: _np.ndarray | Real | None = None,
        #
        generator: Callable | None = None,
        make_symmetric: bool = True,
        trim_negative: bool = True,
    ) -> None:
        r"""
        """
        if make_symmetric:
            cov = symmetrize(cov, )
        eig_values, eig_vectors = _np.linalg.eigh(cov, )
        # eig_vectors, eig_values, *_ = _np.linalg.svd(cov, )
        if trim_negative:
            print(_np.min(eig_values))
            eig_values = _np.maximum(eig_values, 0)
        if _np.any(eig_values < 0):
            raise ValueError("Negative eigenvalues found in covariance matrix.")
        std_devs = _np.sqrt(eig_values, )
        self.factor = eig_vectors @ _np.diag(std_devs, )
        self._populate_mean(mean, )
        self.generator = generator if generator is not None else _DEFAULT_GENERATOR

    def _populate_mean(
        self,
        mean: _np.ndarray | Real | None,
    ) -> None:
        r"""
        """
        shape = (self.num_variables, 1, )
        if mean is None:
            mean = 0
        if isinstance(mean, Real):
            mean = _np.full(shape, mean, dtype=float, )
        self.mean = mean.reshape(shape, order="F", )

    @property
    def num_variables(self, /, ) -> int:
        r"""
        """
        return self.factor.shape[0]

    def simulate(
        self,
        num_draws: int,
    ) -> _np.ndarray:
        r"""
        """
        shape = self.num_variables, num_draws,
        x = self.factor @ self.generator(shape, )
        if self.mean is not None:
            x += self.mean
        return x

    #]

