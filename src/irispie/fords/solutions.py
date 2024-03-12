r"""
# First-order solution matrices


## Square solution:

$$
\begin{gathered}
\xi_t = T \, \xi_{t-1} + P \, u_t + R \, v_t + K
\\
y_t = Z \, \xi_t + H \, w_t + D
\end{gathered}
$$


## Equivalent block-triangular solution:

$$
\begin{gathered}
\alpha_t = T_\alpha \, \alpha_{t-1} + P_\alpha \, u_t + R_\alpha \, v_t + K_\alpha
\\
y_t = Z_\alpha \, \alpha_t + D + H \, w_t
\\
\xi_t \equiv = U_alpha \, \alpha_t
\end{gathered}
$$


## Forward expansion:

$$
\cdots
$$
"""


#[
from __future__ import annotations

from typing import (Self, Callable, )
from numbers import (Real, Number, )
import dataclasses as _dc

import enum as _en
import numpy as _np
import scipy as _sp

from ..fords import systems as _systems
from ..fords import descriptors as _descriptors
#]


class EigenValueKind(_en.Flag):
    STABLE = _en.auto()
    UNIT = _en.auto()
    UNSTABLE = _en.auto()


class SystemStabilityKind(_en.Flag):
    STABLE = _en.auto()
    MULTIPLE_STABLE = _en.auto()
    NO_STABLE = _en.auto()


class VariableStability(_en.Flag):
    STABLE = _en.auto()
    UNIT_ROOT = _en.auto()


@_dc.dataclass(slots=True, )
class Solution:
    """
    ## Square solution:

    T: Transition matrix
    P: Impact matrix of unanticipated shocks
    R: Impact matrix of anticipated shocks
    K: Intercept in transition equation
    Z: Measurement matrix
    H: Impact matrix of measurement shocks
    D: Intercept in measurement equation


    ## Equivalent block-triangular solution:

    Ta: Transition matrix in triangular system
    Pa: Impact matrix of unanticipated shocks in triangular system
    Ra: Impact matrix of anticipated shocks in triangular system
    Ka: Intercept in transition equation in triangular system
    Za: Measurement matrix in triangular system
    Ua: Rotation matrix from triangular to square system


    ## Forward expansion of square solution:

    J: Power matrix
    Ru: Forward-looking impact matrix of transition shocks
    X: Impact matrix in square system
    Xa: Impact matrix in triangular system
    """
    #[
    T: _np.ndarray | None = None
    P: _np.ndarray | None = None
    R: _np.ndarray | None = None
    K: _np.ndarray | None = None
    Z: _np.ndarray | None = None
    H: _np.ndarray | None = None
    D: _np.ndarray | None = None

    Ta: _np.ndarray | None = None
    Pa: _np.ndarray | None = None
    Ra: _np.ndarray | None = None
    Ka: _np.ndarray | None = None
    Za: _np.ndarray | None = None
    Ua: _np.ndarray | None = None

    J: _np.ndarray | None = None
    Ru: _np.ndarray | None = None
    X: _np.ndarray | None = None
    Xa: _np.ndarray | None = None

    eigenvalues: tuple[Number, ...] | None = None
    eigenvalues_stability: tuple[EigenValueKind, ...] | None = None
    system_stability: SystemStabilityKind | None = None
    transition_vector_stability: VariableStability | None = None
    measurement_vector_stability: VariableStability | None = None

    def __init__(
        self, 
        descriptor: _descriptors.Descriptor,
        system: _systems.System,
        /,
        *,
        tolerance: float = 1e-12,
        clip_small: bool = False,
    ) -> Self:
        """
        """
        def is_alpha_beta_stable_or_unit_root(alpha: Real, beta: Real, /, ) -> bool:
            return abs(beta) < (1 + tolerance)*abs(alpha)
        def is_stable_root(root: Real, /, ) -> bool:
            return abs(root) < (1 - tolerance)
        def is_unit_root(root: Real, /, ) -> bool:
            return abs(root) >= (1 - tolerance) and abs(root) < (1 + tolerance)
        def clip_func(x: _np.ndarray, /, ) -> _np.ndarray:
            return _np.where(_np.abs(x) < tolerance, 0, x)
        clip = clip_func if clip_small else None
        #
        # Detach unstable from (stable + unit) roots and solve out expectations
        # The system is triangular but because stable and unit roots are
        # not detached yet, the system is called "preliminary"
        qz, self.eigenvalues = \
            _solve_ordqz(system, is_alpha_beta_stable_or_unit_root, is_stable_root, is_unit_root, )
        triangular_solution_prelim = \
            _solve_transition_equations(descriptor, system, qz, )
        #
        # Detach unit from stable roots to create the final triangular solution
        # From the final triangular solution, calculate the square solution
        triangular_solution = detach_stable_from_unit_roots(
            triangular_solution_prelim,
            is_unit_root,
            clip=clip,
        )
        #
        square_solution = _square_from_triangular(triangular_solution, )
        #
        self.Ua, self.Ta, self.Pa, self.Ra, self.Ka, self.Xa, self.J, self.Ru = triangular_solution
        self.T, self.P, self.R, self.K, self.X = square_solution
        #
        # Solve measurement equations
        self.Z, self.H, self.D, self.Za = _solve_measurement_equations(
            descriptor,
            system,
            self.Ua,
            clip=clip,
        )
        self._classify_eigenvalues_stability(is_stable_root, is_unit_root, )
        self._classify_system_stability(descriptor.get_num_forwards(), )
        self._classify_transition_vector_stability(tolerance=tolerance, )
        self._classify_measurement_vector_stability(tolerance=tolerance, )

    @property
    def num_alpha(self, /, ) -> int:
        """
        Number of alpha vector elements
        """
        return self.T.shape[0]

    @property
    def num_y(self, /, ) -> int:
        """
        Number of y vector elements
        """
        return self.Z.shape[0]

    @property
    def num_u(self, /, ) -> int:
        """
        Number of v vector elements
        """
        return self.P.shape[1]

    @property
    def num_v(self, /, ) -> int:
        """
        Number of v vector elements
        """
        return self.R.shape[1]

    @property
    def num_w(self, /, ) -> int:
        """
        Number of w vector elements
        """
        return self.H.shape[1]

    @property
    def num_unit_roots(self, /, ) -> int:
        """
        Number of unit roots
        """
        return self.eigenvalues_stability.count(EigenValueKind.UNIT)

    @property
    def num_stable(self, /, ) -> int:
        """
        Number of stable elements in alpha vector
        """
        return self.num_alpha - self.num_unit_roots

    @property
    def Ta_stable(self, /, ) -> _np.ndarray:
        """
        Stable part of transition matrix
        """
        num_unit_roots = self.num_unit_roots
        return self.Ta[num_unit_roots:, num_unit_roots:]

    @property
    def Pa_stable(self, /, ) -> _np.ndarray:
        """
        Stable part of impact matrix of transition shocks
        """
        num_unit_roots = self.num_unit_roots
        return self.Pa[num_unit_roots:, :]

    @property
    def Ra_stable(self, /, ) -> _np.ndarray:
        """
        Stable part of impact matrix of transition shocks
        """
        num_unit_roots = self.num_unit_roots
        return self.Ra[num_unit_roots:, :]

    @property
    def Ka_stable(self, /, ) -> _np.ndarray:
        """
        Stable part of intercept in transition equation
        """
        num_unit_roots = self.num_unit_roots
        return self.Ka[num_unit_roots:]

    @property
    def Za_stable(self, /, ) -> _np.ndarray:
        """
        Stable part of measurement matrix
        """
        num_unit_roots = self.num_unit_roots
        return self.Za[:, num_unit_roots:]

    @property
    def boolex_stable_transition_vector(self, /, ) -> tuple[int, ...]:
        """
        Index of stable transition vector elements
        """
        return _np.array(tuple(
            i == VariableStability.STABLE
            for i in self.transition_vector_stability
        ), dtype=bool, )

    @property
    def boolex_stable_measurement_vector(self, /, ) -> tuple[int, ...]:
        """
        Index of stable measurement vector elements
        """
        return _np.array(tuple(
            i == VariableStability.STABLE
            for i in self.measurement_vector_stability
        ), dtype=bool, )

    def expand_square_solution(self, forward, /, ) -> list[_np.ndarray]:
        """
        Expand R matrices of square solution for t+1...t+forward
        """
        R, X, J, Ru = self.R, self.X, self.J, self.Ru
        if (R is None) or (X is None) or (J is None) or (Ru is None):
            return None
        #
        # return [R(t+1), R(t+2), ..., R(t+forward)]
        #
        # R(t+k) = -X J**(k-1) Ru e(t+k)
        # k = 1, ..., forward or k-1 = 0, ..., forward-1
        #
        return [
            -X @ _np.linalg.matrix_power(J, k_minus_1) @ Ru
            for k_minus_1 in range(0, forward)
        ]

    def _classify_eigenvalues_stability(
        self,
        is_stable_root: Callable[[Real], bool],
        is_unit_root: Callable[[Real], bool],
        /,
    ) -> None:
        self.eigenvalues_stability = tuple(
            _classify_eigenvalue_stability(v, is_stable_root, is_unit_root, )
            for v in self.eigenvalues
        )

    def _classify_system_stability(
        self,
        num_forwards: int,
        /,
    ) -> None:
        num_unstable = self.eigenvalues_stability.count(EigenValueKind.UNSTABLE)
        if num_unstable == num_forwards:
            self.system_stability = SystemStabilityKind.STABLE
        elif num_unstable > num_forwards:
            self.system_stability = SystemStabilityKind.NO_STABLE
        else:
            self.system_stability = SystemStabilityKind.MULTIPLE_STABLE

    def _classify_transition_vector_stability(
        self,
        /,
        tolerance: float = 1e-12,
    ) -> None:
        self.transition_vector_stability \
            = _classify_solution_vector_stability(
                self.Ua,
                self.num_unit_roots,
                tolerance=tolerance,
            )

    def _classify_measurement_vector_stability(
        self,
        /,
        tolerance: float = 1e-12,
    ) -> None:
        self.measurement_vector_stability \
            = _classify_solution_vector_stability(
                self.Za,
                self.num_unit_roots,
                tolerance=tolerance,
            )
    #]


def left_div(A, B):
    r"""
    Solve A \ B = pinv(A) @ B or inv(A) @ B
    """
    return _np.linalg.lstsq(A, B, rcond=None)[0]


def right_div(B, A):
    r"""
    Solve B / A which is (A' \ B')'
    """
    return _np.linalg.lstsq(A.T, B.T, rcond=None)[0].T


def _square_from_triangular(
    triangular_solution: tuple[_np.ndarray, ...],
    /,
) -> tuple[_np.ndarray, ...]:
    r"""
    T <- Ua @ Ta / Ua
    P <- Ua @ Pa
    R <- Ua @ Ra
    X <- Xa @ Ra
    K <- Ua @ Ka
    xi[t] = ... -X J**(k-1) Ru e[t+k]
    """
    #[
    Ua, Ta, Pa, Ra, Ka, Xa, *_ = triangular_solution
    T = Ua @ right_div(Ta, Ua) # Ua @ (Ta / Ua)
    P = Ua @ Pa
    R = Ua @ Ra
    K = Ua @ Ka
    X = Ua @ Xa
    return T, P, R, K, X
    #]


def detach_stable_from_unit_roots(
    transition_solution_prelim: tuple[_np.ndarray, ...],
    is_unit_root: Callable[[Real], bool],
    /,
    clip: Callable | None,
) -> tuple[_np.ndarray, ...]:
    """
    Apply a secondary Schur decomposition to detach stable eigenvalues from unit roots
    """
    #[
    Ug, Tg, Pg, Rg, Kg, Xg, J, Ru = transition_solution_prelim
    num_xib = Tg.shape[0]
    Ta, u, check_num_unit_roots = _sp.linalg.schur(Tg, sort=is_unit_root, ) # Tg = u @ Ta @ u.T
    Ua = Ug @ u if Ug is not None else u
    Ua = clip(Ua) if clip is not None else Ua
    Pa = u.T @ Pg
    Ra = u.T @ Rg
    Ka = u.T @ Kg
    Xa = u.T @ Xg if Xg is not None else None
    return Ua, Ta, Pa, Ra, Ka, Xa, J, Ru
    #]


def _solve_measurement_equations(
    descriptor,
    system,
    Ua,
    /,
    clip: Callable | None,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    num_forwards = descriptor.get_num_forwards()
    G = system.G[:, num_forwards:]
    Z = left_div(-system.F, G) # -F \ G
    H = left_div(-system.F, system.J) # -F \ J
    D = left_div(-system.F, system.H) # -F \ H
    Z = clip(Z) if clip is not None else Z
    Za = Z @ Ua
    return Z, H, D, Za
    #]


def _solve_transition_equations(
    descriptor,
    system,
    qz,
    /,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    num_backwards = descriptor.get_num_backwards()
    num_forwards = descriptor.get_num_forwards()
    num_stable = num_backwards
    S, T, Q, Z = qz
    #
    S11 = S[:num_stable, :num_stable]
    S12 = S[:num_stable, num_stable:]
    S22 = S[num_stable:, num_stable:]
    #
    T11 = T[:num_stable, :num_stable]
    T12 = T[:num_stable, num_stable:]
    T22 = T[num_stable:, num_stable:]
    #
    Z21 = Z[num_forwards:, :num_stable]
    Z22 = Z[num_forwards:, num_stable:]
    #
    # Constant in transition
    Q_CC = Q @ system.C
    Q_CC1 = Q_CC[:num_stable]
    Q_CC2 = Q_CC[num_stable:]
    #
    # Unanticipated shocks in transition
    Q_DD = Q @ system.D
    Q_DD1 = Q_DD[:num_stable, :]
    Q_DD2 = Q_DD[num_stable:, :]
    #
    # Antiicipated shocks in transition
    Q_EE = Q @ system.E
    Q_EE1 = Q_EE[:num_stable, :]
    Q_EE2 = Q_EE[num_stable:, :]
    #
    # Unstable block
    #
    G = left_div(-Z21, Z22) # -Z21 \ Z22
    Pu = left_div(-T22, Q_DD2) # -T22 \ Q_DD2
    Ru = left_div(-T22, Q_EE2) # -T22 \ Q_EE2
    Ku = left_div(-(S22 + T22), Q_CC2) # -(S22+T22) \ Q_CC2
    #
    # Transform stable block==transform backward-looking variables:
    # gamma(t) = s(t) + G u(t+1)
    #
    Xg0 = left_div(S11, T11 @ G + T12)
    Xg1 = G + left_div(S11, S12)
    #
    Tg = left_div(-S11, T11)
    Pg = -Xg0 @ Pu - left_div(S11, Q_DD1)
    Rg = -Xg0 @ Ru - left_div(S11, Q_EE1)
    Kg = -(Xg0 + Xg1) @ Ku - left_div(S11, Q_CC1)
    Ug = Z21 # xib = Ug @ gamma
    #
    # Forward expansion
    # gamma(t) = ... -Xg J**(k-1) Ru e(t+k)
    #
    J = left_div(-T22, S22) # -T22 \ S22
    Xg = Xg1 + Xg0 @ J
    #
    return Ug, Tg, Pg, Rg, Kg, Xg, J, Ru
    #]


def _solve_ordqz(
    system,
    is_alpha_beta_stable_or_unit_root,
    is_stable_root,
    is_unit_root,
    /,
) -> tuple[tuple[_np.ndarray, ...], tuple[Real, ], ]:
    """
    """
    #[
    S, T, alpha, beta, Q, Z = _sp.linalg.ordqz(system.A, system.B, sort=is_alpha_beta_stable_or_unit_root, )
    Q = Q.T
    #
    inx_nonzero_alpha = alpha != 0
    eigenvalues = _np.full(beta.shape, _np.inf, dtype=complex, )
    eigenvalues[inx_nonzero_alpha] = -beta[inx_nonzero_alpha] / alpha[inx_nonzero_alpha]
    eigenvalues = tuple(eigenvalues)
    #
    #
    return (S, T, Q, Z), eigenvalues
    #]


def _classify_eigenvalue_stability(
    eigenvalue,
    is_stable_root,
    is_unit_root,
) -> EigenValueKind:
    #[
    abs_eigenvalue = _np.abs(eigenvalue)
    if is_stable_root(abs_eigenvalue):
        return EigenValueKind.STABLE
    elif is_unit_root(abs_eigenvalue):
        return EigenValueKind.UNIT
    else:
        return EigenValueKind.UNSTABLE
    #]


def _classify_solution_vector_stability(
    transform_matrix: _np.ndarray,
    num_unit_roots: int,
    /,
    tolerance: float = 1e-12,
) -> None:
    """
    """
    #[
    test_matrix = _np.abs(transform_matrix[:, :num_unit_roots], )
    index = _np.any(test_matrix > tolerance, axis=1, )
    return tuple(
        VariableStability.UNIT_ROOT if i
        else VariableStability.STABLE
        for i in index
    )
    #]

