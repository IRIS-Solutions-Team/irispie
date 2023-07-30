"""
# First-order solution matrices
"""


#[
import enum as en_
import numpy as np_
import scipy as sp_
from typing import (Self, NoReturn, Callable, )
from numbers import (Number, )
import dataclasses as _dc

from ..fords import (systems as _fy, descriptors as _fd, )
from ..models import (flags as _mg, )
#]


class EigenValueKind(en_.Flag):
    STABLE = en_.auto()
    UNIT = en_.auto()
    UNSTABLE = en_.auto()


class SystemStabilityKind(en_.Flag):
    STABLE = en_.auto()
    MULTIPLE_STABLE = en_.auto()
    NO_STABLE = en_.auto()


@_dc.dataclass(slots=True, )
class Solution:
    """
    ## Square solution:
    T: Transition matrix
    R: Impact matrix of transition shocks
    K: Intercept in transition equation
    Z: Measurement matrix
    H: Impact matrix of measurement shocks
    D: Intercept in measurement equation

    ## Triangular solution:
    Ta: Transition matrix in triangular system
    Ra: Impact matrix of transition shocks in triangular system
    Ka: Intercept in transition equation in triangular system
    Za: Measurement matrix in triangular system

    ## Forward expansion of square solution:
    J: Power matrix
    Ru: Forward-looking impact matrix of transition shocks
    X: Impact matrix in square system
    Xa: Impact matrix in triangular system
    """
    #[
    T: np_.ndarray | None = None
    R: np_.ndarray | None = None
    K: np_.ndarray | None = None
    Z: np_.ndarray | None = None
    H: np_.ndarray | None = None
    D: np_.ndarray | None = None

    Ta: np_.ndarray | None = None
    Ra: np_.ndarray | None = None
    Ka: np_.ndarray | None = None
    Za: np_.ndarray | None = None
    Ua: np_.ndarray | None = None

    J: np_.ndarray | None = None
    Ru: np_.ndarray | None = None
    X: np_.ndarray | None = None
    Xa: np_.ndarray | None = None

    eigen_values: tuple[Number, ...] | None = None
    eigen_values_stability: tuple[EigenValueKind, ...] | None = None
    system_stability: SystemStabilityKind | None = None

    def __init__(
        self, 
        descriptor: _fd.Descriptor,
        system: _fy.System,
        model_flags: _mg.Flags,
        /,
        *,
        tolerance: float = 1e-12,
    ) -> Self:
        """
        """
        is_alpha_beta_stable_or_unit_root = lambda alpha, beta: abs(beta) < (1 + tolerance)*abs(alpha)
        is_stable_root = lambda root: abs(root) < (1 - tolerance)
        is_unit_root = lambda root: abs(root) >= (1 - tolerance) and abs(root) < (1 + tolerance)
        #
        # Detach unstable from (stable + unit) roots and solve out expectations
        qz, eigen_values, eigen_values_stability = _solve_ordqz(system, is_alpha_beta_stable_or_unit_root, is_stable_root, is_unit_root, )
        system_stability = _classify_system_stability(descriptor, eigen_values_stability, )
        triangular_solution_prelim = _solve_transition_equations(descriptor, system, qz, )
        #
        # Detach unit from stable roots and transform to square form
        triangular_solution = detach_stable_from_unit_roots(triangular_solution_prelim, is_unit_root, )
        square_solution = _square_from_triangular(triangular_solution, )
        self.Ua, self.Ta, self.Ra, self.Ka, self.Xa, self.J, self.Ru = triangular_solution
        self.T, self.R, self.K, self.X = square_solution
        #
        # Solve measurement equations
        self.Z, self.H, self.D, self.Za = _solve_measurement_equations(descriptor, system, self.Ua, )
        self.eigen_values, self.eigen_values_stability = eigen_values, eigen_values_stability
        self.system_stability = system_stability
        #
        return self

    def expand_square_solution(self, forward, /, ) -> list[np_.ndarray]:
        """
        Expand R matrices of square solution for t+1...t+forward
        """
        R, X, J, Ru = self.R, self.X, self.J, self.Ru
        if (R is None) or (X is None) or (J is None) or (Ru is None):
            return None
        Jk = np_.eye(J.shape[0])
        #
        # return [R(t+1), R(t+2), ..., R(t+forward)]
        #
        # R(t+k) = -X J**(k-1) Ru e(t+k)
        # k = 1, ..., forward or k-1 = 0, ..., forward-1
        #
        return [
            -X @ np_.linalg.matrix_power(J, k_minus_1) @ Ru 
            for k_minus_1 in range(0, forward)
        ]
    #]


def _left_div(A, B):
    """
    Solve A \ B
    """
    return np_.linalg.lstsq(A, B, rcond=None)[0]


def _right_div(B, A):
    """
    Solve B/A = (A'\B')'
    """
    return np_.linalg.lstsq(A.T, B.T, rcond=None)[0].T


def _square_from_triangular(
    triangular_solution: tuple[np_.ndarray, ...],
    /,
) -> tuple[np_.ndarray, ...]:
    """
    T <- Ua @ Ta/U; note that Ta/U == (U'\Ta')'
    R <- Ua @ Ra;
    X <- Xa @ Ra;
    K <- Ua @ Ka;
    xi(t) = ... -X J**(k-1) Ru e(t+k)
    """
    #[
    Ua, Ta, Ra, Ka, Xa, *_ = triangular_solution
    T = Ua @ _right_div(Ta, Ua) # Ua @ Ta / Ua
    R = Ua @ Ra
    K = Ua @ Ka
    X = Ua @ Xa
    return T, R, K, X
    #]


def detach_stable_from_unit_roots(
    transition_solution_prelim: tuple[np_.ndarray, ...],
    is_unit_root: Callable[[Number], bool],
    /,
) -> tuple[np_.ndarray, ...]:
    """
    """
    #[
    Ug, Tg, Rg, Kg, Xg, J, Ru = transition_solution_prelim
    num_xib = Tg.shape[0]
    Ta, u, check_num_unit_roots = sp_.linalg.schur(Tg, sort=is_unit_root, ) # Tg = u @ Ta @ u.T
    Ua = Ug @ u if Ug is not None else u
    Ra = u.T @ Rg
    Ka = u.T @ Kg
    Xa = u.T @ Xg if Xg is not None else None
    return Ua, Ta, Ra, Ka, Xa, J, Ru
    #]


def _solve_measurement_equations(descriptor, system, Ua, ) -> tuple[np_.ndarray, ...]:
    """
    """
    #[
    num_forwards = descriptor.get_num_forwards()
    G = system.G[:, num_forwards:]
    Z = _left_div(-system.F, G) # -F \ G
    H = _left_div(-system.F, system.J) # -F \ J
    D = _left_div(-system.F, system.H) # -F \ H
    Za = Z @ Ua
    return Z, H, D, Za
    #]


def _solve_transition_equations(descriptor, system, qz, ) -> tuple[np_.ndarray, ...]:
    """
    """
    #[
    num_backwards = descriptor.get_num_backwards()
    num_forwards = descriptor.get_num_forwards()
    num_stable = num_backwards
    S, T, Q, Z = qz
    A, B, C, D = system.A, system.B, system.C, system.D
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
    QC = Q @ C
    QC1 = QC[:num_stable, ...]
    QC2 = QC[num_stable:, ...]
    #
    QD = Q @ D
    QD1 = QD[:num_stable, ...]
    QD2 = QD[num_stable:, ...]
    #
    # Unstable block
    #
    G = _left_div(-Z21, Z22) # -Z21 \ Z22
    Ru = _left_div(-T22, QD2) # -T22 \ QD2
    Ku = _left_div(-(S22 + T22), QC2) # -(S22+T22) \ QC2
    #
    # Transform stable block==transform backward-looking variables:
    # gamma(t) = s(t) + G u(t+1)
    #
    Xg0 = _left_div(S11, T11 @ G + T12)
    Xg1 = G + _left_div(S11, S12)
    #
    Tg = _left_div(-S11, T11)
    Rg = -Xg0 @ Ru - _left_div(S11, QD1)
    Kg = -(Xg0 + Xg1) @ Ku - _left_div(S11, QC1)
    Ug = Z21 # xib = Ug @ gamma
    #
    # Forward expansion
    # gamma(t) = ... -Xg J**(k-1) Ru e(t+k)
    #
    J = _left_div(-T22, S22) # -T22 \ S22
    Xg = Xg1 + Xg0 @ J
    #
    return Ug, Tg, Rg, Kg, Xg, J, Ru
    #]


def _solve_ordqz(
    system,
    is_alpha_beta_stable_or_unit_root,
    is_stable_root,
    is_unit_root,
    /,
) -> tuple[tuple[np_.ndarray, ...], tuple[Number, ], tuple[EigenValueKind, ...]]:
    """
    """
    #[
    S, T, alpha, beta, Q, Z = sp_.linalg.ordqz(system.A, system.B, sort=is_alpha_beta_stable_or_unit_root, )
    Q = Q.T
    #
    inx_nonzero_alpha = alpha != 0
    eigen_values = np_.full(beta.shape, np_.inf, dtype=complex, )
    eigen_values[inx_nonzero_alpha] = -beta[inx_nonzero_alpha] / alpha[inx_nonzero_alpha]
    eigen_values = tuple(eigen_values)
    #
    eigen_values_stability = tuple(
        _classify_eig_value_stability(v, is_stable_root, is_unit_root, )
        for v in eigen_values
    )
    #
    return (S, T, Q, Z), eigen_values, eigen_values_stability
    #]


def _classify_eig_value_stability(eig_value, is_stable_root, is_unit_root, ) -> EigenValueKind:
    #[
    abs_eig_value = np_.abs(eig_value)
    if is_stable_root(abs_eig_value):
        return EigenValueKind.STABLE
    elif is_unit_root(abs_eig_value):
        return EigenValueKind.UNIT
    else:
        return EigenValueKind.UNSTABLE
    #]


def _classify_system_stability(descriptor, eigen_values_stability, ):
    #[
    num_unstable = sum(1 for s in eigen_values_stability if s==EigenValueKind.UNSTABLE)
    num_forwards = descriptor.get_num_forwards()
    if num_unstable == num_forwards:
        stability = SystemStabilityKind.STABLE
    elif num_unstable > num_forwards:
        stability = SystemStabilityKind.NO_STABLE
    else:
        stability = SystemStabilityKind.MULTIPLE_STABLE
    return stability
    #]


