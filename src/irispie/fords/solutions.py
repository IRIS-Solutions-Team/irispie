"""
# First-order solution matrices
"""


#[
from __future__ import annotations

from typing import (Self, Callable, )
from numbers import (Number, )
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
    T: _np.ndarray | None = None
    R: _np.ndarray | None = None
    K: _np.ndarray | None = None
    Z: _np.ndarray | None = None
    H: _np.ndarray | None = None
    D: _np.ndarray | None = None

    Ta: _np.ndarray | None = None
    Ra: _np.ndarray | None = None
    Ka: _np.ndarray | None = None
    Za: _np.ndarray | None = None
    Ua: _np.ndarray | None = None

    J: _np.ndarray | None = None
    Ru: _np.ndarray | None = None
    X: _np.ndarray | None = None
    Xa: _np.ndarray | None = None

    eigen_values: tuple[Number, ...] | None = None
    eigen_values_stability: tuple[EigenValueKind, ...] | None = None
    system_stability: SystemStabilityKind | None = None

    def __init__(
        self, 
        descriptor: _descriptors.Descriptor,
        system: _systems.System,
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
        # The system is triangular but because stable and unit roots are
        # not detached yet, the system is called "preliminary"
        qz, eigen_values, eigen_values_stability = _solve_ordqz(system, is_alpha_beta_stable_or_unit_root, is_stable_root, is_unit_root, )
        system_stability = _classify_system_stability(descriptor, eigen_values_stability, )
        triangular_solution_prelim = _solve_transition_equations(descriptor, system, qz, )
        #
        # Detach unit from stable roots to create the final triangular solution
        # From the final triangular solution, calculate the square solution
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
    #]


def _left_div(A, B):
    """
    Solve A \ B
    """
    return _np.linalg.lstsq(A, B, rcond=None)[0]


def _right_div(B, A):
    """
    Solve B/A = (A'\B')'
    """
    return _np.linalg.lstsq(A.T, B.T, rcond=None)[0].T


def _square_from_triangular(
    triangular_solution: tuple[_np.ndarray, ...],
    /,
) -> tuple[_np.ndarray, ...]:
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
    transition_solution_prelim: tuple[_np.ndarray, ...],
    is_unit_root: Callable[[Number], bool],
    /,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    Ug, Tg, Rg, Kg, Xg, J, Ru = transition_solution_prelim
    num_xib = Tg.shape[0]
    Ta, u, check_num_unit_roots = _sp.linalg.schur(Tg, sort=is_unit_root, ) # Tg = u @ Ta @ u.T
    Ua = Ug @ u if Ug is not None else u
    Ra = u.T @ Rg
    Ka = u.T @ Kg
    Xa = u.T @ Xg if Xg is not None else None
    return Ua, Ta, Ra, Ka, Xa, J, Ru
    #]


def _solve_measurement_equations(descriptor, system, Ua, ) -> tuple[_np.ndarray, ...]:
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


def _solve_transition_equations(descriptor, system, qz, ) -> tuple[_np.ndarray, ...]:
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
) -> tuple[tuple[_np.ndarray, ...], tuple[Number, ], tuple[EigenValueKind, ...]]:
    """
    """
    #[
    S, T, alpha, beta, Q, Z = _sp.linalg.ordqz(system.A, system.B, sort=is_alpha_beta_stable_or_unit_root, )
    Q = Q.T
    #
    inx_nonzero_alpha = alpha != 0
    eigen_values = _np.full(beta.shape, _np.inf, dtype=complex, )
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
    abs_eig_value = _np.abs(eig_value)
    match (is_stable_root(abs_eig_value), is_unit_root(abs_eig_value), ):
        case (True, _, ):
            return EigenValueKind.STABLE
        case (_, True, ):
            return EigenValueKind.UNIT
        case _:
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


