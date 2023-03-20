"""
"""


#[
from __future__ import annotations
from IPython import embed
import enum as en_
import numpy as np_
import scipy as sp_

from ..fords import systems as sy_
from ..fords import descriptors as de_
#]


class EigenValueKind(en_.Flag):
    STABLE = en_.auto()
    UNIT = en_.auto()
    UNSTABLE = en_.auto()


class StabilityKind(en_.Flag):
    STABLE = en_.auto()
    MULTIPLE_STABLE = en_.auto()
    NO_STABLE = en_.auto()


class Solution:
    """
    """
    #[
    @classmethod
    def for_model(
        cls, 
        descriptor: de_.Descriptor,
        system: sy_.System,
        /,
        tolerance: float = 1e-12,
    ) -> Self:
        self = cls()
        #
        qz = _solve_ordqz(system, tolerance, )
        stability = _classify_system_stability(descriptor, qz, tolerance, )

        triangular_solution_prelim = _solve_transition_equations(descriptor, system, qz, )
        #traingula_solution = _separate_unit_roots(

        U = triangular_solution_prelim[0]
        measurement_solution = _solve_measurement_equations(descriptor, system, U)
        square_solution = _square_from_triangular(triangular_solution_prelim)

        self.U, self.Ta, self.Ra, self.Ka, self.Xa, self.J, self.Ru = triangular_solution_prelim
        self.Tb, self.Rb, self.Kb, self.Xb = square_solution
        self.Za, self.Zb, self.H, self.D = measurement_solution
    #]


def _left_div(A, B):
    """
    Solve A\B
    """
    return np_.linalg.lstsq(A, B, rcond=None)[0]


def _right_div(B, A):
    """
    Solve B/A = (A'\B')'
    """
    return np_.linalg.lstsq(A.T, B.T, rcond=None)[0].T


def _square_from_triangular(triangular_solution):
    U, Ta, Ra, Ka, Xa, J, Ru = triangular_solution
    # T <- U*T/U note that T/U == (U'\T')'
    # R <- U*R;
    # K <- U*K;
    #
    # a(t) = ... -Xa J**(k-1) Ru e(t+k)
    Tb = U @ _right_div(Ta, U)
    Rb = U @ Ra
    Kb = U @ Ka
    Xb = U @ Xa
    #
    return Tb, Rb, Kb, Xb


def _detach_stable_from_unit_roots(T, R, k, Z, H, d, U):
    num_xib = T.shape[0]
    U = U if U else n


def _solve_measurement_equations(descriptor, system, U, ):
    num_forwards = descriptor.get_num_forwards()
    F, G, H, J = system.F, system.G, system.H, system.J
    #
    Gb = G[:, num_forwards:]
    Zb = _left_div(-F, Gb)
    Hb = _left_div(-F, J)
    Db = _left_div(-F, H)
    Za = Zb @ U
    #
    return Za, Zb, Hb, Db


def _solve_transition_equations(descriptor, system, qz, ):
    """
    """
    num_backwards = descriptor.get_num_backwards()
    num_forwards = descriptor.get_num_forwards()
    num_stable = num_backwards
    S, T, Q, Z, eig_values, eig_stability = qz
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
    G = _left_div(-Z21, Z22)
    Ru = _left_div(-T22, QD2)
    Ku = _left_div(-(S22 + T22), QC2)
    #
    # Transform stable block==transform backward-looking variables:
    # alpha(t) = s(t) + G u(t+1)
    #
    Xa0 = _left_div(S11, T11 @ G + T12)
    Xa1 = G + _left_div(S11, S12)
    #
    Ta = _left_div(-S11, T11)
    Ra = -Xa0 @ Ru - _left_div(S11, QD1)
    Ka = -(Xa0 + Xa1) @ Ku - _left_div(S11, QC1)
    U = Z21
    #
    # Forward expansion
    # a(t) = ... -Xa J**(k-1) Ru e(t+k)
    #
    J = _left_div(-T22, S22)
    Xa = Xa1 + Xa0 @ J
    #
    return U, Ta, Ra, Ka, Xa, J, Ru


def _solve_ordqz(system, tolerance, ):
    _stable_and_unit_first = lambda alpha, beta: abs(beta) < (1 + tolerance)*abs(alpha)
    S, T, alpha, beta, Q, Z = sp_.linalg.ordqz(system.A, system.B, sort=_stable_and_unit_first, )
    Q = Q.T
    #
    inx_nonzero_alpha = alpha != 0
    eig_values = np_.full(beta.shape, np_.inf, dtype=complex, )
    eig_values[inx_nonzero_alpha] = -beta[inx_nonzero_alpha] / alpha[inx_nonzero_alpha]
    #
    eig_stability = tuple(_classify_eig_value_stability(v, tolerance) for v in eig_values)
    #
    return S, T, Q, Z, eig_values, eig_stability


def _classify_eig_value_stability(eig_value, tolerance, ) -> EigenValueKind:
    abs_eig_value = np_.abs(eig_value)
    if abs_eig_value < 1 - tolerance:
        return EigenValueKind.STABLE
    elif abs_eig_value < 1 + tolerance:
        return EigenValueKind.UNIT
    else:
        return EigenValueKind.UNSTABLE


def _classify_system_stability(descriptor, qz, tolerance, ):
    *_, eig_stability = qz
    num_unstable = sum(1 for s in eig_stability if s==EigenValueKind.UNSTABLE)
    num_forwards = descriptor.get_num_forwards()
    if num_unstable == num_forwards:
        stability = StabilityKind.STABLE
    elif num_unstable > num_forwards:
        stability = StabilityKind.NO_STABLE
    else:
        stability = StabilityKind.MULTIPLE_STABLE
    return stability


