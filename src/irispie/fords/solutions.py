r"""
# First-order solution matrices


## Square solution:

$$
\begin{gathered}
\xi_t = T \, \xi_{t-1} + P \, u_t + \sum \, R \, v_t + K
\\
y_t = Z \, \xi_t + H \, w_t + D
\end{gathered}
$$


## Equivalent block-triangular solution:

$$
\begin{gathered}
\alpha_t = T_\alpha \, \alpha_{t-1} + P_\alpha \, u_t + \sum \, R_\alpha \, v_t + K_\alpha
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

import enum as _en
import numpy as _np
import scipy as _sp
import copy as _co
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self, Callable, Iterable
    from numbers import Real
    from ..fords import descriptors as _descriptors
    from ..fords import systems as _systems

#]


__all__ = (
    "STABLE", "UNIT_ROOT", "UNSTABLE",
)


class QzReorderingException(Exception):
    r"""
    Reordering of the QZ decomposition failed because some eigenvalues are too
    close to swap
    """
    pass


class ElementStability(_en.Flag, ):
    STABLE = _en.auto()
    UNIT_ROOT = _en.auto()
    UNSTABLE = _en.auto()
    ALL = STABLE | UNIT_ROOT | UNSTABLE


class SystemStabilityVerdict(_en.Enum, ):
    r"""
    Outcome of solving the first-order system, as five mutually exclusive
    verdicts

    UNIQUE_STABLE: both Blanchard-Kahn conditions hold
    MULTIPLE_STABLE: fewer unstable roots than forward-looking variables
    NO_STABLE: more unstable roots than forward-looking variables
    RANK_DEFICIENT: the root counts are right but the stable subspace fails to
    project onto the predetermined variables
    DECOMPOSITION_FAILED: the decomposition broke down before any eigenvalue
    could be calculated, so no Blanchard-Kahn condition could be evaluated
    """
    #[

    UNIQUE_STABLE = _en.auto()
    MULTIPLE_STABLE = _en.auto()
    NO_STABLE = _en.auto()
    RANK_DEFICIENT = _en.auto()
    DECOMPOSITION_FAILED = _en.auto()

    @property
    def message_template(self, ) -> str:
        return _SYSTEM_STABILITY_VERDICT_MESSAGE_TEMPLATE[self]

    #]


_SYSTEM_STABILITY_VERDICT_MESSAGE_TEMPLATE = {
    SystemStabilityVerdict.UNIQUE_STABLE:
        "Unique stable solution",
    SystemStabilityVerdict.MULTIPLE_STABLE:
        "Multiple stable solutions: unstable roots {num_unstable}, "
        "forward-looking variables {num_forwards}",
    SystemStabilityVerdict.NO_STABLE:
        "No stable solution: unstable roots {num_unstable}, "
        "forward-looking variables {num_forwards}",
    SystemStabilityVerdict.RANK_DEFICIENT:
        "No unique stable solution: the stable subspace does not span the "
        "{num_backwards} predetermined variables, with condition number {rcond:.3g} < {rank_tolerance:.3g}",
    SystemStabilityVerdict.DECOMPOSITION_FAILED:
        "No solution calculated: the QZ decomposition could not be reordered "
        "because some eigenvalues are too close to separate, which usually "
        "indicates badly scaled or nearly singular system matrices",
}


@dataclass(frozen=True, slots=True, )
class Stability:
    r"""
    Diagnosis of the first-order system, as read off the QZ decomposition

    Instances come from `from_decomposition`, which evaluates the
    Blanchard-Kahn conditions and fills in every field consistently. The plain
    constructor performs no validation and is not meant to be called directly
    """
    #[

    eigenvalues: tuple[complex, ...] | None
    eigenvalue_stability: tuple[ElementStability, ...] | None
    num_forwards: int | None
    num_backwards: int | None
    rcond: float | None
    rank_tolerance: float | None
    system_stability: SystemStabilityVerdict

    @classmethod
    def from_decomposition(
        klass,
        *,
        eigenvalues: tuple[complex, ...],
        eigenvalue_stability: tuple[ElementStability, ...],
        num_forwards: int,
        num_backwards: int,
        qz_matrixes: tuple[_np.ndarray, ...],
        rank_tolerance: float,
    ) -> Self:
        r"""
        Evaluate the Blanchard-Kahn conditions and return a complete Stability

        The order condition requires the number of unstable roots to equal the
        number of forward-looking variables; too few means multiple stable
        solutions, too many means none.

        The rank condition requires the stable deflating subspace to project
        onto the predetermined variables without a loss of rank. It is
        evaluated, and its reciprocal condition number computed, only once the
        order condition holds, because only then do the leading num_backwards
        columns of Z span that subspace and Z21 mean what it is supposed to
        mean; otherwise rcond is left as None
        """
        rcond = None
        num_unstable = eigenvalue_stability.count(ElementStability.UNSTABLE, )
        #
        # Order test
        if num_unstable < num_forwards:
            system_stability = SystemStabilityVerdict.MULTIPLE_STABLE
        elif num_unstable > num_forwards:
            system_stability = SystemStabilityVerdict.NO_STABLE
        else:
            #
            # Rank test
            #
            # The predetermined variables decompose as xib = Z21 @ w1 + Z22 @
            # w2, and ruling out explosive paths pins down w2; the stable
            # coordinates w1 are therefore recoverable from xib for every
            # initial condition only if Z21 is invertible. Near singularity is
            # what matters rather than exact rank deficiency, because roundoff
            # in the decomposition keeps Z21 off the singular manifold even
            # when the model is structurally degenerate
            *_, Z = qz_matrixes
            rcond = _estimate_rcond(Z[num_forwards:, :num_backwards], )
            system_stability = (
                SystemStabilityVerdict.RANK_DEFICIENT if rcond < rank_tolerance
                else SystemStabilityVerdict.UNIQUE_STABLE
            )
        #
        return klass(
            eigenvalues=eigenvalues,
            eigenvalue_stability=eigenvalue_stability,
            num_forwards=num_forwards,
            num_backwards=num_backwards,
            rcond=rcond,
            rank_tolerance=rank_tolerance,
            system_stability=system_stability,
        )

    @classmethod
    def from_failed_decomposition(
        klass,
        *,
        num_forwards: int,
        num_backwards: int,
        rank_tolerance: float,
    ) -> Self:
        r"""
        Record that the decomposition broke down before any eigenvalue could be
        calculated

        Neither Blanchard-Kahn condition could be evaluated, so the eigenvalue
        tuples are empty and the derived counts all come out zero; the verdict
        is the only field carrying information
        """
        return klass(
            eigenvalues=None,
            eigenvalue_stability=None,
            num_forwards=num_forwards,
            num_backwards=num_backwards,
            rcond=None,
            rank_tolerance=rank_tolerance,
            system_stability=SystemStabilityVerdict.DECOMPOSITION_FAILED,
        )

    @property
    def num_stable(self, ) -> int:
        return (
            self.eigenvalue_stability.count(ElementStability.STABLE, )
            if self.eigenvalue_stability is not None else None
        )

    @property
    def num_unit_roots(self, ) -> int:
        return (
            self.eigenvalue_stability.count(ElementStability.UNIT_ROOT, )
            if self.eigenvalue_stability is not None else None
        )

    @property
    def num_unstable(self, ) -> int:
        return (
            self.eigenvalue_stability.count(ElementStability.UNSTABLE, )
            if self.eigenvalue_stability is not None else None
        )

    @property
    def message(self, ) -> str:
        return self.system_stability.message_template.format(
            num_forwards=self.num_forwards,
            num_backwards=self.num_backwards,
            num_unstable=self.num_unstable,
            rcond=self.rcond,
            rank_tolerance=self.rank_tolerance,
        )

    @property
    def is_success(self, ) -> bool:
        return self.system_stability == SystemStabilityVerdict.UNIQUE_STABLE

    def copy(self, ) -> Self:
        r"""
        """
        return self

    #]


class Solution:
    r"""
    ## Square solution:

    T: Transition matrix
    P: Impact matrix of transition shocks
    K: Intercept in transition equation
    Z: Measurement matrix
    H: Impact matrix of measurement shocks
    D: Intercept in measurement equation


    ## Equivalent block-triangular solution:

    Ta: Transition matrix in triangular system
    Pa: Impact matrix of transition shocks in triangular system
    Ka: Intercept in transition equation in triangular system
    Za: Measurement matrix in triangular system
    Ua: Rotation matrix from triangular to square system


    ## Forward expansion of square solution:

    J: Power matrix
    Ru: Forward-looking impact matrix of transition shocks
    X: Impact matrix in square system
    Xa: Impact matrix in triangular system


    ## Covariance matrices:

    cov_u: Covariance matrix of transition shocks
    cov_w: Covariance matrix of measurement shocks

    """
    #[

    __slots__ = (
        "T", "P", "K", "Z", "H", "D",
        "Ta", "Pa", "Ka", "Za", "Ua",
        "J", "Ru", "X", "Xa",
        "square_expansion",
        "triangular_expansion",
        "num_unit_roots",

        "transition_vector_stability",
        "measurement_vector_stability",

        "cov_u",
        "cov_w",
    )

    def __init__(self, **kwargs, ) -> None:
        r"""
        """
        for k in self.__slots__:
            setattr(self, k, None, )
        for k, v in kwargs.items():
            if k in self.__slots__:
                setattr(self, k, v, )

    def create_deviation_solution(self, ) -> Self:
        r"""
        Create a shallow copy of the solution, and replace constant vectors with
        zeros
        """
        new = type(self)()
        for n in new.__slots__:
            setattr(new, n, getattr(self, n, None), )
        if new.K is not None:
            new.K = _np.zeros_like(self.K, )
        if new.Ka is not None:
            new.Ka = _np.zeros_like(self.Ka, )
        if new.D is not None:
            new.D = _np.zeros_like(self.D, )
        return new

    @property
    def num_xi(self, ) -> int:
        """==Number of xi vector elements=="""
        return self.T.shape[0]

    @property
    def num_alpha(self, ) -> int:
        """==Number of alpha vector elements=="""
        return self.Ta.shape[0]

    @property
    def num_y(self, ) -> int:
        """==Number of y vector elements=="""
        return self.Z.shape[0]

    @property
    def num_u(self, ) -> int:
        """==Number of u vector elements=="""
        return self.P.shape[1]

    @property
    def num_v(self, ) -> int:
        """==Number of v vector elements=="""
        return self.P.shape[1]

    @property
    def num_w(self, ) -> int:
        """==Number of w vector elements=="""
        return self.H.shape[1]

    @property
    def num_stable(self, ) -> int:
        """==Number of stable elements in alpha vector=="""
        return self.num_alpha - self.num_unit_roots

    @property
    def Ta_stable(self, ) -> _np.ndarray:
        """==Stable part of transition matrix=="""
        num_unit_roots = self.num_unit_roots
        return self.Ta[num_unit_roots:, num_unit_roots:]

    @property
    def Pa_stable(self, ) -> _np.ndarray:
        """==Stable part of impact matrix of transition shocks=="""
        num_unit_roots = self.num_unit_roots
        return self.Pa[num_unit_roots:, :]

    @property
    def Ka_stable(self, ) -> _np.ndarray:
        """==Stable part of intercept in transition equation=="""
        num_unit_roots = self.num_unit_roots
        return self.Ka[num_unit_roots:]

    @property
    def Za_stable(self, ) -> _np.ndarray:
        """==Stable part of measurement matrix=="""
        num_unit_roots = self.num_unit_roots
        return self.Za[:, num_unit_roots:]

    @property
    def boolex_stable_transition_vector(self, ) -> tuple[int, ...]:
        r"""==Index of stable transition vector elements=="""
        return _np.array(tuple(
            i == ElementStability.STABLE
            for i in self.transition_vector_stability
        ), dtype=bool, )

    @property
    def boolex_stable_measurement_vector(self, ) -> tuple[int, ...]:
        r"""==Index of stable measurement vector elements=="""
        return _np.array(tuple(
            i == ElementStability.STABLE
            for i in self.measurement_vector_stability
        ), dtype=bool, )

    def unpack_square_solution(self, ) -> tuple[_np.ndarray, ...]:
        r"""
        Return square solution matrices in the following order:
        T, P, K, Z, H, D, None
        """
        return self.T, self.P, self.K, self.Z, self.H, self.D, None,

    def unpack_triangular_solution(self, ) -> tuple[_np.ndarray, ...]:
        r"""
        Return triangular solution matrices in the following order:
        Ta, Pa, Ka, Za, H, D, Ua
        """
        return self.Ta, self.Pa, self.Ka, self.Za, self.H, self.D, self.Ua,

    def copy(self, ) -> Self:
        new = type(self)()
        for n in self.__slots__:
            setattr(new, n, getattr(self, n, None, ), )
        new.square_expansion = list(self.square_expansion or [], )
        new.triangular_expansion = list(self.triangular_expansion or [], )
        return new

    def expand_square_solution(self, forward: int, ) -> list[_np.ndarray]:
        r"""
        Expand R matrices of square solution for t+1...t+forward
        """
        return _get_solution_expansion(
            self.square_expansion,
            self.P, self.X, self.J, self.Ru,
            forward,
        )

    def expand_triangular_solution(self, forward: int, ) -> list[_np.ndarray]:
        """
        Expand Ra matrices of square solution for t+1...t+forward
        """
        return _get_solution_expansion(
            self.triangular_expansion,
            self.Pa, self.Xa, self.J, self.Ru,
            forward,
        )

    def _classify_transition_vector_stability(
        self,
        tolerance: float,
    ) -> None:
        self.transition_vector_stability \
            = _classify_solution_vector_stability(
                self.Ua,
                self.num_unit_roots,
                tolerance=tolerance,
            )

    def _classify_measurement_vector_stability(
        self,
        tolerance: float,
    ) -> None:
        self.measurement_vector_stability \
            = _classify_solution_vector_stability(
                self.Za,
                self.num_unit_roots,
                tolerance=tolerance,
            )
    #]


def calculate_stability_and_solution(
    descriptor: _descriptors.Descriptor,
    system: _systems.System,
    tolerance: dict[str, float],
) -> tuple[Stability, Solution | None]:
    r"""
    Calculate the first-order solution for an unsolved-expectations system

    The eigenvalues are separated into three clusters (unit roots, stable
    roots, unstable roots) within a single generalized Schur decomposition,
    and are therefore classified exactly once; see `_solve_ordqz`
    """
    #[
    eigenvalue_tolerance = tolerance["eigenvalue"]
    stationarity_tolerance = tolerance["stationarity"]
    rank_tolerance = tolerance["rank"]
    num_forwards = descriptor.get_num_forwards()
    num_backwards = descriptor.get_num_backwards()
    #
    # Detach unstable from (stable + unit) roots, and unit from stable
    # roots, ordering the eigenvalues as unit, stable, unstable
    try:
        (
            qz_matrixes,
            eigenvalues,
            eigenvalue_stability,
            num_unit_roots,
        ) = _solve_ordqz(system, eigenvalue_tolerance, )
    except QzReorderingException:
        # The decomposition itself broke down, so there is nothing to diagnose;
        # report it as a verdict so that it reaches the user by the same route
        # as a Blanchard-Kahn failure
        return Stability.from_failed_decomposition(
            num_forwards=num_forwards,
            num_backwards=num_backwards,
            rank_tolerance=rank_tolerance,
        ), None,
    #
    # Diagnose the system, evaluating the Blanchard-Kahn conditions
    stability = Stability.from_decomposition(
        eigenvalues=eigenvalues,
        eigenvalue_stability=eigenvalue_stability,
        num_forwards=num_forwards,
        num_backwards=num_backwards,
        qz_matrixes=qz_matrixes,
        rank_tolerance=rank_tolerance,
    )
    #
    if not stability.is_success:
        return stability, None,
    #
    solution = Solution()
    solution.num_unit_roots = num_unit_roots
    #
    # Solve out expectations; the resulting transition matrix is already
    # block triangular with the unit roots in the leading block because the
    # QZ decomposition was reordered that way
    triangular_solution = _solve_transition_equations(descriptor, system, qz_matrixes, )
    #
    solution.Ua, solution.Ta, solution.Pa, solution.Ka, solution.Xa, solution.J, solution.Ru, = triangular_solution
    #
    # From the final triangular solution, calculate the square solution
    solution.T, solution.P, solution.K, solution.X, = _square_from_triangular(triangular_solution, )
    #
    # Solve measurement equations
    solution.Z, solution.H, solution.D, solution.Za, = _solve_measurement_equations(
        descriptor,
        system,
        solution.Ua,
    )
    solution._classify_transition_vector_stability(tolerance=stationarity_tolerance, )
    solution._classify_measurement_vector_stability(tolerance=stationarity_tolerance, )
    #
    solution.square_expansion = []
    solution.triangular_expansion = []
    #
    return stability, solution,


def left_div(A: _np.ndarray, B: _np.ndarray, ) -> _np.ndarray:
    r"""
    Solve A \ B = pinv(A) @ B or inv(A) @ B
    """
    return _np.linalg.lstsq(A, B, rcond=None)[0]


def right_div(B: _np.ndarray, A: _np.ndarray, ) -> _np.ndarray:
    r"""
    Solve B / A which is (A' \ B')'
    """
    return _np.linalg.lstsq(A.T, B.T, rcond=None)[0].T


def _solve_triangular(
    A: _np.ndarray,
    B: _np.ndarray,
) -> _np.ndarray:
    r"""
    Solve A \ B for an upper triangular A

    Note that the B factor of the real generalized Schur form is upper
    triangular, while the A factor is only quasi-upper-triangular; this
    function must never be pointed at the latter because the 2x2 blocks would
    be silently ignored
    """
    #[
    return _sp.linalg.solve_triangular(A, B, lower=False, )
    #]


def _factorize(
    A: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray, ]:
    r"""
    LU factorization of a general square A, to be reused across several
    right-hand sides in _solve_factorized
    """
    #[
    return _sp.linalg.lu_factor(A, )
    #]


def _solve_factorized(
    lu_piv: tuple[_np.ndarray, _np.ndarray, ],
    B: _np.ndarray,
    trans: int = 0,
) -> _np.ndarray:
    r"""
    Solve A \ B when trans=0, or A' \ B when trans=1, from an LU factorization
    of A obtained from _factorize
    """
    #[
    return _sp.linalg.lu_solve(lu_piv, B, trans=trans, )
    #]


def _estimate_rcond(
    A: _np.ndarray,
) -> float:
    r"""
    Estimate the reciprocal condition number of a general square A in the
    1-norm from its LU factorization

    This is a continuous measure of near singularity rather than an
    all-or-nothing rank verdict, and it costs an LU factorization plus O(n^2)
    instead of a singular value decomposition. An empty matrix is reported as
    perfectly conditioned: there is nothing for it to be singular about, and
    LAPACK rejects a zero leading dimension
    """
    #[
    if A.size == 0:
        return 1.0
    lu, _ = _sp.linalg.lu_factor(A, )
    gecon, = _sp.linalg.get_lapack_funcs(("gecon", ), (A, ), )
    rcond, info = gecon(lu, _np.linalg.norm(A, 1, ), norm="1", )
    if info < 0:
        raise ValueError(f"Illegal value in argument {-info} of gecon", )
    return float(rcond, )
    #]


def _square_from_triangular(
    triangular_solution: tuple[_np.ndarray, ...],
) -> tuple[_np.ndarray, ...]:
    r"""
    T <- Ua @ Ta / Ua
    R <- Ua @ Ra
    X <- Xa @ Ra
    K <- Ua @ Ka
    xi[t] = ... -X J**(k-1) Ru e[t+k]
    """
    #[
    Ua, Ta, Ra, Ka, Xa, *_ = triangular_solution
    #
    # T <- Ua @ (Ta / Ua), with Ta / Ua == (Ua' \ Ta')'
    # T = Ua @ right_div(Ta, Ua) # Ua @ (Ta / Ua)
    lu_Ua = _factorize(Ua, )
    T = Ua @ _solve_factorized(lu_Ua, Ta.T, trans=1, ).T
    #
    R = Ua @ Ra
    K = Ua @ Ka
    X = Ua @ Xa
    return T, R, K, X,
    #]


def _solve_measurement_equations(
    descriptor,
    system,
    Ua,
) -> tuple[_np.ndarray, ...]:
    r"""
    """
    #[
    num_forwards = descriptor.get_num_forwards()
    G = system.G[:, num_forwards:]
    #
    # All three solves share the same matrix, so factorize F only once
    lu_F = _factorize(system.F, )
    #
    # Z <- -(F \ G)
    # Z = left_div(-system.F, G) # -F \ G
    Z = -_solve_factorized(lu_F, G, )
    #
    # H <- -(F \ J)
    # H = left_div(-system.F, system.J) # -F \ J
    H = -_solve_factorized(lu_F, system.J, )
    #
    # D <- -(F \ H)
    # D = left_div(-system.F, system.H) # -F \ H
    D = -_solve_factorized(lu_F, system.H, )
    #
    Za = Z @ Ua
    return Z, H, D, Za,
    #]


def _solve_transition_equations(
    descriptor,
    system,
    qz_matrixes: tuple[_np.ndarray, ...],
) -> tuple[_np.ndarray, ...]:
    r"""
    """
    #[
    num_backwards = descriptor.get_num_backwards()
    num_forwards = descriptor.get_num_forwards()
    num_stable = num_backwards
    S, T, Q, Z, = qz_matrixes
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
    # Constant in transition equations
    Q_CC = Q @ system.C
    Q_CC1 = Q_CC[:num_stable]
    Q_CC2 = Q_CC[num_stable:]
    #
    # Transition shocks in transition equations
    Q_DD = Q @ system.D
    Q_DD1 = Q_DD[:num_stable, :]
    Q_DD2 = Q_DD[num_stable:, :]
    #
    # Unstable block
    #
    # S11 is solved against five right-hand sides and Z21 against one here and
    # one more in _square_from_triangular, so factorize each of them once; T22
    # is upper triangular and needs no factorization at all
    lu_S11 = _factorize(S11, )
    lu_Z21 = _factorize(Z21, )
    #
    # G <- -(Z21 \ Z22)
    # G = left_div(-Z21, Z22) # -Z21 \ Z22
    G = -_solve_factorized(lu_Z21, Z22, )
    #
    # Ru <- -(T22 \ Q_DD2)
    # Ru = left_div(-T22, Q_DD2) # -T22 \ Q_DD2
    Ru = -_solve_triangular(T22, Q_DD2, )
    #
    # Ku <- -((S22 + T22) \ Q_CC2)
    # Ku = left_div(-(S22 + T22), Q_CC2) # -(S22+T22) \ Q_CC2
    Ku = -_sp.linalg.solve(S22 + T22, Q_CC2, )
    #
    # Transform stable block==transform backward-looking variables:
    # gamma(t) = s(t) + G u(t+1)
    #
    # Xg0 <- S11 \ (T11 @ G + T12)
    # Xg0 = left_div(S11, T11 @ G + T12)
    Xg0 = _solve_factorized(lu_S11, T11 @ G + T12, )
    #
    # Xg1 <- G + (S11 \ S12)
    # Xg1 = G + left_div(S11, S12)
    Xg1 = G + _solve_factorized(lu_S11, S12, )
    #
    # Tg <- -(S11 \ T11)
    # Tg = left_div(-S11, T11)
    Tg = -_solve_factorized(lu_S11, T11, )
    #
    # Rg <- -Xg0 @ Ru - (S11 \ Q_DD1)
    # Rg = -Xg0 @ Ru - left_div(S11, Q_DD1)
    Rg = -Xg0 @ Ru - _solve_factorized(lu_S11, Q_DD1, )
    #
    # Kg <- -(Xg0 + Xg1) @ Ku - (S11 \ Q_CC1)
    # Kg = -(Xg0 + Xg1) @ Ku - left_div(S11, Q_CC1)
    Kg = -(Xg0 + Xg1) @ Ku - _solve_factorized(lu_S11, Q_CC1, )
    #
    Ug = Z21 # xib = Ug @ gamma
    #
    # Forward expansion
    # gamma(t) = ... -Xg J**(k-1) Ru e(t+k)
    #
    # J <- -(T22 \ S22)
    # J = left_div(-T22, S22) # -T22 \ S22
    J = -_solve_triangular(T22, S22, )
    #
    Xg = Xg1 + Xg0 @ J
    #
    return Ug, Tg, Rg, Kg, Xg, J, Ru,
    #]


def _solve_ordqz(
    system: _systems.System,
    tolerance: float,
) -> tuple[tuple[_np.ndarray, ...], tuple[complex, ...], tuple[ElementStability, ...], int, ]:
    r"""
    Calculate a generalized Schur (QZ) decomposition of the system with the
    eigenvalues separated into three clusters, ordered as unit roots, stable
    roots, unstable roots

    Scipy can only separate two clusters at a time. Instead of running a second
    decomposition (which would recompute the eigenvalues and could well classify
    them differently), the eigenvalues are computed and classified exactly once
    here, and the existing decomposition is then reordered in place by the
    LAPACK routine ?tgsen
    """
    #[
    #
    # Detach unstable roots from stable and unit roots; scipy calls the sort
    # criterion once, with the whole alpha and beta arrays
    def sort_stable_or_unit_root(alpha, beta, ) -> _np.ndarray:
        return _distance_from_unit_circle(alpha, beta, ) <= tolerance
    #
    try:
        S, T, alpha, beta, Q, Z = _sp.linalg.ordqz(
            system.A, system.B,
            sort=sort_stable_or_unit_root,
        )
    except ValueError as exception:
        # Scipy implements ordqz as an unsorted decomposition followed by a
        # reordering, and raises when that reordering fails
        raise QzReorderingException from exception
    #
    # Calculate and classify the eigenvalues; the classification relies on the
    # very same measure of the distance from the unit circle as the sort
    # criterion above, and is never revisited afterwards
    distance = _distance_from_unit_circle(alpha, beta, )
    eigenvalues = tuple(
        _eigenvalue_from_alpha_beta(a, b, )
        for a, b in zip(alpha, beta, )
    )
    eigenvalue_stability = tuple(
        _classify_from_distance(d, tolerance, )
        for d in distance
    )
    #
    # Detach unit roots from stable roots by moving the unit roots to the top
    # of the existing decomposition; the relative order of the remaining roots
    # is preserved, and hence the resulting order is unit, stable, unstable
    select = _np.array(
        tuple(i == ElementStability.UNIT_ROOT for i in eigenvalue_stability),
        dtype=_np.int32,
    )
    select = _sync_select_over_2x2_blocks(select, S, )
    num_unit_roots_check = int(select.sum())
    #
    tgsen, = _sp.linalg.get_lapack_funcs(("tgsen", ), (S, T, ), )
    S, T, _alphar, _alphai, _beta, Q, Z, num_unit_roots, *_, info = \
        tgsen(select, S, T, Q, Z, ijob=0, lwork=4*select.size+16, liwork=1, )
    if info < 0:
        raise ValueError(f"Illegal value in argument {-info} of tgsen", )
    if info > 0 or num_unit_roots != num_unit_roots_check:
        raise QzReorderingException
    #
    qz_matrixes = S, T, Q.T, Z,
    #
    # Reorder the eigenvalues and their classification exactly the way the
    # diagonal blocks have been reordered
    eigenvalues = (
        tuple(e for e, i in zip(eigenvalues, select, ) if i)
        + tuple(e for e, i in zip(eigenvalues, select, ) if not i)
    )
    eigenvalue_stability = (
        tuple(s for s, i in zip(eigenvalue_stability, select, ) if i)
        + tuple(s for s, i, in zip(eigenvalue_stability, select, ) if not i)
    )
    #
    return qz_matrixes, eigenvalues, eigenvalue_stability, num_unit_roots,
    #]


def _distance_from_unit_circle(
    alpha: _np.ndarray,
    beta: _np.ndarray,
) -> _np.ndarray:
    r"""
    Relative distance of the eigenvalues -beta/alpha from the unit circle:
    negative inside, zero on, and positive outside the unit circle

    Normalizing by the larger of the two moduli keeps the measure well scaled
    and free of overflow for infinite eigenvalues (alpha=0). A zero scale means
    a singular pencil (alpha=beta=0); an infinite distance is reported so that
    such a root is consistently treated as unstable both when the decomposition
    is sorted and when the eigenvalues are classified
    """
    #[
    abs_alpha = _np.abs(alpha, )
    abs_beta = _np.abs(beta, )
    scale = _np.maximum(abs_alpha, abs_beta, )
    return _np.divide(
        abs_beta - abs_alpha, scale,
        out=_np.full(_np.shape(scale, ), _np.inf, dtype=float, ),
        where=(scale != 0),
    )
    #]


def _eigenvalue_from_alpha_beta(
    alpha: Real | complex,
    beta: Real | complex,
) -> complex:
    r"""
    Eigenvalue -beta/alpha, reported as infinity when alpha=0, and as nan when
    the pencil is singular (alpha=beta=0)
    """
    #[
    if alpha:
        return complex(-beta / alpha, )
    return complex(_np.inf, 0, ) if beta else complex(_np.nan, _np.nan, )
    #]


def _classify_from_distance(
    distance: Real,
    tolerance: float,
) -> ElementStability:
    r"""
    Classify an eigenvalue as stable, unit root, or unstable, based on its
    relative distance from the unit circle
    """
    #[
    if distance < -tolerance:
        return ElementStability.STABLE
    if distance <= tolerance:
        return ElementStability.UNIT_ROOT
    return ElementStability.UNSTABLE
    #]


def _sync_select_over_2x2_blocks(
    select: _np.ndarray,
    S: _np.ndarray,
) -> _np.ndarray:
    r"""
    Make the selection consistent within the 2x2 diagonal blocks of the real
    generalized Schur form

    Both halves of a complex conjugate pair must be selected or unselected
    together. This follows from the classification itself because the two halves
    have identical moduli, and is enforced here only as a safety net because
    LAPACK rejects an inconsistent selection
    """
    #[
    select = _np.array(select, dtype=_np.int32, )
    subdiagonal = _np.diag(S, -1, )
    index = 0
    while index < select.size - 1:
        if subdiagonal[index]:
            both = select[index] or select[index+1]
            select[index] = both
            select[index+1] = both
            index += 2
        else:
            index += 1
    return select
    #]


def _classify_solution_vector_stability(
    transform_matrix: _np.ndarray,
    num_unit_roots: int,
    tolerance: float,
) -> tuple[ElementStability, ...]:
    r"""
    Classify the elements of a solution vector as stable or nonstationary
    depending on whether they load on any of the unit-root components

    The loadings are compared to the overall scale of the corresponding row so
    that the outcome does not depend on how the rows happen to be scaled
    """
    #[
    test_matrix = _np.abs(transform_matrix[:, :num_unit_roots], )
    row_scale = _np.max(
        _np.abs(transform_matrix, ),
        axis=1, keepdims=True, initial=0,
    )
    row_scale[row_scale == 0] = 1
    index = _np.any(test_matrix > tolerance*row_scale, axis=1, )
    return tuple(
        ElementStability.UNIT_ROOT if i
        else ElementStability.STABLE
        for i in index
    )
    #]


def _get_solution_expansion(
    existing_expansion: list[_np.ndarray],
    P, X, J, Ru,
    forward: int,
) -> list[_np.ndarray]:
    """
    Expand R matrices of square solution for t+1...t+forward
    """
    if (P is None) or (X is None) or (J is None) or (Ru is None):
        return None
    #
    # return [R(t), R(t+1), R(t+2), ..., R(t+forward)]
    #
    # R(t) = R
    # R(t+k) = -X J**(k-1) Ru e(t+k)
    # k = 1, ..., forward or k-1 = 0, ..., forward-1
    #
    R0 = _np.array(P, )
    existing_forward = len(existing_expansion)
    for k_minus_1 in range(existing_forward, forward):
        Rk = -X @ _np.linalg.matrix_power(J, k_minus_1, ) @ Ru
        existing_expansion.append(Rk, )
    return [R0, ] + existing_expansion[:forward]
    #
    # return [R, ] + [
    #     -X @ _np.linalg.matrix_power(J, k_minus_1) @ Ru
    #     for k_minus_1 in range(0, forward)
    # ]



STABLE = ElementStability.STABLE
UNIT_ROOT = ElementStability.UNIT_ROOT
UNSTABLE = ElementStability.UNSTABLE

