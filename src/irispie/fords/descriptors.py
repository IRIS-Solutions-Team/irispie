"""
# Descriptor for first-order systems and solutions

## Unsolved system

$$
A E[x_{t}] + B E[x_{t-1}] + C + D u_{t} + E v_{t} = 0 \\
F y_{t} + G x_{t} + H + J w_{t} = 0
$$

## State-space solution

$$
x_{t} = T x_{t-1} + K + R v_{t} \\
y_{t} = Z x_{t} + D + H w_{t}
$$
"""


#[
from __future__ import annotations

from typing import (Self, Any, )
from collections.abc import (Iterable, )
import itertools as _it
import functools as _ft
import dataclasses as _dc
import numpy as _np

from ..incidences import main as _incidence
from .. import equations as _equations
from .. import quantities as _quantities
from ..aldi import differentiators as _differentiators
from ..aldi import maps as _maps
from .. import sources as _sources
#]


class Descriptor:
    """
    Descriptor of first-order systems and solutions, and aldi Atom factory provider
    """
    #[

    __slots__ = (
        "system_vectors",
        "solution_vectors",
        "system_map",
        "aldi_context",
        "_column_to_eval",
    )

    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        quantities: Iterable[_quantities.Quantity],
        context: dict[str, Any] | None,
        /,
    ) -> None:
        self.system_vectors = SystemVectors(equations, quantities)
        self.solution_vectors = SolutionVectors(self.system_vectors)
        self.system_map = SystemMap(self.system_vectors)
        system_equations = _custom_order_equations_by_eids(
            equations,
            self.system_vectors.transition_eids
            + self.system_vectors.measurement_eids,
        )
        #
        # Create the evaluation context for the algorithmic differentiator
        atom_factory = self
        self.aldi_context = _differentiators.Context(
            atom_factory,
            system_equations,
            eid_to_wrts=self.system_vectors.eid_to_wrt_tokens,
            qid_to_logly=_quantities.create_qid_to_logly(quantities),
            context=context,
        )

    def get_num_backwards(self: Self) -> int:
        return self.system_vectors.get_num_backwards()

    def get_num_forwards(self: Self) -> int:
        return self.system_vectors.get_num_forwards()

    # ===== Implement AtomFactoryProtocol =====

    def create_diff_for_token(
        self,
        token: Token,
        wrt_tokens: tuple[Token, ...],
        /,
    ) -> _np.ndarray:
        """
        """
        if token is None:
            return _np.zeros((len(wrt_tokens), 1, ), )
        try:
            index = wrt_tokens.index(token, )
            diff = _np.zeros((len(wrt_tokens), 1, ), )
            diff[index] = 1
            return diff
        except:
            return 0

    def create_data_index_for_token(
        self,
        token: Token,
    ) -> tuple[int, slice]:
        """
        """
        return (token.qid, token.shift, )

    #]


"""
Quantity kinds that can be used in the transition part of a first-order system
"""
_TRANSITION_SYSTEM_QUANTITY = (
    _quantities.QuantityKind.TRANSITION_VARIABLE
    | _quantities.QuantityKind.UNANTICIPATED_SHOCK
    | _quantities.QuantityKind.ANTICIPATED_SHOCK
)


"""
Quantity kinds that can be used in the measurement part of a first-order system
"""
_MEASUREMENT_SYSTEM_QUANTITY = (
    _quantities.QuantityKind.MEASUREMENT_VARIABLE
    | _quantities.QuantityKind.TRANSITION_VARIABLE
    | _quantities.QuantityKind.MEASUREMENT_SHOCK
)


"""
Quantity kinds that can be used in a first-order system
"""
_SYSTEM_QUANTITY = (
    _TRANSITION_SYSTEM_QUANTITY
    | _MEASUREMENT_SYSTEM_QUANTITY
)


@_dc.dataclass(slots=True, )
class SystemVectors:
    """
    Vectors of quantities and equation ids in first-order system matrices
    """
    #[
    transition_eids: Iterable[int] | None = None
    measurement_eids: Iterable[int] | None = None
    eid_to_wrt_tokens: dict[int, Iterable[_incidence.Token]] | None = None
    #
    transition_variables: Iterable[_incidence.Token] | None = None
    transition_variables_are_logly: list[bool] | None = None
    are_initial_conditions: list[bool] | None = None,
    unanticipated_shocks: tuple[_incidence.Token, ...] | None = None
    anticipated_shocks: tuple[_incidence.Token, ...] | None = None
    measurement_variables: Iterable[_incidence.Token] | None = None
    measurement_variables_are_logly: list[bool] | None = None
    measurement_shocks: tuple[_incidence.Token, ...] | None = None
    #
    shape_A_excl_dynid: tuple[int, int] | None = None
    shape_B_excl_dynid: tuple[int, int] | None = None
    shape_C_excl_dynid: tuple[int] | None = None
    shape_D_excl_dynid: tuple[int, int] | None = None
    shape_E_excl_dynid: tuple[int, int] | None = None
    shape_F: tuple[int, int] | None = None
    shape_G: tuple[int, int] | None = None
    shape_H: tuple[int] | None = None
    shape_J: tuple[int, int] | None = None

    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        quantities: Iterable[_quantities.Quantity],
        /,
    ) -> None:
        """
        Construct system vectors from a list of equations and a list of quantities
        """
        qid_to_logly = _quantities.create_qid_to_logly(quantities, )
        #
        self.transition_eids \
            = sorted([eqn.id for eqn in equations if eqn.kind in _equations.EquationKind.TRANSITION_EQUATION])
        self.measurement_eids \
            = sorted([eqn.id for eqn in equations if eqn.kind in _equations.EquationKind.MEASUREMENT_EQUATION])
        qid_to_kind = _quantities.create_qid_to_kind(quantities, )
        all_tokens = set(_equations.generate_all_tokens_from_equations(equations))
        all_wrt_tokens = set(_incidence.generate_tokens_of_kinds(all_tokens, qid_to_kind, _SYSTEM_QUANTITY))
        self.eid_to_wrt_tokens = _equations.create_eid_to_wrt_tokens(equations, all_wrt_tokens)
        #
        actual_tokens_transition_variables \
            = set(_incidence.generate_tokens_of_kinds(all_tokens, qid_to_kind, _quantities.QuantityKind.TRANSITION_VARIABLE))
        #
        # Make adjustment for transition variables in measurement
        # equations: each x(t-k) in measurement needs to be in the current
        # dated (LHS) vector of transition variables; this is done by
        # pretending x(t-k-1) is needed
        #
        adjusted_tokens_transition_variables \
            = _adjust_for_measurement_equations(actual_tokens_transition_variables, equations, qid_to_kind)
        #
        self.transition_variables \
            = _incidence.sort_tokens(_create_system_transition_vector(adjusted_tokens_transition_variables))
        self.transition_variables_are_logly \
            = [ qid_to_logly[tok.qid] for tok in self.transition_variables ]
        self.are_initial_conditions = [
            _incidence.Token(t.qid, t.shift-1) in actual_tokens_transition_variables and t.shift <= 0
            for t in self.transition_variables
        ]
        #
        # Unanticipated shocks
        #
        self.unanticipated_shocks \
            = tuple(_incidence.sort_tokens(_incidence.generate_tokens_of_kinds(
                all_tokens, qid_to_kind, _quantities.QuantityKind.UNANTICIPATED_SHOCK,
            )))
        #
        # Anticipated shocks
        #
        self.anticipated_shocks \
            = tuple(_incidence.sort_tokens(_incidence.generate_tokens_of_kinds(
                all_tokens, qid_to_kind, _quantities.QuantityKind.ANTICIPATED_SHOCK,
            )))
        #
        # Measurement variables
        #
        self.measurement_variables \
            = tuple(_incidence.sort_tokens(_incidence.generate_tokens_of_kinds(
                all_tokens, qid_to_kind, _quantities.QuantityKind.MEASUREMENT_VARIABLE,
            )))
        self.measurement_variables_are_logly \
            = [ qid_to_logly[tok.qid] for tok in self.measurement_variables ]
        #
        # Measurement shocks
        #
        self.measurement_shocks \
            = tuple(_incidence.sort_tokens(_incidence.generate_tokens_of_kinds(
                all_tokens, qid_to_kind, _quantities.QuantityKind.MEASUREMENT_SHOCK,
            )))
        #
        # Shapes of matrices
        #
        self.shape_A_excl_dynid = (len(self.transition_eids), len(self.transition_variables), )
        self.shape_B_excl_dynid = self.shape_A_excl_dynid
        self.shape_C_excl_dynid = (len(self.transition_eids), )
        self.shape_D_excl_dynid = (len(self.transition_eids), len(self.unanticipated_shocks), )
        self.shape_E_excl_dynid = (len(self.transition_eids), len(self.anticipated_shocks), )
        self.shape_F = (len(self.measurement_eids), len(self.measurement_variables), )
        self.shape_G = (len(self.measurement_eids), len(self.transition_variables), )
        self.shape_H = (len(self.measurement_eids), )
        self.shape_J = (len(self.measurement_eids), len(self.measurement_shocks), )

    def get_num_backwards(self) -> int:
        return _get_num_backwards(self.transition_variables)

    def get_num_forwards(self) -> int:
        return _get_num_forwards(self.transition_variables)
    #]


@_dc.dataclass(slots=True, )
class SolutionVectors:
    """
    Vectors of quantities in first-order solution matrices
    """
    #[
    transition_variables: tuple[_incidence.Token, ...] | None = None
    are_initial_conditions: list[bool, ...] | None = None
    unanticipated_shocks: tuple[_incidence.Token, ...] | None = None
    anticipated_shocks: tuple[_incidence.Token, ...] | None = None
    measurement_variables: tuple[_incidence.Token, ...] | None = None
    measurement_shocks: tuple[_incidence.Token, ...] | None = None

    def __init__(self, system_vectors: SystemVectors, /, ) -> None:
        """
        Construct solution vectors and initial conditions indicator
        """
        self.transition_variables, self.are_initial_conditions = \
            _solution_vector_from_system_vector(system_vectors.transition_variables, system_vectors.are_initial_conditions)
        self.unanticipated_shocks = tuple(system_vectors.unanticipated_shocks)
        self.anticipated_shocks = tuple(system_vectors.anticipated_shocks)
        self.measurement_variables = tuple(system_vectors.measurement_variables)
        self.measurement_shocks = tuple(system_vectors.measurement_shocks)

    def get_initials(
        self,
        /,
    ) -> Iterable[_incidence.Token]:
        """
        Get tokens representing required initial conditions
        """
        return list(_it.compress(self.transition_variables, self.are_initial_conditions))
    #]


@_dc.dataclass(slots=True, )
class HumanSolutionVectors:
    """
    """
    #[

    transition_variables: tuple[str, ...] | None = None
    unanticipated_shocks: tuple[str, ...] | None = None
    anticipated_shocks: tuple[str, ...] | None = None
    measurement_variables: tuple[str, ...] | None = None
    measurement_shocks: tuple[str, ...] | None = None

    def __init__(
        self,
        solution_vectors: SolutionVectors,
        qid_to_name: dict[int, str],
        qid_to_logly: dict[int, bool],
        /,
    ) -> None:
        """
        """
        print_tokens = _ft.partial(_incidence.print_tokens, qid_to_name=qid_to_name, qid_to_logly=qid_to_logly, )
        for a in self.__slots__:
            setattr(self, a, print_tokens(getattr(solution_vectors, a)))

    #]


def _create_system_transition_vector(
    tokens_transition_variables: Iterable[_incidence.Token],
    /,
) -> Iterable[_incidence.Token]:
    """
    From tokens of transition variables, create vector of transition variables
    along columns of matrix A in unsolved system
    """
    #[
    tokens_transition_variables = set(tokens_transition_variables)
    min_shifts = _incidence.get_some_shift_by_quantities(tokens_transition_variables, lambda x: min(min(x), -1))
    max_shifts = _incidence.get_some_shift_by_quantities(tokens_transition_variables, max)
    #
    vector_for_id = lambda qid: [_incidence.Token(qid, sh) for sh in range(min_shifts[qid]+1, max_shifts[qid]+1)]
    unique_ids = set(t.qid for t in tokens_transition_variables)
    return _it.chain.from_iterable(vector_for_id(i) for i in unique_ids)
    #]


def _solution_vector_from_system_vector(
    system_transition_vector: Iterable[_incidence.Token],
    are_initial_conditions: Iterable[bool],
    /,
) -> Iterable[_incidence.Token]:
    """
    From sorted system vector, get vector of transition variables in solved
    system and the indicator of required initial conditions
    """
    num_forwards = _get_num_forwards(system_transition_vector)
    return (
        tuple(system_transition_vector[num_forwards:]),
        tuple(are_initial_conditions[num_forwards:]),
    )


def _get_num_forwards(system_transition_vector: Iterable[_incidence.Token]):
    """
    Number of forward-looking tokens in a vector of tokens
    """
    return sum(1 for t in system_transition_vector if t.shift>0)


def _get_num_backwards(system_transition_vector: Iterable[_incidence.Token]):
    """
    Number of backward-looking tokens in a vector of tokens
    """
    return len(system_transition_vector) - _get_num_forwards(system_transition_vector)


@_dc.dataclass(slots=True, )
class SystemMap:
    """
    """
    #[
    A: _maps.ArrayMap | None = None
    B: _maps.ArrayMap | None = None
    C: None = None
    D: _maps.ArrayMap | None = None
    E: _maps.ArrayMap | None = None
    #
    dynid_A: _np.ndarray | None = None
    dynid_B: _np.ndarray | None = None
    dynid_C: _np.ndarray | None = None
    dynid_D: _np.ndarray | None = None
    dynid_E: _np.ndarray | None = None
    #
    F: _maps.ArrayMap | None = None
    G: _maps.ArrayMap | None = None
    H: None = None
    J: _maps.ArrayMap | None = None

    def __init__(
        self,
        system_vectors: SystemVectors,
    ) -> None:
        """
        """
        system_eids = system_vectors.transition_eids + system_vectors.measurement_eids
        #
        # Create the map from equation ids to rhs offset; the offset is the
        # number of rows in the Jacobian matrix that precede the equation
        eid_to_rhs_offset = _maps.create_eid_to_rhs_offset(system_eids, system_vectors.eid_to_wrt_tokens)
        #
        # Transition equations
        #
        self.A = _maps.ArrayMap(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.transition_variables,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        lagged_transition_variables = [ t.shifted(-1) for t in system_vectors.transition_variables ]
        lagged_transition_variables = [
            t if t not in system_vectors.transition_variables else None
            for t in lagged_transition_variables
        ]
        #
        self.B = _maps.ArrayMap(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            lagged_transition_variables,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.A.remove_nones()
        self.B.remove_nones()
        #
        self.C = _maps.VectorMap(system_vectors.transition_eids, )
        #
        self.D = _maps.ArrayMap(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.unanticipated_shocks,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.E = _maps.ArrayMap(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.anticipated_shocks,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        num_dynid_rows = len(system_vectors.transition_variables) - len(system_vectors.transition_eids)
        self.dynid_A, self.dynid_B = _create_dynid_matrices(system_vectors.transition_variables, )
        self.dynid_C = _np.zeros((num_dynid_rows, ), dtype=float, )
        self.dynid_D = _np.zeros((num_dynid_rows, system_vectors.shape_D_excl_dynid[1]), dtype=float, )
        self.dynid_E = _np.zeros((num_dynid_rows, system_vectors.shape_E_excl_dynid[1]), dtype=float, )
        #
        # Measurement equations
        #
        self.F = _maps.ArrayMap(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.measurement_variables,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.G = _maps.ArrayMap(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.transition_variables,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.H = _maps.VectorMap(system_vectors.measurement_eids, )
        #
        self.J = _maps.ArrayMap(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.measurement_shocks,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
    #]


def _create_dynid_matrices(system_transition_vector: Iterable[_incidence.Token]):
    """
    Create dynamic identity matrix for unsolved system
    """
    #[
    max_shifts = _incidence.get_some_shift_by_quantities(system_transition_vector, max)
    #
    index_A = ([], [])
    index_B = ([], [])
    row_count = 0
    for i, t in enumerate(system_transition_vector):
        if t.shift==max_shifts[t.qid]:
            continue
        j = system_transition_vector.index(t.shifted(+1))
        index_A[0].append(row_count)
        index_A[1].append(i)
        index_B[0].append(row_count)
        index_B[1].append(j)
        row_count += 1
    #
    num_columns = len(system_transition_vector)
    dynid_A = _np.zeros((row_count, num_columns), dtype=float)
    dynid_B = _np.zeros((row_count, num_columns), dtype=float)
    dynid_A[index_A] = 1
    dynid_B[index_B] = -1
    return dynid_A, dynid_B
    #]


def _adjust_for_measurement_equations(
    tokens_transition_variables: Iterable[_quantities.Quantity],
    equations: Iterable[_equations.Equation],
    qid_to_kind: dict[int, _quantities.QuantityKind],
    /,
) -> Iterable[_quantities.Quantity]:
    """
    """
    #[
    tokens_in_measurement_equations = _it.chain.from_iterable(e.incidence for e in equations if e.kind is _equations.EquationKind.MEASUREMENT_EQUATION)
    pretend_needed = [
        _incidence.Token(t.qid, t.shift-1) for t in tokens_in_measurement_equations
        if qid_to_kind[t.qid] in _quantities.QuantityKind.TRANSITION_VARIABLE
    ]
    return set(tokens_transition_variables).union(pretend_needed)
    #]


def _custom_order_equations_by_eids(
    equations: Iterable[_equations.Equation],
    eids: list[int],
    /,
) -> tuple[_equations.Equation, ...]:
    """
    """
    #[
    eid_to_equation = { eqn.id: eqn for eqn in equations }
    return tuple( eid_to_equation[eid] for eid in eids )
    #]

