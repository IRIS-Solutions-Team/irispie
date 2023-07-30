"""
# Descriptor for first-order systems and solutions

## Unsolved system

$$
A E[x_{t}] + B E[x_{t-1}] + C + D v_{t} = 0 \\
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

from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )
import itertools as it_
import dataclasses as _dc
import numpy as np_

from .. import (incidence as in_, equations as eq_, quantities as qu_, )
from ..aldi import (differentiators as ad_, )
from ..aldi import (maps as am_, )
#]


class _AtomFactory:
    """
    """
    #[
    @staticmethod
    def create_diff_for_token(
        token: Token,
        wrt_tokens: tuple[Token, ...],
        /,
    ) -> np_.ndarray:
        """
        """
        try:
            index = wrt_tokens.index(token)
            diff = np_.zeros((len(wrt_tokens), 1))
            diff[index] = 1
            return diff
        except:
            return 0

    @staticmethod
    def create_data_index_for_token(
        token: Token,
        columns_to_eval: tuple[int, slice],
    ) -> tuple[int, slice]:
        """
        """
        return (
            token.qid,
            slice(columns_to_eval[0]+token.shift, columns_to_eval[1]+token.shift+1),
        )
    #]


@_dc.dataclass(slots=True, )
class Descriptor:
    """
    """
    #[
    system_vectors: SystemVectors | None = None
    solution_vectors: SolutionVectors | None = None
    system_map: SystemMap | None = None
    aldi_context: ad_.Context | None = None

    def __init__(
        self,
        equations: eq_.Equations,
        quantities: qu_.Quantities,
        custom_functions: dict | None,
        /,
    ) -> NoReturn:
        self.system_vectors = SystemVectors(equations, quantities)
        self.solution_vectors = SolutionVectors(self.system_vectors)
        self.system_map = SystemMap(self.system_vectors)
        #
        # Create the context for the algorithmic differentiator
        system_equations = _select_system_equations_from_equations(
            equations,
            self.system_vectors.transition_eids,
            self.system_vectors.measurement_eids,
        )
        #
        # Create the evaluation context for the algorithmic differentiator
        self.aldi_context = ad_.Context(
            system_equations,
            _AtomFactory,
            eid_to_wrts=self.system_vectors.eid_to_wrt_tokens,
            qid_to_logly=qu_.create_qid_to_logly(quantities),
            num_columns_to_eval=1,
            custom_functions=custom_functions,
        )

    def get_num_backwards(self: Self) -> int:
        return self.system_vectors.get_num_backwards()

    def get_num_forwards(self: Self) -> int:
        return self.system_vectors.get_num_forwards()
    #]


"""
Quantity kinds that can be used in the transition part of a first-order system
"""
_TRANSITION_SYSTEM_QUANTITY = (
    qu_.QuantityKind.TRANSITION_VARIABLE
    | qu_.QuantityKind.TRANSITION_SHOCK
)


"""
Quantity kinds that can be used in the measurement part of a first-order system
"""
_MEASUREMENT_SYSTEM_QUANTITY = (
    qu_.QuantityKind.MEASUREMENT_VARIABLE
    | qu_.QuantityKind.TRANSITION_VARIABLE
    | qu_.QuantityKind.MEASUREMENT_SHOCK
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
    eid_to_wrt_tokens: dict[int, in_.Tokens] | None = None
    #
    transition_variables: in_.Tokens | None = None
    initial_conditions: Iterable[bool] | None = None,
    transition_shocks: in_.Tokens | None = None 
    measurement_variables: in_.Tokens | None = None
    measurement_shocks: in_.Tokens | None = None 
    #
    shape_A_excl_dynid: tuple[int, int] | None = None
    shape_B_excl_dynid: tuple[int, int] | None = None
    shape_C_excl_dynid: tuple[int, int] | None = None
    shape_D_excl_dynid: tuple[int, int] | None = None
    shape_F: tuple[int, int] | None = None
    shape_G: tuple[int, int] | None = None
    shape_H: tuple[int, int] | None = None
    shape_J: tuple[int, int] | None = None

    def __init__(self, equations: eq_.Equations, quantities: qu_.Quantities) -> NoReturn:
        """
        Construct system vectors from a list of equations and a list of quantities
        """
        self.transition_eids = sorted([eqn.id for eqn in equations if eqn.kind in eq_.EquationKind.TRANSITION_EQUATION])
        self.measurement_eids = sorted([eqn.id for eqn in equations if eqn.kind in eq_.EquationKind.MEASUREMENT_EQUATION])
        qid_to_kind = qu_.create_qid_to_kind(quantities)
        all_tokens = set(eq_.generate_all_tokens_from_equations(equations))
        all_wrt_tokens = set(in_.generate_tokens_of_kinds(all_tokens, qid_to_kind, _SYSTEM_QUANTITY))
        self.eid_to_wrt_tokens = eq_.create_eid_to_wrt_tokens(equations, all_wrt_tokens)
        #
        actual_tokens_transition_variables = set(in_.generate_tokens_of_kinds(all_tokens, qid_to_kind, qu_.QuantityKind.TRANSITION_VARIABLE))
        #
        # Make adjustment for transition variables in measurement
        # equations: each x(t-k) in measurement needs to be in the current
        # dated (LHS) vector of transition variables; this is done by
        # pretending x(t-k-1) is needed
        adjusted_tokens_transition_variables = _adjust_for_measurement_equations(actual_tokens_transition_variables, equations, qid_to_kind)
        #
        self.transition_variables = in_.sort_tokens(_create_system_transition_vector(adjusted_tokens_transition_variables))
        self.initial_conditions = [ in_.Token(t.qid, t.shift-1) in actual_tokens_transition_variables and t.shift <= 0 for t in self.transition_variables ]
        self.transition_shocks = in_.sort_tokens(in_.generate_tokens_of_kinds(all_tokens, qid_to_kind, qu_.QuantityKind.TRANSITION_SHOCK))
        #
        self.measurement_variables = in_.sort_tokens(in_.generate_tokens_of_kinds(all_tokens, qid_to_kind, qu_.QuantityKind.MEASUREMENT_VARIABLE))
        self.measurement_shocks = in_.sort_tokens(in_.generate_tokens_of_kinds(all_tokens, qid_to_kind, qu_.QuantityKind.MEASUREMENT_SHOCK))
        #
        self.shape_A_excl_dynid = (len(self.transition_eids), len(self.transition_variables))
        self.shape_B_excl_dynid = self.shape_A_excl_dynid
        self.shape_C_excl_dynid = (len(self.transition_eids), 1)
        self.shape_D_excl_dynid = (len(self.transition_eids), len(self.transition_shocks))
        #
        self.shape_F = (len(self.measurement_eids), len(self.measurement_variables))
        self.shape_G = (len(self.measurement_eids), len(self.transition_variables))
        self.shape_H = (len(self.measurement_eids), 1)
        self.shape_J = (len(self.measurement_eids), len(self.measurement_shocks))

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
    transition_variables: in_.Tokens | None = None
    initial_conditions: Iterable[bool] | None = None,
    transition_shocks: in_.Tokens | None = None 
    measurement_variables: in_.Tokens | None = None
    measurement_shocks: in_.Tokens | None = None 

    def __init__(self, system_vectors: SystemVectors, /, ) -> NoReturn:
        """
        Construct solution vectors and initial conditions indicator
        """
        self.transition_variables, self.initial_conditions = _solution_vector_from_system_vector(system_vectors.transition_variables, system_vectors.initial_conditions)
        self.transition_shocks = system_vectors.transition_shocks
        self.measurement_variables = system_vectors.measurement_variables
        self.measurement_shocks = system_vectors.measurement_shocks

    def get_initials(
        self,
        /, 
        kind: Literal["required"] | Literal["discarded"] = "required",
    ) -> in_.Tokens:
        """
        Get tokens representing required initial conditions
        """
        return list(it_.compress(self.transition_variables, self.initial_conditions))
    #]


def _create_system_transition_vector(tokens_transition_variables: in_.Tokens, /, ) -> in_.Tokens:
    """
    From tokens of transition variables, create vector of transition variables
    along columns of matrix A in unsolved system
    """
    #[
    tokens_transition_variables = set(tokens_transition_variables)
    min_shifts = in_.get_some_shift_by_quantities(tokens_transition_variables, lambda x: min(min(x), -1))
    max_shifts = in_.get_some_shift_by_quantities(tokens_transition_variables, max)
    #
    vector_for_id = lambda qid: [in_.Token(qid, sh) for sh in range(min_shifts[qid]+1, max_shifts[qid]+1)]
    unique_ids = set(t.qid for t in tokens_transition_variables)
    return it_.chain.from_iterable(vector_for_id(i) for i in unique_ids)
    #]


def _solution_vector_from_system_vector(
    system_transition_vector: in_.Tokens, 
    initial_conditions: Iterable[bool],
    /,
) -> in_.Tokens:
    """
    From sorted system vector, get vector of transition variables in solved
    system and the indicator of required initial conditions
    """
    num_forwards = _get_num_forwards(system_transition_vector)
    return system_transition_vector[num_forwards:], initial_conditions[num_forwards:]


def _get_num_forwards(system_transition_vector: in_.Tokens):
    """
    Number of forward-looking tokens in a vector of tokens
    """
    return sum(1 for t in system_transition_vector if t.shift>0)


def _get_num_backwards(system_transition_vector: in_.Tokens):
    """
    Number of backward-looking tokens in a vector of tokens
    """
    return len(system_transition_vector) - _get_num_forwards(system_transition_vector)


@_dc.dataclass(slots=True, )
class SystemMap:
    """
    """
    #[
    A: am_.ArrayMap | None = None
    B: am_.ArrayMap | None = None
    C: None = None
    D: am_.ArrayMap | None = None
    dynid_A: np_.ndarray | None = None
    dynid_B: np_.ndarray | None = None
    dynid_C: np_.ndarray | None = None
    dynid_D: np_.ndarray | None = None
    #
    F: am_.ArrayMap | None = None
    G: am_.ArrayMap | None = None
    H: None = None
    J: am_.ArrayMap | None = None

    def __init__(
        self,
        system_vectors: SystemVectors,
    ) -> NoReturn:
        """
        """
        system_eids = system_vectors.transition_eids + system_vectors.measurement_eids
        #
        # Create the map from equation ids to rhs offset; the offset is the
        # number of rows in the Jacobian matrix that precede the equation
        eid_to_rhs_offset = am_.create_eid_to_rhs_offset(system_eids, system_vectors.eid_to_wrt_tokens)
        #
        # Transition equations
        #
        self.A = am_.ArrayMap.for_equations(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.transition_variables,
            eid_to_rhs_offset,
            #
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
        self.B = am_.ArrayMap.for_equations(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            lagged_transition_variables, 
            eid_to_rhs_offset,
            #
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.A.remove_nones()
        self.B.remove_nones()
        #
        self.C = am_.ArrayMap.constant_vector(system_vectors.transition_eids)
        #
        self.D = am_.ArrayMap.for_equations(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.transition_shocks,
            eid_to_rhs_offset,
            #
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        num_dynid_rows = len(system_vectors.transition_variables) - len(system_vectors.transition_eids)
        self.dynid_A, self.dynid_B = _create_dynid_matrices(system_vectors.transition_variables, )
        self.dynid_C = np_.zeros((num_dynid_rows, system_vectors.shape_C_excl_dynid[1]), dtype=float, )
        self.dynid_D = np_.zeros((num_dynid_rows, system_vectors.shape_D_excl_dynid[1]), dtype=float, )
        #
        # Measurement equations
        #
        self.F = am_.ArrayMap.for_equations(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.measurement_variables, 
            eid_to_rhs_offset,
            #
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.G = am_.ArrayMap.for_equations(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.transition_variables, 
            eid_to_rhs_offset,
            #
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.H = am_.ArrayMap.constant_vector(system_vectors.measurement_eids)
        #
        self.J = am_.ArrayMap.for_equations(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.measurement_shocks, 
            eid_to_rhs_offset,
            #
            rhs_column=0,
            lhs_column_offset=0,
        )
    #]


def _create_dynid_matrices(system_transition_vector: in_.Tokens):
    """
    Create dynamic identity matrix for unsolved system
    """
    #[
    max_shifts = in_.get_some_shift_by_quantities(system_transition_vector, max)
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
    dynid_A = np_.zeros((row_count, num_columns), dtype=float)
    dynid_B = np_.zeros((row_count, num_columns), dtype=float)
    dynid_A[index_A] = 1
    dynid_B[index_B] = -1
    return dynid_A, dynid_B
    #]


def _adjust_for_measurement_equations(
    tokens_transition_variables: qu_.Quantities,
    equations: eq_.Equations,
    qid_to_kind: dict[int, qu_.QuantityKind],
    /,
) -> qu_.Quantities:
    """
    """
    #[
    tokens_in_measurement_equations = it_.chain.from_iterable(e.incidence for e in equations if e.kind is eq_.EquationKind.MEASUREMENT_EQUATION)
    pretend_needed = [
        in_.Token(t.qid, t.shift-1) for t in tokens_in_measurement_equations
        if qid_to_kind[t.qid] in qu_.QuantityKind.TRANSITION_VARIABLE
    ]
    return set(tokens_transition_variables).union(pretend_needed)
    #]


def _select_system_equations_from_equations(
    equations: eq_.Equations,
    transition_eids: list[int],
    measurement_eids: list[int],
    /
) -> eq_.Equations:
    eid_to_equation = { eqn.id:eqn for eqn in equations }
    system_eids = transition_eids + measurement_eids
    return ( eid_to_equation[eid] for eid in system_eids )


