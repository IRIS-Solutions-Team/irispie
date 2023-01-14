"""
# Metadata on first-order system and solution

## Unsolved system

$$
A E[x_t] + B E[x_{t-1}] + C + D v_t = 0 \\
F y_t + G x_t + H + J w_t = 0
$$

## State-space solution

$$
x_t = T x{t-1} + K + R v_t \\
y_t = Z x_t + D + H w_t
$$

"""

#[
from __future__ import annotations

import dataclasses
import numpy 
import itertools
import operator

from typing import Self, NoReturn
from collections.abc import Iterable

from .equations import (
    EquationKind, Equations,
    generate_all_tokens_from_equations,
    create_eid_to_wrt_tokens,
)
from .quantities import (
    QuantityKind, Quantities,
    create_qid_to_kind,
)
from .incidence import (
    Token, Tokens,
    get_some_shifts_by_quantities,
    sort_tokens, generate_tokens_of_kinds,
)
from .audi import (
    Context, DifferentiationAtom,
)
#]


@dataclasses.dataclass
class SystemVectors:
    """
    Vectors of quantities in first-order system matrices
    """
    #[
    transition_equations: Equations | None = None
    transition_eid_to_wrt_tokens: dict[int, Tokens] | None = None

    measurement_equations: Equations | None = None
    measurement_eid_to_wrt_tokens: dict[int, Tokens] | None = None

    transition_variables: Tokens | None = None
    transition_shocks: Tokens | None = None 

    measurement_variables: Tokens | None = None
    measurement_shocks: Tokens | None = None 

    def __init__(self, equations: Equations, quantities: Quantities) -> NoReturn:
        all_tokens = set(generate_all_tokens_from_equations(equations))
        qid_to_kind = create_qid_to_kind(quantities)

        all_transition_wrt_tokens = set(tok for tok in all_tokens if qid_to_kind[tok.qid] in QuantityKind.IN_TRANSITION_SYSTEM_WRTS)
        self.transition_equations = sorted([ eqn for eqn in equations if eqn.kind in EquationKind.TRANSITION_EQUATION ], key=operator.attrgetter("id"))
        self.transition_eid_to_wrt_tokens = create_eid_to_wrt_tokens(self.transition_equations, all_transition_wrt_tokens)

        all_measurement_wrt_tokens = set(tok for tok in all_tokens if qid_to_kind[tok.qid] in QuantityKind.IN_MEASUREMENT_SYSTEM_WRTS)
        self.measurement_equations = sorted([ eqn for eqn in equations if eqn.kind in EquationKind.MEASUREMENT_EQUATION ], key=operator.attrgetter("id"))
        self.measurement_eid_to_wrt_tokens = create_eid_to_wrt_tokens(self.measurement_equations, all_measurement_wrt_tokens)

        tokens_transition_variables = generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.TRANSITION_VARIABLE)
        self.transition_variables = sort_tokens(_create_system_transition_vector(tokens_transition_variables))
        self.transition_shocks = sort_tokens(generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.TRANSITION_SHOCK))
        self.measurement_variables = sort_tokens(generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.MEASUREMENT_VARIABLE))
        self.measurement_shocks = sort_tokens(generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.MEASUREMENT_SHOCK))
    #]


@dataclasses.dataclass
class SystemDifferentiationContexts:
    """
    """
    #[
    transition_equations: Context | None = None
    measurement_equations: Context | None = None

    def __init__(
        self,
        system_vectors: SystemVectors,
        /
    ) -> Self:
        """
        """
        self.transition_equations = Context.for_equations(
            DifferentiationAtom, 
            system_vectors.transition_equations,
            system_vectors.transition_eid_to_wrt_tokens,
        )
        self.measurement_equations = Context.for_equations(
            DifferentiationAtom, 
            system_vectors.measurement_equations,
            system_vectors.measurement_eid_to_wrt_tokens,
        )
    #]


@dataclasses.dataclass
class SolutionVectors:
    """
    Vectors of quantities in first-order solution matrices
    """
    #[
    transition_variables: Tokens | None = None
    transition_shocks: Tokens | None = None 

    measurement_variables: Tokens | None = None
    measurement_shocks: Tokens | None = None 

    def __init__(self, system_vectors: SystemVectors) -> NoReturn:
        self.transition_variables = _solution_vector_from_system_vector(system_vectors.transition_variables)
        self.transition_shocks = system_vectors.transition_shocks
        self.measurement_variables = system_vectors.measurement_variables
        self.measurement_shocks = system_vectors.measurement_shocks
    #]


def _create_system_transition_vector(tokens_transition_variables: Tokens) -> Tokens:
    """
    From tokens of transition variables, create vector of transition variables
    along columns of matrix A in unsolved system
    """
    #[
    tokens_transition_variables = set(tokens_transition_variables)
    min_shifts = get_some_shifts_by_quantities(tokens_transition_variables, lambda x: min(min(x), -1))
    max_shifts = get_some_shifts_by_quantities(tokens_transition_variables, max)
    vector_for_id = lambda qid: [Token(qid, sh) for sh in range(min_shifts[qid]+1, max_shifts[qid]+1)]
    unique_ids = set(t.qid for t in tokens_transition_variables)
    return itertools.chain.from_iterable(vector_for_id(i) for i in unique_ids)
    #]


def _solution_vector_from_system_vector(system_transition_vector: Tokens) -> Tokens:
    """
    From sorted system vector, get vector of transition variables in solved system 
    """
    num_forwards = _get_num_forwards(system_transition_vector)
    return system_transition_vector[num_forwards:]


def _get_num_forwards(system_transition_vector: Tokens):
    """
    Number of forward-looking tokens in a vector of tokens
    """
    return sum(1 for t in system_transition_vector if t.shift>0)


def _get_num_backwards(system_transition_vector: Tokens):
    """
    Number of backward-looking tokens in a vector of tokens
    """
    return len(system_transition_vector) - _get_num_forwards(system_transition_vector)


class _ArrayMap:
    """
    """
    #[
    def __init__(self) -> NoReturn:
        self.lhs = ([], [])
        self.rhs = ([], [])

    def __len__(self: Self) -> int:
        return len(self.lhs[0])

    def append(
        self,
        lhs: tuple[int, int], 
        rhs: tuple[int, int]
    ) -> NoReturn:
        """
        """
        self.lhs = (self.lhs[0]+[lhs[0]], self.lhs[1]+[lhs[1]])
        self.rhs = (self.rhs[0]+[rhs[0]], self.rhs[1]+[rhs[1]])

    def merge_with(
        self,
        other: Self,
    ) -> NoReturn:
        """
        """
        self.lhs = (self.lhs[0]+other.lhs[0], self.lhs[1]+other.lhs[1])
        self.rhs = (self.rhs[0]+other.rhs[0], self.rhs[1]+other.rhs[1])

    def offset(
        self,
        lhs_row: int, 
        rhs_row_offset: int,
    ) -> NoReturn:
        """
        """
        self.lhs = ([lhs_row if i is not None else None for i in self.lhs[0]], self.lhs[1])
        self.rhs = ([i+rhs_row_offset if i is not None else None for i in self.rhs[0]], self.rhs[1])
    #]

    def remove_nones(self) -> NoReturn:
        #[
        zipped_pruned = [
            i for i in zip(self.lhs[0], self.lhs[1], self.rhs[0], self.rhs[1])
            if i[0] is not None
        ]
        unzipped_pruned = list(zip(*zipped_pruned))
        self.lhs = (list(unzipped_pruned[0]), list(unzipped_pruned[1]))
        self.rhs = (list(unzipped_pruned[2]), list(unzipped_pruned[3]))
        #]


def _vstack_maps(maps: Iterable[_ArrayMap]) -> _ArrayMap:
    """
    """
    #[
    stacked_map = _ArrayMap()
    rhs_row_offset = 0
    for lhs_row, m in enumerate(maps):
        m.offset(lhs_row, rhs_row_offset)
        rhs_row_offset += len(m)
        stacked_map.merge_with(m)
    return stacked_map
    #]


@dataclasses.dataclass
class SystemMap:
    """
    """
    #[
    A: _ArrayMap | None = None
    B: _ArrayMap | None = None
    C: None = None
    D: _ArrayMap | None = None
    dynid_A: numpy.ndarray | None = None
    dynid_B: numpy.ndarray | None = None
    F: _ArrayMap | None = None
    G: _ArrayMap | None = None
    H: None = None
    J: _ArrayMap | None = None


    def __init__(
        self,
        system_vectors: SystemVectors,
    ) -> NoReturn:
        """
        """
        # Transition equations

        self.A = _vstack_maps(
            _create_map_for_equation(system_vectors.transition_eid_to_wrt_tokens[eqn.id], system_vectors.transition_variables, )
            for eqn in system_vectors.transition_equations
        )

        lagged = [ t.lag() for t in system_vectors.transition_variables ]
        lagged = [ t if t not in system_vectors.transition_variables else None for t in lagged ]
        self.B = _vstack_maps(
            _create_map_for_equation(system_vectors.transition_eid_to_wrt_tokens[eqn.id], lagged, )
            for eqn in system_vectors.transition_equations
        )

        self.A.remove_nones()
        self.B.remove_nones()

        self.dynid_A, self.dynid_B = _create_dynid_matrices(system_vectors.transition_variables)

        self.D = _vstack_maps(
            _create_map_for_equation(system_vectors.transition_eid_to_wrt_tokens[eqn.id], system_vectors.transition_shocks, )
            for eqn in system_vectors.transition_equations
        )

        # Measurement equations

        self.F = _vstack_maps(
            _create_map_for_equation(system_vectors.measurement_eid_to_wrt_tokens[eqn.id], system_vectors.measurement_variables, )
            for eqn in system_vectors.measurement_equations
        )

        self.G = _vstack_maps(
            _create_map_for_equation(system_vectors.measurement_eid_to_wrt_tokens[eqn.id], system_vectors.transition_variables, )
            for eqn in system_vectors.measurement_equations
        )

        self.J = _vstack_maps(
            _create_map_for_equation(system_vectors.measurement_eid_to_wrt_tokens[eqn.id], system_vectors.measurement_shocks, )
            for eqn in system_vectors.measurement_equations
        )
    #]


def _create_map_for_equation(
    wrt: Tokens,
    vector: Tokens,
) -> _ArrayMap:
    """
    """
    #[
    map = _ArrayMap()
    for i, t in enumerate(wrt):
        if t in vector:
            map.append((0, vector.index(t)), (i, 0))
        else:
            map.append((None, None), (None, None))
    return map
    #]


def _create_dynid_matrices(system_transition_vector: Tokens):
    """
    Create dynamic identity matrix for unsolved system
    """
    #[
    num_columns = len(system_transition_vector)
    max_shifts = get_some_shifts_by_quantities(system_transition_vector, max)
    index_A = ([], [])
    index_B = ([], [])
    row_count = 0
    for i, t in enumerate(system_transition_vector):
        if t.shift==max_shifts[t.qid]:
            continue
        j = system_transition_vector.index(t.lead())
        index_A[0].append(row_count)
        index_A[1].append(i)
        index_B[0].append(row_count)
        index_B[1].append(j)
        row_count += 1
    dynid_A = numpy.zeros((row_count, num_columns), dtype=float)
    dynid_B = numpy.zeros((row_count, num_columns), dtype=float)
    dynid_A[index_A] = 1
    dynid_B[index_B] = -1
    return dynid_A, dynid_B
    #]


