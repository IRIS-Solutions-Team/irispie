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
    generate_eids_by_kind, sort_equations,
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
#]


@dataclasses.dataclass
class SystemVectors:
    """
    Vectors of quantities and equation ids in first-order system matrices
    """
    #[
    transition_eids: Iterable[int] | None = None
    measurement_eids: Iterable[int] | None = None
    eid_to_wrt_tokens: dict[int, Tokens] | None = None

    transition_variables: Tokens | None = None
    transition_shocks: Tokens | None = None 
    measurement_variables: Tokens | None = None
    measurement_shocks: Tokens | None = None 

    shape_AB_excl_dynid: tuple[int, int] | None = None
    shape_C_excl_dynid: tuple[int, int] | None = None
    shape_D_excl_dynid: tuple[int, int] | None = None
    shape_F: tuple[int, int] | None = None
    shape_G: tuple[int, int] | None = None
    shape_H: tuple[int, int] | None = None

    def __init__(self, equations: Equations, quantities: Quantities) -> NoReturn:
        """
        Construct system vectors from a list of equations and a list of quantities
        """
        self.transition_eids = sorted([eqn.id for eqn in equations if eqn.kind in EquationKind.TRANSITION_EQUATION])
        self.measurement_eids = sorted([eqn.id for eqn in equations if eqn.kind in EquationKind.MEASUREMENT_EQUATION])
        qid_to_kind = create_qid_to_kind(quantities)
        all_tokens = set(generate_all_tokens_from_equations(equations))
        all_wrt_tokens = set(generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.SYSTEM_QUANTITY))
        self.eid_to_wrt_tokens = create_eid_to_wrt_tokens(equations, all_wrt_tokens)

        tokens_transition_variables = generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.TRANSITION_VARIABLE)
        self.transition_variables = sort_tokens(_create_system_transition_vector(tokens_transition_variables))
        self.transition_shocks = sort_tokens(generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.TRANSITION_SHOCK))
        self.measurement_variables = sort_tokens(generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.MEASUREMENT_VARIABLE))
        self.measurement_shocks = sort_tokens(generate_tokens_of_kinds(all_tokens, qid_to_kind, QuantityKind.MEASUREMENT_SHOCK))

        self.shape_AB_excl_dynid = (len(self.transition_eids), len(self.transition_variables))
        self.shape_C_excl_dynid = (len(self.transition_eids), 1)
        self.shape_D_excl_dynid = (len(self.transition_eids), len(self.transition_shocks))

        self.shape_F = (len(self.measurement_eids), len(self.measurement_variables))
        self.shape_G = (len(self.measurement_eids), len(self.transition_variables))
        self.shape_H = (len(self.measurement_eids), 1)
        self.shape_J = (len(self.measurement_eids), len(self.measurement_shocks))

    def generate_system_equations_from_equations(
        self,
        equations: Equations,
        /
    ) -> Equations:
        eid_to_equation = { eqn.id:eqn for eqn in equations }
        system_eids = self.transition_eids + self.measurement_eids
        return ( eid_to_equation[eid] for eid in system_eids )
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


@dataclasses.dataclass
class SystemMap:
    """
    """
    #[
    A: ArrayMap | None = None
    B: ArrayMap | None = None
    C: None = None
    D: ArrayMap | None = None
    dynid_A: numpy.ndarray | None = None
    dynid_B: numpy.ndarray | None = None
    dynid_C: numpy.ndarray | None = None
    dynid_D: numpy.ndarray | None = None

    F: ArrayMap | None = None
    G: ArrayMap | None = None
    H: None = None
    J: ArrayMap | None = None

    def __init__(
        self,
        system_vectors: SystemVectors,
    ) -> NoReturn:
        """
        """

        system_eids = system_vectors.transition_eids + system_vectors.measurement_eids
        rhs_offsets = list(itertools.accumulate(
            len(system_vectors.eid_to_wrt_tokens[eid]) 
            for eid in system_eids
        ))
        rhs_offsets.pop()
        rhs_offsets.insert(0, 0)
        eid_to_rhs_offset = dict(zip(system_eids, rhs_offsets))

        # Transition equations

        self.A = vstack_array_maps(
            ArrayMap.for_equation(
                system_vectors.eid_to_wrt_tokens[eid],
                system_vectors.transition_variables,
                eid_to_rhs_offset[eid],
                lhs_row,
            )
            for lhs_row, eid in enumerate(system_vectors.transition_eids)
        )

        lagged_transition_variables = [ t.shifted(-1) for t in system_vectors.transition_variables ]
        lagged_transition_variables = [ 
            t if t not in system_vectors.transition_variables else None 
            for t in lagged_transition_variables 
        ]
        self.B = vstack_array_maps(
            ArrayMap.for_equation(
                system_vectors.eid_to_wrt_tokens[eid],
                lagged_transition_variables, 
                eid_to_rhs_offset[eid],
                lhs_row,
            )
            for lhs_row, eid in enumerate(system_vectors.transition_eids)
        )

        self.A.remove_nones()
        self.B.remove_nones()

        num_dynid_rows = len(system_vectors.transition_variables) - len(system_vectors.transition_eids)
        self.dynid_A, self.dynid_B = _create_dynid_matrices(system_vectors.transition_variables, )
        self.dynid_C = numpy.zeros((num_dynid_rows, system_vectors.shape_C_excl_dynid[1]), dtype=float, )
        self.dynid_D = numpy.zeros((num_dynid_rows, system_vectors.shape_D_excl_dynid[1]), dtype=float, )

        self.D = vstack_array_maps(
            ArrayMap.for_equation(
                system_vectors.eid_to_wrt_tokens[eid],
                system_vectors.transition_shocks,
                eid_to_rhs_offset[eid],
                lhs_row,
            )
            for lhs_row, eid in enumerate(system_vectors.transition_eids)
        )

        # Measurement equations

        self.F = vstack_array_maps(
            ArrayMap.for_equation(
                system_vectors.eid_to_wrt_tokens[eid],
                system_vectors.measurement_variables, 
                eid_to_rhs_offset[eid],
                lhs_row,
            )
            for lhs_row, eid in enumerate(system_vectors.measurement_eids)
        )

        self.G = vstack_array_maps(
            ArrayMap.for_equation(
                system_vectors.eid_to_wrt_tokens[eid],
                system_vectors.transition_variables, 
                eid_to_rhs_offset[eid],
                lhs_row,
            )
            for lhs_row, eid in enumerate(system_vectors.measurement_eids)
        )

        self.J = vstack_array_maps(
            ArrayMap.for_equation(
                system_vectors.eid_to_wrt_tokens[eid],
                system_vectors.measurement_shocks, 
                eid_to_rhs_offset[eid],
                lhs_row,
            )
            for lhs_row, eid in enumerate(system_vectors.measurement_eids)
        )
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
        j = system_transition_vector.index(t.shifted(+1))
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


class ArrayMap:
    """
    """
    NOT_FOUND = ((None, None), (None, None))
    #[
    def __init__(self) -> NoReturn:
        self.lhs = ([], [])
        self.rhs = ([], [])

    def __len__(self: Self) -> int:
        return len(self.lhs[0])

    def append(
        self,
        lhs: tuple[int, int], 
        rhs: tuple[int, int],
        /
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

    def remove_nones(self) -> NoReturn:
        """
        """
        zipped_pruned = [
            i for i in zip(self.lhs[0], self.lhs[1], self.rhs[0], self.rhs[1])
            if i[0] is not None
        ]
        unzipped_pruned = list(zip(*zipped_pruned))
        self.lhs = (list(unzipped_pruned[0]), list(unzipped_pruned[1]))
        self.rhs = (list(unzipped_pruned[2]), list(unzipped_pruned[3]))

    @classmethod
    def for_equation(
        cls,
        tokens_in_equation_on_rhs: Tokens,
        tokens_in_columns_on_lhs: Tokens,
        rhs_offset: int,
        lhs_row: int,
    ) -> ArrayMap:
        """
        """
        map = cls()
        for rhs_row, t in enumerate(tokens_in_equation_on_rhs, start=rhs_offset):
            if t in tokens_in_columns_on_lhs:
                lhs_column = tokens_in_columns_on_lhs.index(t)
                map.append((lhs_row, lhs_column), (rhs_row, 0))
        return map
    #]


def vstack_array_maps(maps: Iterable[ArrayMap]) -> ArrayMap:
    """
    """
    #[
    stacked_map = ArrayMap()
    for m in maps:
        stacked_map.merge_with(m)
    return stacked_map
    #]


