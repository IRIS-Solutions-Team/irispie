
from typing import (
    Self,
)

from dataclasses import (
    dataclass,
)

from numpy import (
    ndarray, zeros,
)


@dataclass
class _ArrayMap:
    lhs: tuple[list[int], list[int]]
    rhs: tuple[list[int], list[int]]

    def __init__(self: Self) -> None:
        self.lhs = ([], [])
        self.rhs = ([], [])

    def append(
        self: Self, 
        lhs: tuple[int, int], 
        rhs: tuple[int, int]
    ) -> None:
        """
        """
        self.lhs[0].append(lhs[0])
        self.lhs[1].append(lhs[1])
        self.rhs[0].append(rhs[0])
        self.rhs[1].append(rhs[1])

    def merge_with(
        self: Self,
        other: Self,
    ) -> None:
        """
        """
        self.lhs[0] += other.lhs[0]
        self.lhs[1] += other.lhs[1]
        self.rhs[0] += other.rhs[0]
        self.rhs[1] += other.rhs[1]

    def offset(
        self: Self,
        equation: int, 
        row_offset: int,
    ) -> None:
        """
        """
        self.lhs[0] = [ equation for _ in self.lhs[0] ]
        self.rhs[0] = [ i+row_offset for i in self.rhs[0] ]


def _vstack_maps(maps: Iterable[_ArrayMap]) -> _ArrayMap:
    """
    """
    stacked_map = _ArrayMap()
    row_offset = 0
    for equation, m in enumerate(maps):
        m.offset(equation, row_offset)
        stacked_map.merge_wtih(m)
    return stacked_map


@dataclass
class SystemMap:
    """
    """
    # Transition equations
    # A E[x] + B E[x{-1}] + C + D e = 0
    A: _ArrayMap | None = None
    B: _ArrayMap | None = None
    C: _ArrayMap | None = None
    D: _ArrayMap | None = None
    # Dynamic identities
    dynid_A: ndarray | None = None
    dynid_B: ndarray | None = None
    # Measurement equations
    # E y + F x + G + H u = 0
    E: _ArrayMap | None = None
    F: _ArrayMap | None = None
    G: _ArrayMap | None = None
    H: _ArrayMap | None = None

    def populate(
        self: Self, 
        system_vectors: FirstOrderSystemVectors,
        wrt_tokens_transition_equations: Tokens,
        wrt_tokens_measurement_equations: Tokens,
    ) -> None:
        """
        """
        self.A = _vstack(
            _create_map_for_equation(system_vectors.transition_variables, wrt) 
            for wrt in wrt_tokens_transition_equations
        )
        min_lags = get_some_shifts_by_quantities(system_vectors.transition_variables, min)
        min_lags = [ Token(quantity_id, shift) for quantity_id, shift in min_lags.items() ]
        transition_variables_min_lags_only = _nullify_except(system_vectors.transition_variables, min_lags)
        self.B = _vstack(
            _create_map_for_equation(transition_variables_min_lags_only, wrt) 
            for wrt in wrt_tokens_transition_equations
        )
        self.D = _vstack(
            _create_map_for_equation(system_vectors.transition_shocks, wrt) 
            for wrt in wrt_tokens_transition_equations
        )
        self.dynid_A, self.dynid_B = _create_dynid_matrices(system_vectors.transition_variables)


def _create_map_for_equation(
    vector: Tokens,
    wrt: Tokens,
) -> _ArrayMap:
    map = _ArrayMap()
    for i, t in enumerate(wrt):
        if t not in vector:
            continue
        j = vector.index(t)
        map.append((0, j), (i, 0))
    return map


def _nullify_except(vector: Tokens, exceptions: Tokens) -> Tokens:
    return [t if t is exceptions else None for t in vector ]


def _create_dynid_matrices(system_transition_vector: Tokens):
    """
    From system transition vector, create dynamic identity matrix for unsolved system
    """
    num_columns = len(system_transition_vector)
    max_shifts = get_some_shifts_by_quantities(system_transition_vector, max)
    index_A = ([], [])
    index_B = ([], [])
    row_count = 0
    for i, t in enumerate(system_transition_vector):
        if t.shift==max_shifts[t.quantity_id]:
            continue
        j = system_transition_vector.index(t.lead())
        index_A[0].append(row_count)
        index_A[1].append(i)
        index_A[0].append(row_count)
        index_A[1].append(j)
        row_count += 1
    dynid_A = zeros((row_count, num_columns), dtype=float)
    dynid_B = zeros((row_count, num_columns), dtype=float)
    dynid_A[index_A] = 1
    dynid_B[index_B] = -1
    return dynid_A, dynid_B


