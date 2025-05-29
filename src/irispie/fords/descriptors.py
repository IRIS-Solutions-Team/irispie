"""
Descriptor for first-order systems and solutions
=================================================

Unsolved system
----------------

$$
A E[x_{t}] + B E[x_{t-1}] + C + D u_{t} + E v_{t} = 0 \\
F y_{t} + G x_{t} + H + J w_{t} = 0
$$

## State-space solution

$$
x_{t} = T x_{t-1} + K + R v_{t} \\
y_{t} = Z x_{t} + D + H w_{t}
$$


Table of contents
------------------

# List classes and standalone functions in order of appearance

* `Descriptor`
* `SystemVectors`
* `SolutionVectors`
* `HumanSolutionVectors`
* `_TRANSITION_SYSTEM_QUANTITY`
* `_MEASUREMENT_SYSTEM_QUANTITY`
* `_SYSTEM_QUANTITY`
* `_create_system_transition_vector`
* `_solution_vector_from_system_vector`
* `_get_num_forwards`
* `_get_num_backwards`
* `SystemMap`
* `_create_dynid_matrices`
* `_adjust_for_measurement_equations`
* `_custom_order_equations_by_eids`
* `Squid`

"""


#[
from __future__ import annotations

from typing import (Self, Any, Protocol, )
from types import (SimpleNamespace, )
from collections.abc import (Iterable, )
import itertools as _it
import functools as _ft
import dataclasses as _dc
import numpy as _np

from ..incidences.main import (Token, )
from ..incidences import main as _incidence
from .. import equations as _equations
from ..equations import Equation, EquationKind
from .. import quantities as _quantities
from ..aldi.differentiators import (AtomFactoryProtocol, Context, )
from ..aldi import maps as _maps
from .. import sources as _sources
#]


# Implement AtomFactoryProtocol


class _AtomFactory:
    """
    """
    #[

    @staticmethod
    def create_data_index_for_token(token: Token, ) -> tuple[int, int]:
        return token.qid, token.shift,

    @staticmethod
    def create_diff_for_token(
        token: Token,
        wrt_tokens: tuple[Token, ...],
    ) -> _np.ndarray:
        try:
            index = wrt_tokens.index(token, )
        except ValueError:
            return 0
        diff = _np.zeros((len(wrt_tokens), 1, ), dtype=float, )
        diff[index] = 1
        return diff

    @staticmethod
    def get_diff_shape(wrt_tokens: tuple[Token, ...]) -> tuple[int, int]:
        return len(wrt_tokens), 1,

    #]


class Descriptor:
    """
    Descriptor of first-order systems and solutions, and aldi Atom factory provider
    """
    #[

    __slots__ = (
        # Unsolved system vectors
        "system_vectors",

        # Solution vectors
        "solution_vectors",

        # Mapping from aldi results to unsolved matrices
        "system_map",

        # Aldi context for creating unsolved system matrices
        "aldi_context",
    )

    def __init__(
        self,
        equations: Iterable[Equation],
        quantities: Iterable[_quantities.Quantity],
        context: dict[str, Any] | None,
    ) -> None:
        """
        """
        equations = tuple(
            i for i in equations
            if i.kind == EquationKind.TRANSITION_EQUATION or i.kind == EquationKind.MEASUREMENT_EQUATION
        )
        self.system_vectors = SystemVectors(equations, quantities)
        self.solution_vectors = SolutionVectors.from_system_vectors(self.system_vectors)
        self.system_map = SystemMap(self.system_vectors)
        system_equations = _custom_order_equations_by_eids(
            equations,
            self.system_vectors.transition_eids
            + self.system_vectors.measurement_eids,
        )
        #
        # Create the evaluation context for the algorithmic differentiator
        self.aldi_context = Context(
            _AtomFactory,
            system_equations,
            eid_to_wrts=self.system_vectors.eid_to_wrt_tokens,
            qid_to_logly=_quantities.create_qid_to_logly(quantities),
            context=context,
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
    eid_to_wrt_tokens: dict[int, Iterable[Token]] | None = None
    #
    transition_variables: Iterable[Token] | None = None
    transition_variables_are_logly: list[bool] | None = None
    true_initials: list[bool] | None = None,
    unanticipated_shocks: tuple[Token, ...] | None = None
    anticipated_shocks: tuple[Token, ...] | None = None
    measurement_variables: Iterable[Token] | None = None
    measurement_variables_are_logly: list[bool] | None = None
    measurement_shocks: tuple[Token, ...] | None = None
    std_unanticipated_shocks: tuple[Token, ...] | None = None
    std_measurement_shocks: tuple[Token, ...] | None = None
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
        equations: Iterable[Equation],
        quantities: Iterable[_quantities.Quantity],
    ) -> None:
        """
        Construct system vectors from a list of equations and a list of quantities
        """
        qid_to_logly = _quantities.create_qid_to_logly(quantities, )
        #
        self.transition_eids \
            = sorted([eqn.id for eqn in equations if eqn.kind in EquationKind.TRANSITION_EQUATION])
        self.measurement_eids \
            = sorted([eqn.id for eqn in equations if eqn.kind in EquationKind.MEASUREMENT_EQUATION])
        qid_to_kind = _quantities.create_qid_to_kind(quantities, )
        #
        # Collect all tokens from equations but also add zero-shifted tokens
        # created from all variables, shocks, and parameters to catch any names
        # not used in equations (shocks, parameters)
        all_tokens = _collect_all_tokens(equations, quantities, )
        #
        all_wrt_tokens = set(_incidence.generate_tokens_of_kinds(all_tokens, qid_to_kind, _SYSTEM_QUANTITY))
        self.eid_to_wrt_tokens = _equations.create_eid_to_wrt_tokens(equations, all_wrt_tokens)
        #
        actual_transition_variable_tokens \
            = set(_incidence.generate_tokens_of_kinds(all_tokens, qid_to_kind, _quantities.QuantityKind.TRANSITION_VARIABLE))
        #
        # Make adjustment for transition variables in measurement
        # equations: each x(t-k) in measurement needs to be in the current
        # dated (LHS) vector of transition variables; this is done by
        # pretending x(t-k-1) is needed
        #
        adjusted_transition_variable_tokens \
            = _adjust_for_measurement_equations(actual_transition_variable_tokens, equations, qid_to_kind)
        #
        self.transition_variables \
            = _incidence.sort_tokens(_create_system_transition_vector(adjusted_transition_variable_tokens))
        self.transition_variables_are_logly \
            = [ qid_to_logly[tok.qid] for tok in self.transition_variables ]
        self.populate_true_initials(actual_transition_variable_tokens, )
        #
        # Unanticipated shocks and stds
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
        # Measurement shocks and stds
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

    def populate_true_initials(
        self,
        actual_transition_variable_tokens: set[Token],
    ) -> None:
        """
        """
        # Get the maximum lag for each transition variable
        # actual_min_shifts[qid] = min(shifts) across all tokens with qid
        actual_min_shifts = _incidence.get_some_shift_by_quantities(actual_transition_variable_tokens, min, )
        def is_true_initial(token, ):
            shift = token.shift - 1
            min_shift = actual_min_shifts[token.qid]
            return min_shift <= shift < 0
        self.true_initials = tuple(
            is_true_initial(i) for i in self.transition_variables
        )
    #]


class SolutionVectors:
    """
    Vectors of quantities in first-order solution matrices
    """
    #[

    __slots__ = (
        "transition_variables",
        "unanticipated_shocks",
        "anticipated_shocks",
        "measurement_variables",
        "measurement_shocks",
        "true_initials",
    )

    def __init__(self, **kwargs, ) -> None:
        """
        """
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot, None, ), )

    @classmethod
    def from_system_vectors(
        klass,
        system_vectors: SystemVectors,
    ) -> None:
        """
        Construct solution vectors and initial conditions indicator
        """
        self = klass()
        self.transition_variables, self.true_initials = \
            _solution_vector_from_system_vector(system_vectors.transition_variables, system_vectors.true_initials, )
        self.unanticipated_shocks = tuple(system_vectors.unanticipated_shocks)
        self.anticipated_shocks = tuple(system_vectors.anticipated_shocks)
        self.measurement_variables = tuple(system_vectors.measurement_variables)
        self.measurement_shocks = tuple(system_vectors.measurement_shocks)
        return self

    def get_initials(self, ) -> list[Token]:
        """
        Get tokens representing required initial conditions
        """
        return list(_it.compress(self.transition_variables, self.true_initials))

    def get_curr_transition_indexes(self, ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        """
        lhs_rhs_tuples = [
            (t.qid, i)
            for i, t in enumerate(self.transition_variables, )
            if not t.shift
        ]
        return tuple(zip(*lhs_rhs_tuples))

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
    ) -> None:
        """
        """
        print_tokens = _ft.partial(_incidence.print_tokens, qid_to_name=qid_to_name, qid_to_logly=qid_to_logly, )
        for a in self.__slots__:
            setattr(self, a, print_tokens(getattr(solution_vectors, a)))

    #]


def _create_system_transition_vector(
    transition_variable_tokens: Iterable[Token],
) -> Iterable[Token]:
    """
    From tokens of transition variables, create vector of transition variables
    along columns of matrix A in unsolved system
    """
    #[
    transition_variable_tokens = set(transition_variable_tokens)
    min_shifts = _incidence.get_some_shift_by_quantities(transition_variable_tokens, lambda x: min(min(x), -1))
    max_shifts = _incidence.get_some_shift_by_quantities(transition_variable_tokens, max)
    #
    def list_for_qid(qid: int, ):
        return [Token(qid, sh) for sh in range(min_shifts[qid]+1, max_shifts[qid]+1)]
    unique_qids = set(t.qid for t in transition_variable_tokens)
    return _it.chain.from_iterable(list_for_qid(i) for i in unique_qids)
    #]


def _solution_vector_from_system_vector(
    system_transition_vector: Iterable[Token],
    true_initials: Iterable[bool],
) -> Iterable[Token]:
    """
    From sorted system vector, get vector of transition variables in solved
    system and the indicator of required initial conditions
    """
    num_forwards = _get_num_forwards(system_transition_vector)
    return (
        tuple(system_transition_vector[num_forwards:]),
        tuple(true_initials[num_forwards:]),
    )


def _get_num_forwards(system_transition_vector: Iterable[Token]):
    """
    Number of forward-looking tokens in a vector of tokens
    """
    return sum(1 for t in system_transition_vector if t.shift>0)


def _get_num_backwards(system_transition_vector: Iterable[Token]):
    """
    Number of backward-looking tokens in a vector of tokens
    """
    return len(system_transition_vector) - _get_num_forwards(system_transition_vector)


class SystemMap:
    """
    """
    #[

    __slots__ = (
        "A",
        "B",
        "C",
        "D",
        "E",
        "dynid_A",
        "dynid_B",
        "dynid_C",
        "dynid_D",
        "dynid_E",
        "F",
        "G",
        "H",
        "J",
    )

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
        self.A = _maps.ArrayMap.static(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.transition_variables,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        lagged_transition_variables = [
            t.shifted(-1)
            for t in system_vectors.transition_variables
        ]
        #
        lagged_transition_variables = [
            t if t not in system_vectors.transition_variables else None
            for t in lagged_transition_variables
        ]
        #
        self.B = _maps.ArrayMap.static(
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
        self.C = _maps.VectorMap.static(system_vectors.transition_eids, )
        #
        self.D = _maps.ArrayMap.static(
            system_vectors.transition_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.unanticipated_shocks,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.E = _maps.ArrayMap.static(
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
        self.F = _maps.ArrayMap.static(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.measurement_variables,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.G = _maps.ArrayMap.static(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.transition_variables,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self.H = _maps.VectorMap.static(system_vectors.measurement_eids, )
        #
        self.J = _maps.ArrayMap.static(
            system_vectors.measurement_eids,
            system_vectors.eid_to_wrt_tokens,
            system_vectors.measurement_shocks,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
    #]


def _create_dynid_matrices(system_transition_vector: Iterable[Token]):
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
    transition_variable_tokens: Iterable[_quantities.Quantity],
    equations: Iterable[Equation],
    qid_to_kind: dict[int, _quantities.QuantityKind],
) -> Iterable[_quantities.Quantity]:
    """
    """
    #[
    tokens_in_measurement_equations = _it.chain.from_iterable(e.incidence for e in equations if e.kind is EquationKind.MEASUREMENT_EQUATION)
    pretend_needed = [
        Token(t.qid, t.shift-1) for t in tokens_in_measurement_equations
        if qid_to_kind[t.qid] in _quantities.QuantityKind.TRANSITION_VARIABLE
    ]
    return set(transition_variable_tokens).union(pretend_needed)
    #]


def _custom_order_equations_by_eids(
    equations: Iterable[Equation],
    eids: list[int],
) -> tuple[Equation, ...]:
    """
    """
    #[
    eid_to_equation = { eqn.id: eqn for eqn in equations }
    return tuple( eid_to_equation[eid] for eid in eids )
    #]


class SquidableProtocol(Protocol, ):
    """
    """
    #[

    solution_vectors: SolutionVectors

    shock_qid_to_std_qid: dict[int, int] | None

    #]


class Squid:
    """
    """
    #[

    __slots__ = (
        "curr_xi_qids",
        "curr_xi_indexes",
        "y_qids",
        "u_qids",
        "v_qids",
        "w_qids",
        "std_u_qids",
        "std_v_qids",
        "std_w_qids",
        "num_xi",
    )

    def __init__(
        self,
        solution_vectors: SolutionVectors,
        shock_qid_to_std_qid: dict[int, int] | None = None,
    ) -> None:
        """
        """
        vec = solution_vectors
        #
        self.num_xi = len(vec.transition_variables)
        self.curr_xi_qids, self.curr_xi_indexes = vec.get_curr_transition_indexes()
        self.curr_xi_qids = tuple(self.curr_xi_qids)
        self.curr_xi_indexes = tuple(self.curr_xi_indexes)
        #
        self.y_qids = tuple(t.qid for t in vec.measurement_variables)
        self.u_qids = tuple(t.qid for t in vec.unanticipated_shocks)
        self.v_qids = tuple(t.qid for t in vec.anticipated_shocks)
        self.w_qids = tuple(t.qid for t in vec.measurement_shocks)
        #
        self.std_u_qids = ()
        self.std_v_qids = ()
        self.std_w_qids = ()
        #
        if shock_qid_to_std_qid is not None:
            self.std_u_qids = tuple(
                shock_qid_to_std_qid[t.qid]
                for t in vec.unanticipated_shocks
            )
            self.std_v_qids = tuple(
                shock_qid_to_std_qid[t.qid]
                for t in vec.anticipated_shocks
            )
            self.std_w_qids = tuple(
                shock_qid_to_std_qid[t.qid]
                for t in vec.measurement_shocks
            )

    @classmethod
    def from_squidable(
        klass,
        squidable: SquidableProtocol,
    ) -> Self:
        """
        """
        solution_vectors = squidable.solution_vectors
        shock_qid_to_std_qid = squidable.shock_qid_to_std_qid if squidable.shock_qid_to_std_qid else None
        return klass(solution_vectors, shock_qid_to_std_qid, )

    @property
    def num_curr_xi(self) -> int:
        """==Number of current-dated transition variables in transition vector=="""
        return len(self.curr_xi_qids)

    @property
    def num_u(self) -> int:
        """==Number of unanticipated shocks in system=="""
        return len(self.u_qids)

    @property
    def num_v(self) -> int:
        """==Number of anticipated shocks in system=="""
        return len(self.v_qids)

    @property
    def num_w(self) -> int:
        """==Number of measurement shocks in system=="""
        return len(self.w_qids)

    #]


def _collect_all_tokens(
    equations: Iterable[Equation],
    quantities: Iterable[_quantities.Quantity],
) -> set[Token]:
    """
    Combine all tokens from equations and zero-shifted tokens created from
    variables, shocks and parameters.
    """
    #[
    # Tokens from equations
    tokens_from_equations \
        = set(_equations.generate_all_tokens_from_equations(equations, ))
    #
    # Zero-shift tokens from all variables, shocks and parameters
    kind = _quantities.ANY_VARIABLE | _quantities.ANY_SHOCK | _quantities.PARAMETER
    tokens_from_quantities \
        = set(_incidence.generate_zero_shift_tokens_from_quantities(quantities, kind=kind))
    return tokens_from_equations | tokens_from_quantities
    #]

