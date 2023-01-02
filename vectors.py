
#( External imports
from numpy import (
    ndarray, 
)

from typing import (
    Iterable, Self,
)

from dataclasses import (
    dataclass,
)
#)

#( Internal imports
from .incidence import (
    Token, Tokens,
    get_some_shifts_by_quantities, generate_tokens_of_kind,
)

from .quantities import (
    QuantityKind
)
#)


@dataclass
class FirstOrderSystemVectors:
    transition_variables: Tokens | None = None
    transition_shocks: Tokens | None = None 
    measurement_variables: Tokens | None = None
    measurement_shocks: Tokens | None = None 

    @classmethod
    def from_all_tokens(cls: type, all_tokens: Tokens, id_to_kind: dict[int, str]) -> Self:
        self = cls()
        tokens_transition_variables = set(generate_tokens_of_kind(all_tokens, id_to_kind, QuantityKind.TRANSITION_VARIABLE))
        self.transition_variables = _sort_tokens(_create_system_transition_vector(tokens_transition_variables))
        self.transition_shocks = _sort_tokens(generate_tokens_of_kind(all_tokens, id_to_kind, QuantityKind.TRANSITION_SHOCK))
        self.measurement_variables = _sort_tokens(generate_tokens_of_kind(all_tokens, id_to_kind, QuantityKind.MEASUREMENT_VARIABLE))
        self.measurement_shocks = _sort_tokens(generate_tokens_of_kind(all_tokens, id_to_kind, QuantityKind.MEASUREMENT_SHOCK))
        return self


@dataclass
class FirstOrderSolutionVectors:
    transition_variables: Tokens | None = None
    transition_shocks: Tokens | None = None 
    measurement_variables: Tokens | None = None
    measurement_shocks: Tokens | None = None 

    @classmethod
    def from_system_vectors(cls: type, system_vectors: FirstOrderSystemVectors) -> Self:
        self = cls()
        self.transition_variables = _solution_vector_from_system_vector(system_vectors.transition_variables)
        self.transition_shocks = system_vectors.transition_shocks
        self.measurement_variables = system_vectors.measurement_variables
        self.measurement_shocks = system_vectors.measurement_shocks
        return self


def print_vector(
    tokens: Tokens,
    id_to_name: dict[int, str]
) -> Iterable[str]:
    """
    Create list of printed tokens
    """
    return [ t.print(id_to_name) for t in tokens ]


def _sort_tokens(tokens: Tokens) -> Tokens:
    """
    Sort tokens by id
    """
    return sorted(tokens, key=lambda x: (-x.shift, x.quantity_id))


def _create_system_transition_vector(tokens_transition_variables: Tokens) -> Tokens:
    """
    From tokens of transition variables, create vector of transition variables
    along columns of matrix A in unsolved system
    """
    shift_ranges = get_some_shifts_by_quantities(tokens_transition_variables, lambda x: reversed(range(0, max(x)+1)))
    sorted_ids = sorted(t.quantity_id for t in tokens_transition_variables)
    vector_for_id = lambda quantity_id: [Token(quantity_id, sh) for sh in shift_ranges[quantity_id]]
    vector = [vector_for_id(quantity_id) for quantity_id in sorted_ids]
    vector = list(chain.from_iterable(vector))
    return vector


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


