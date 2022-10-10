
from __future__ import annotations
from typing import NamedTuple, Optional, Callable, Iterable
from numbers import Number
from enum import Flag, auto
from collections import Counter
from dataclasses import dataclass
from copy import deepcopy

from numpy import (
    ndarray, array, copy, tile, reshape,
    zeros, log, exp, nan_to_num,
)



from .incidence import (
    get_max_shift, get_min_shift
)

from .equations import (
    EquationKind, Equation,
    finalize_equations_from_humans, generate_all_tokens
)



class QuantityKind(Flag):
    UNSPECIFIED = auto()
    TRANSITION_VARIABLE = auto()
    TRANSITION_SHOCK = auto()
    MEASUREMENT_VARIABLE = auto()
    MEASUREMENT_SHOCK = auto()
    PARAMETER = auto()

    VARIABLE = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE
    SHOCK = TRANSITION_SHOCK | MEASUREMENT_SHOCK
    FIRST_ORDER_SYSTEM = VARIABLE | SHOCK



class EvaluatorKind(Flag):
    STEADY = auto()
    STACKED = auto()
    PERIOD = auto()



@dataclass
class Quantity():
    """
    """
    id: int
    human: str

    kind: QuantityKind = QuantityKind.UNSPECIFIED
    log_flag: bool = False



def _check_unique_names(names: Iterable[str]) -> None:
    name_counter = Counter(names)
    if any( c>1 for c in name_counter.values() ):
        duplicates = ( n for n, c in name_counter.items() if c>1 )
        raise Exception("Duplicate names " + ", ".join(duplicates))



class ModelSource:
    """
    """
    def __init__(self):
        self.quantities: list[Quantity] = []
        self.equations: EquationsT = []
        self.sealed = False


    def seal(self):
        _check_unique_names(qty.human for qty in self.quantities)
        self.sealed = True

    @property
    def name_to_id(self) -> dict[str, int]:
        return { qty.human: qty.id for qty in self.quantities }

    @property
    def id_to_name(self) -> dict[int, str]:
        return { qty.id: qty.human for qty in self.quantities }

    @property
    def id_to_log_flag(self) -> dict[int, bool]:
        return { q.id: q.log_flag for q in self.quantities }

    @property
    def num_quantities(self) -> int:
        return len(self.quantities)

    @property
    def all_names(self) -> list[str]:
        return [ qty.human for qty in self.quantities ]

    @property
    def max_shift(self) -> int:
        return get_max_shift(generate_all_tokens(self.equations))

    @property
    def min_shift(self) -> int:
        return get_min_shift(generate_all_tokens(self.equations))

    def add_parameters(self, names: Optional[list[str]]) -> None:
        self._add_quantities(names, QuantityKind.PARAMETER)

    def add_transition_variables(self, names: Optional[list[str]]) -> None:
        self._add_quantities(names, QuantityKind.TRANSITION_VARIABLE)

    def add_transition_shocks(self, names: Optional[list[str]]) -> None:
        self._add_quantities(names, QuantityKind.TRANSITION_SHOCK)

    def add_transition_equations(self, humans: Optional[list[str]]) -> None:
        self._add_equations(humans, EquationKind.TRANSITION_EQUATION)

    def _add_quantities(self, names: Optional[list[str]], kind: QuantityKind) -> None:
        if not names:
            return
        offset = len(self.quantities)
        self.quantities = self.quantities + [
            Quantity(id=id, human=name.replace(" ", ""), kind=kind) for id, name in enumerate(names, start=offset)
        ]


    def _add_equations(self, humans: Optional[list[str]], kind: EquationKind) -> None:
        if not humans:
            return
        offset = len(self.equations)
        self.equations = self.equations + [ Equation(id=id, human=human.replace(" ", ""), kind=kind) for id, human in enumerate(humans, start=offset) ]


    def _get_names_to_assign(self, name_value: dict[str, float]) -> set[str]:
        names_shocks = [ qty.human for qty in self.quantities if qty.kind in QuantityKind.SHOCK ]
        names_to_assign = set(name_value.keys()).intersection(self.all_names).difference(names_shocks)
        return names_to_assign


    def get_quantity_ids_by_kind(self, kind: QuantityKind) -> list[int]:
        return [ q.id for q in self.quantities if q.kind in kind ]


    def get_quantity_names_by_kind(self, kind: QuantityKind) -> list[str]:
        return [ qty.human for qty in self.quantities if qty.kind in kind ]


    def get_equation_ids_by_kind(self, kind: EquationKind) -> list[int]:
        return [ eqn.id for eqn in self.equations if eqn.kind in kind ]
    #)



class Evaluator:
    pass


class SteadyEvaluator(Evaluator):
    """
    """
    kind = EvaluatorKind.STEADY

    def __init__(self, model: Model) -> None:
        self._equations: list[Equation] = []
        self._quantities: list[Quantity] = []
        self._resolve_incidences(model)
        self._quantity_ids: list[int] = self._get_quantity_ids()
        self._t_zero = -model.min_shift
        self._create_evaluator_function()
        self._create_incidence_matrix()
        num_columns = self._t_zero + 1 + model.max_shift
        self._x: ndarray = model.create_steady_array(num_columns)
        self._z0: ndarray = self._z_from_x()


    @property
    def init(self) -> ndarray:
        return copy(self._z0)

    @property
    def quantities_solved(self) -> list[str]:
        return [ qty.human for qty in self._quantities ]

    @property
    def equations_solved(self) -> list[str]:
        return [ eqn.human for eqn in self._equations ]

    @property
    def num_equations(self) -> int:
        return len(self._equations)

    @property
    def num_quantities(self) -> int:
        return len(self._quantities)

    def eval(self, current: Optional[ndarray]=None):
        x = self._x_from_z(current)
        return self._func(x)

    def _resolve_incidences(self, model: Model) -> None:
        equation_ids = model._model_source.get_equation_ids_by_kind(EquationKind.EQUATION)
        quantity_ids = model._model_source.get_quantity_ids_by_kind(QuantityKind.VARIABLE)
        self._equations = [ eqn for eqn in model._model_source.equations if eqn.id in equation_ids ]
        self._quantities = [ qty for qty in model._model_source.quantities if qty.id in quantity_ids ]

    def _create_evaluator_function(self) -> None:
        xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = ",".join(xtrings)
        self._func = eval(f"lambda x, t={self._t_zero}: array([{func_string}], dtype=float)")

    def _get_quantity_ids(self) -> list[int]:
        return [ qty.id for qty in self._quantities ]

    def _create_incidence_matrix(self) -> None:
        matrix = zeros((self.num_equations, self.num_quantities), dtype=int)
        for row_index, eqn in enumerate(self._equations):
            column_indices = list(set( tok.quantity_id for tok in eqn.incidence if tok.quantity_id in self._quantity_ids ))
            matrix[row_index, column_indices] = 1
        self.incidence_matrix = matrix

    def _z_from_x(self):
        return self._x[self._quantity_ids, self._t_zero]

    def _x_from_z(self, z: ndarray) -> ndarray:
        x = copy(self._x)
        if z is not None:
            x[self._quantity_ids, :] = reshape(z, (-1,1))
        return x


class Variant:
    """
    """
    def __init__(self, num_names: int=0) -> None:
        self._values: list[Optional[Number]] = [None]*num_names

    def assign(self, name_value: dict[str, Number], names_to_assign: list[str], name_to_id: dict[str, int]) -> None:
        for name in names_to_assign:
            self._values[name_to_id[name]] = name_value[name]

    def _assign_auto_values(self, pos: set[int], auto_value: Number) -> None:
        for p in pos: self._values[p] = auto_value

    @classmethod
    def from_model_source(cls, ms: ModelSource) -> Variant:
        self = cls(ms.num_quantities)
        return self



class Model():
    """
    """
    def __init__(self):
        self._model_source: Optional(ModelSource) = None
        self._variants: list[Variant] = []

    def assign(self, name_value: dict[str, Number]) -> set[str]:
        """
        """
        names_assigned = self._model_source._get_names_to_assign(name_value)
        name_to_id = self._model_source.name_to_id
        self._variants[0].assign(name_value, names_assigned, name_to_id)
        return self, names_assigned

    def change_num_variants(self, new_num: int) -> None:
        """
        """
        if new_num<self.num_variants:
            self._shrink_num_variants(new_num)
        elif new_num>self.num_variants:
            self._expand_num_variants(new_num)

    @property
    def num_variants(self) -> int:
        return len(self._variants)

    @property
    def max_shift(self) -> int:
        return self._model_source.max_shift

    @property
    def min_shift(self) -> int:
        return self._model_source.min_shift

    def create_steady_evaluator(self) -> SteadyEvaluator:
        return SteadyEvaluator(self)

    def create_steady_array(self, num_columns: int=1, variant: int=0, missing: Optional[float]=None) -> ndarray:
        steady_vector = array([self._variants[variant]._values], dtype=float).transpose()
        if missing:
            nan_to_num(steady_vector, nan=missing, copy=False)
        steady_array = tile(steady_vector, (1, num_columns))
        return steady_array

    def _assign_auto_values(self) -> None:
        pos_shocks = self._model_source.get_quantity_ids_by_kind(QuantityKind.SHOCK)
        for v in self._variants:
            v._assign_auto_values(pos_shocks, 0)

    def _shrink_num_variants(self, new_num: int) -> None:
        if new_num<1:
            Exception('Number of variants must be one or more')
        self._variants = self._variants[0:new_num]

    def _expand_num_variants(self, new_num: int) -> None:
        for i in range(self.num_variants, new_num):
            self._variants.append(deepcopy(self._variants[-1]))

    @classmethod
    def from_lists( 
        cls,
        transition_variables: list[str], 
        transition_equations: list[str], 
        transition_shocks: Optional[list[str]]=None,
        parameters: Optional[list[str]]=None,
    ) -> Model:

        model_source = ModelSource()
        model_source.add_transition_variables(transition_variables)
        model_source.add_transition_equations(transition_equations)
        model_source.add_transition_shocks(transition_shocks)
        model_source.add_parameters(parameters)
        model_source.seal()

        self = cls()
        self._model_source = model_source
        self._variants = [ Variant.from_model_source(model_source) ]

        finalize_equations_from_humans(model_source.equations, model_source.name_to_id)

        self._assign_auto_values()

        return self
#)



