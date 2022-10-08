
from __future__ import annotations
from typing import NamedTuple, Optional, Callable
from numbers import Number
from enum import Enum, Flag, auto
from collections import Counter
from abc import ABC, abstractmethod

from numpy import ndarray, array, zeros, log, exp, nan_to_num, copy
from re import sub, compile, escape
from copy import deepcopy

from .incidence import Incidence, Token, get_max_shift, get_min_shift
from .eqman import EquationKind, Equation, xtrings_from_equations



class UndeclaredName(Exception):
#(
    def __init__(self, errors: list[tuple[str, str]]) -> None:
        message = "".join([
            f"\n*** Undeclared or mistyped name '{e[0]}' in equation {e[1]}"
            for e in errors
        ])
        message = "\n" + message + "\n"
        super().__init__(message)
#)



_TINX_PATTERN = compile(r"\[t([+-]\d+)?\]" + escape(EQUATION_ID_PLACEHOLDER))



class QuantityKind(Flag):
#(
    UNSPECIFIED = auto()
    TRANSITION_VARIABLE = auto()
    TRANSITION_SHOCK = auto()
    MEASUREMENT_VARIABLE = auto()
    MEASUREMENT_SHOCK = auto()
    PARAMETER = auto()

    VARIABLE = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE
    SHOCK = TRANSITION_SHOCK | MEASUREMENT_SHOCK
    FIRST_ORDER_SYSTEM = VARIABLE | SHOCK
#)



class EvaluatorKind(Flag):
    STEADY = auto()
    STACKED = auto()
    PERIOD = auto()



class Quantity(NamedTuple):
    id: int
    name: str
    kind: QuantityKind = QuantityKind.UNSPECIFIED
    log_flag: bool = False


def _check_unique_names(names: list[str]) -> None:
    name_counter = Counter(names)
    if any([c>1 for c in name_counter.values()]):
        duplicates = [n for n, c in name_counter.items() if c>1]
        raise Exception("Duplicate names " + ", ".join(duplicates))


class ModelSource:
    #(
    def __init__(self):
        self.quantities: list[Quantity] = []
        self.equations: EquationsT = []
        self.sealed = False


    def seal(self):
        _check_unique_names(self.quantities)
        self.sealed = True


    @property
    def name_to_id(self) -> dict[str, int]:
        return { q.name: q.id for q in self.quantities }

    @property
    def id_to_name(self) -> dict[int, str]:
        return { q.id: q.name for q in self.quantities }

    @property
    def id_to_log_flag(self) -> dict[int, bool]:
        return { q.id: q.log_flag for q in self.quantities }

    @property
    def num_quantities(self) -> int:
        return len(self.quantities)

    @property
    def all_names(self) -> list[str]:
        return [ q.name for q in self.quantities ]

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
            Quantity(id=id, name=name.replace(" ", ""), kind=kind) for id, name in enumerate(names, start=offset)
        ]


    def _add_equations(self, humans: Optional[list[str]], kind: EquationKind) -> None:
        if not humans:
            return
        offset = len(self.equations)
        self.equations = self.equations + [ Equation(id=id, human=human.replace(" ", ""), kind=kind) for id, human in enumerate(humans, start=offset) ]


    def _get_names_to_assign(self, name_value: dict[str, float]) -> set[str]:
        names_shocks = [ q.name for q in self.quantities if q.kind in QuantityKind.SHOCK ]
        names_to_assign = set(name_value.keys()).intersection(self.all_names).difference(names_shocks)
        return names_to_assign


    def get_quantity_ids_by_kind(self, kind: QuantityKind) -> list[int]:
        return [ q.id for q in self.quantities if q.kind in kind ]


    def get_quantity_names_by_kind(self, kind: QuantityKind) -> list[str]:
        return [ q.name for q in self.quantities if q.kind in kind ]


    def get_equation_ids_by_kind(self, kind: EquationKind) -> list[int]:
        return [ e.id for e in self.equations if e.kind in kind ]
    #)



class Evaluator:
    pass


class SteadyEvaluator(Evaluator):
#(
    def __init__(self, model: Model) -> None:
        self._model_source: ModelSource = model._model_source

        self._equation_ids: list[int] = []
        self._quantity_ids: list[int] = []
        self._incidences: list[Incidence] = []
        self._resolve_incidence(model)

        select_equations = [ model.parsed_equations[id] for id in self._equation_ids ]
        self._func = SteadyEvaluator._create_evaluator_function(select_equations)

        self._x: ndarray = model.create_steady_vector()
        self._init: ndarray = self._x[self._quantity_ids, :]


    @property
    def init(self) -> ndarray:
        return copy(self._init)


    @property
    def quantities_solved(self) -> list[str]:
        return [ self._model_source.quantities[i].name for i in self._quantity_ids ]


    @property
    def equations_solved(self) -> list[str]:
        return [ self._model_source.equations[i].human for i in self._equation_ids ]


    @property
    def num_equations(self) -> int:
        return len(self._equation_ids)


    @property
    def num_quantities(self) -> int:
        return len(self._quantity_ids)


    @property
    def incidence_matrix(self) -> ndarray:
        row_indices = [ self._equation_ids.index(inc.equation_id) for inc in self._incidences if inc.equation_id ]
        column_indices = [ self._quantity_ids.index(inc.token.quantity_id) for inc in self._incidences if inc.equation_id ]
        matrix = zeros((self.num_equations, self.num_quantities), dtype=int)
        matrix[row_indices, column_indices] = 1
        return matrix


    def eval(self, current: Optional[ndarray]=None):
        x = copy(self._x)
        if current is not None:
            x[self._quantity_ids, :] = current
        return self._func(x)


    def _resolve_incidence(self, model: Model) -> None:
        self._equation_ids = model._model_source.get_equation_ids_by_kind(EquationKind.EQUATION)
        self._quantity_ids = model._model_source.get_quantity_ids_by_kind(QuantityKind.VARIABLE)
        self._incidences = [
            Incidence(inc.equation_id, Token(inc.token.quantity_id, 0))
            for inc in model._incidences
            if inc.equation_id in self._equation_ids and inc.token.quantity_id in self._quantity_ids
        ]
        self._incidences = list(set(self._incidences))


    @staticmethod
    def _create_evaluator_function(equations: list[str]) -> None:
        # Replace x[0][t-1][_] with x[0][:]
        tinx_replace = "[:]"
        equations = ",".join(equations)
        equations_string = _TINX_PATTERN.sub(tinx_replace, equations)
        func_string = f"lambda x: array([{equations_string}], dtype=float)"
        return eval(func_string)
#)



class Variant:
#(
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
#)



class Model():
#(
    def __init__(self):
        self._model_source: Optional(ModelSource) = None
        self._variants: list[Variant] = []
        self.parsed_equations: list[str] = []
        self._incidences: set[Incidence] = set()


    def assign(self, name_value: dict[str, Number]) -> set[str]:
        names_assigned = self._model_source._get_names_to_assign(name_value)
        name_to_id = self._model_source.name_to_id
        self._variants[0].assign(name_value, names_assigned, name_to_id)
        return names_assigned


    def change_num_variants(self, new_num: int) -> None:
        if new_num<self.num_variants:
            self._shrink_num_variants(new_num)
        elif new_num>self.num_variants:
            self._expand_num_variants(new_num)

    @property
    def num_variants(self) -> int:
        return len(self._variants)

    @property
    def steady_evaluator(self) -> SteadyEvaluator:
        return SteadyEvaluator(self)

    @property
    def max_shift(self) -> int:
        return get_max_shift(self._incidences)

    @property
    def min_shift(self) -> int:
        return get_min_shift(self._incidences)

    def create_steady_vector(self, variant: int=0, missing: Optional[float]=None) -> ndarray:
        steady_vector = array([self._variants[variant]._values], dtype=float).transpose()
        if missing:
            nan_to_num(steady_vector, nan=missing, copy=False)
        return steady_vector

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

        self.parsed_equations, self._incidences = xtrings_from_equations(model_source.equations, model_source.name_to_id)

        self._assign_auto_values()

        return self
#)



