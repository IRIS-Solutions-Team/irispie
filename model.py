
from __future__ import annotations

from typing import Iterable, Optional, Callable
from enum import Enum, auto
from collections import namedtuple, Counter
from abc import ABC, abstractmethod

from numpy import array, log, exp, nan_to_num
from re import Match, sub


class QuantityKind(Enum):
    TRANSITION_VARIABLE = auto()
    TRANSITION_SHOCK = auto()
    MEASUREMENT_VARIABLE = auto()
    MEASUREMENT_SHOCK = auto()
    PARAMETER = auto()

    def shocks() -> set[str]:
        return (QuantityKind.TRANSITION_SHOCK, QuantityKind.MEASUREMENT_SHOCK)


class EquationKind(Enum):
    TRANSITION_EQUATION = auto()


Quantity = namedtuple("Quantity", ["id", "name", "kind"])
Equation = namedtuple("Equation", ["id", "input", "kind"])


def _check_unique_quantity_names(existing_names: list[str], adding_names: list[str]) -> None:
    name_counter = Counter(existing_names + adding_names)
    if any([c>1 for c in name_counter.values()]):
        duplicates = [n for n, c in name_counter.items() if c>1]
        raise Exception("Duplicate names " + ", ".join(duplicates))


class ModelSource:

    def __init__(self):
        self._quantities: list[Quantity] = []
        self._equations: list[Equation] = []

    @property
    def name_to_id(self) -> dict[str, int]:
        return { q.name: q.id for q in self._quantities }

    @property
    def id_to_name(self) -> dict[int, str]:
        return { q.id: q.name for q in self._quantities }

    @property
    def num_quantities(self) -> int:
        return len(self._quantities)

    @property
    def all_names(self) -> list[str]:
        return [ q.name for q in self._quantities ]

    def add_parameters(self, names: Optional[Iterable[str]]) -> None:
        self._add_quantities(names, QuantityKind.PARAMETER)

    def add_transition_variables(self, names: Iterable[str]) -> None:
        self._add_quantities(names, QuantityKind.TRANSITION_VARIABLE)

    def add_transition_shocks(self, names: Iterable[str]) -> None:
        self._add_quantities(names, QuantityKind.TRANSITION_SHOCK)

    def add_transition_equations(self, inputs: Iterable[str]) -> None:
        self._add_equations(inputs, EquationKind.TRANSITION_EQUATION)

    def _add_quantities(self, names: Iterable[str], kind: QuantityKind) -> None:
        if not names: return
        _check_unique_quantity_names([q.name for q in self._quantities], names)
        offset = len(self._quantities)
        self._quantities = self._quantities + [
            Quantity(id=id, name=name.replace(" ", ""), kind=kind) for id, name in enumerate(names, start=offset)
        ]

    def _add_equations(self, inputs: Iterable[str], kind: QuantityKind) -> None:
        offset = len(self._equations)
        self._equations = self._equations + [ Equation(id=id, input=input.replace(" ", ""), kind=kind) for id, input in enumerate(inputs, start=offset) ]

    def _get_names_to_assign(self, name_value: dict[str, float]) -> set[str]:
        names_shocks = [ q.name for q in self._quantities if q.kind in QuantityKind.shocks() ]
        names_to_assign = set(name_value.keys()).intersection(self.all_names).difference(names_shocks)
        return names_to_assign

    def _get_pos_quantity_kinds(self, kinds: set[QuantityKind]) -> set[int]:
        return set( q.id for q in self._quantities if q.kind in kinds )


class Evaluator(ABC):

    def __init__(self):
        self._equations: Optional[list[str]] = None
        self._function_string: Optional[str] = None
        self._function: Optional[Callable] = None

    @staticmethod
    @abstractmethod
    def _replace_name_in_input(*args):
        pass

    @staticmethod
    @abstractmethod
    def _create_function_string(*args):
        pass

    def _parse_equation(self, input: str, name_to_id: dict[str, int]) -> str:
        pattern = r"\b([a-zA-Z]\w*)\b({[-+\d]+})?(?!\()"
        input = input.replace("^", "**")
        lhs_rhs = input.split('=')
        input = '-(' + lhs_rhs[0] + ')+' + lhs_rhs[1] if len(lhs_rhs)==2 else input
        return sub(pattern, lambda match: self._replace_name_in_input(match, name_to_id, input), input)

    @classmethod
    def from_model_source(cls, ms: ModelSource) -> Evaluator:
        self = cls()
        self._equations = [ self._parse_equation(e.input, ms.name_to_id) for e in ms._equations ]
        self._function_string = self._create_function_string(self._equations) 
        self._function = eval(self._function_string)
        return self


class SteadyEvaluator(Evaluator):

    @staticmethod
    def _replace_name_in_input(match: [Match], name_to_id: dict[str, int], input: str) -> str:
        name = match.group(1)
        if (id := name_to_id.get(name)) is None:
            raise Exception(f"Unknown name {name} in equation {input}")
        return f"x[{id}]"

    @staticmethod
    def _create_function_string(equations: list[str]) -> str:
        return f"lambda x: array([{','.join(equations)}]).T"



class Variant():

    def __init__(self, num_names: int=0) -> None:
        self._values = [None]*num_names

    def assign(self, name_value: dict[str, float], names_to_assign: list[str], name_to_id: dict[str, int]) -> None:
        for name in names_to_assign:
            self._values[name_to_id[name]] = name_value[name]

    def _assign_auto_values(self, pos: set[int], auto_value: float) -> None:
        for p in pos: self._values[p] = auto_value

    @classmethod
    def from_model_source(cls, ms: ModelSource) -> Variant:
        self = cls(ms.num_quantities)
        return self


class Model():

    def __init__(self):
        self._model_source: Optional(ModelSource) = None
        self._variants: list[Variant] = []
        self._steady_evaluator: Optional(SteadyEvaluator) = None

    def assign(self, name_value: dict[str, float]) -> set[str]:
        names_assigned = self._model_source._get_names_to_assign(name_value)
        name_to_id = self._model_source.name_to_id
        self._variants[0].assign(name_value, names_assigned, name_to_id)
        return names_assigned

    @property
    def steady_evaluator(self) -> tuple(Callable, str):
        return self._steady_evaluator._function, self._steady_evaluator._function_string

    def create_steady_vector(self, variant: int=0, missing: Optional[float]=1) -> array:
        steady_vector = array([self._variants[variant]._values], dtype=float).T
        if missing:
            nan_to_num(steady_vector, nan=missing, copy=False)
        return steady_vector

    def _assign_auto_values(self) -> None:
        pos_shocks = self._model_source._get_pos_quantity_kinds(QuantityKind.shocks())
        for v in self._variants:
            v._assign_auto_values(pos_shocks, 0)

    @classmethod
    def from_lists( 
        cls,
        transition_variables: list[str], 
        transition_equations: list[str], 
        transition_shocks: Optional(list[str])=None,
        parameters: Optional(list[str])=None,
    ) -> Model:

        model_source = ModelSource()
        model_source.add_transition_variables(transition_variables)
        model_source.add_transition_equations(transition_equations)
        model_source.add_transition_shocks(transition_shocks)
        model_source.add_parameters(parameters)

        self = cls()
        self._model_source = model_source
        self._variants = [ Variant.from_model_source(model_source) ]
        self._steady_evaluator = SteadyEvaluator.from_model_source(model_source)
        self._assign_auto_values()

        return self

