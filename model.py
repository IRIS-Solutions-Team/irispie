
from __future__ import annotations

from typing import Iterable, Optional, Callable
from enum import Enum, auto
from collections import namedtuple, Counter
from abc import ABC, abstractmethod

from numpy import array, log, exp, nan_to_num, copy
from re import Match, sub, compile
from copy import deepcopy


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


_TINX_PATTERN = compile(r",t[+-]\d+")
_QUANTITY_NAME_PATTERN = compile(r"\b([a-zA-Z]\w*)\b({[-+\d]+})?(?!\()")


class QuantityKind(Enum):
    TRANSITION_VARIABLE = auto()
    TRANSITION_SHOCK = auto()
    MEASUREMENT_VARIABLE = auto()
    MEASUREMENT_SHOCK = auto()
    PARAMETER = auto()

    @staticmethod
    def variables() -> set[str]:
        return {QuantityKind.TRANSITION_VARIABLE, QuantityKind.MEASUREMENT_VARIABLE}

    @staticmethod
    def shocks() -> set[str]:
        return {QuantityKind.TRANSITION_SHOCK, QuantityKind.MEASUREMENT_SHOCK}



class EquationKind(Enum):
    TRANSITION_EQUATION = auto()


class EvaluatorKind(Enum):
    STEADY = auto()
    STACKED = auto()
    PERIOD = auto()


Quantity = namedtuple("Quantity", ["id", "name", "kind"])
Equation = namedtuple("Equation", ["id", "human", "kind"])
Incidence = namedtuple("Incidence", ["equation", "quantity", "shift"])


def _check_unique_quantity_names(existing_names: list[str], adding_names: list[str]) -> None:
    name_counter = Counter(existing_names + adding_names)
    if any([c>1 for c in name_counter.values()]):
        duplicates = [n for n, c in name_counter.items() if c>1]
        raise Exception("Duplicate names " + ", ".join(duplicates))


class ModelSource:
#(
    def __init__(self):
        self.quantities: list[Quantity] = []
        self.equations: list[Equation] = []

    @property
    def name_to_id(self) -> dict[str, int]:
        return { q.name: q.id for q in self.quantities }

    @property
    def id_to_name(self) -> dict[int, str]:
        return { q.id: q.name for q in self.quantities }

    @property
    def num_quantities(self) -> int:
        return len(self.quantities)

    @property
    def all_names(self) -> list[str]:
        return [ q.name for q in self.quantities ]

    def add_parameters(self, names: Optional[Iterable[str]]) -> None:
        self._add_quantities(names, QuantityKind.PARAMETER)

    def add_transition_variables(self, names: Iterable[str]) -> None:
        self._add_quantities(names, QuantityKind.TRANSITION_VARIABLE)

    def add_transition_shocks(self, names: Iterable[str]) -> None:
        self._add_quantities(names, QuantityKind.TRANSITION_SHOCK)

    def add_transition_equations(self, humans: Iterable[str]) -> None:
        self._add_equations(humans, EquationKind.TRANSITION_EQUATION)

    def _add_quantities(self, names: Iterable[str], kind: QuantityKind) -> None:
        if not names: return
        _check_unique_quantity_names([q.name for q in self.quantities], names)
        offset = len(self.quantities)
        self.quantities = self.quantities + [
            Quantity(id=id, name=name.replace(" ", ""), kind=kind) for id, name in enumerate(names, start=offset)
        ]

    def _add_equations(self, humans: Iterable[str], kind: EquationKind) -> None:
        offset = len(self.equations)
        self.equations = self.equations + [ Equation(id=id, human=human.replace(" ", ""), kind=kind) for id, human in enumerate(humans, start=offset) ]

    def _get_names_to_assign(self, name_value: dict[str, float]) -> set[str]:
        names_shocks = [ q.name for q in self.quantities if q.kind in QuantityKind.shocks() ]
        names_to_assign = set(name_value.keys()).intersection(self.all_names).difference(names_shocks)
        return names_to_assign

    def _get_pos_quantities_by_kind(self, kinds: set[QuantityKind]) -> set[int]:
        return set( q.id for q in self.quantities if q.kind in kinds )
#)



class Evaluator:
#(
    def __init__(self, model: Model, kind: EvaluatorKind) -> None:
        self._kind: EvaluatorKind = kind
        self._function_string: Optional[str] = None
        self._function: Optional[Callable] = None
        self._x: array = model.create_steady_vector()
        self._index: list[int] = []
        self._init: Optional[array] = None
        self._create_function(model)
        self._create_input(model)
        self._names = model._model_source.all_names


    @property
    def init(self) -> array:
        return copy(self._init)


    @property
    def unknowns(self) -> list[str]:
        return [ self._names[i] for i in self._index ]


    def eval(self, current: Optional[array]=None):
        x = copy(self._x)
        if current is not None:
            x[self._index, :] = current
        return self._function(x)


    def _create_function(self, model: Model) -> None:
        if self._kind is EvaluatorKind.STEADY:
            tinx_replace = ",:"
            lambda_string = "lambda x: array([{equations_string}], dtype=float)"
        else:
            raise NotImplementedError

        equations_string = _TINX_PATTERN.sub(tinx_replace, ",".join(model._parsed_equations))
        self._function_string = lambda_string.format(equations_string=equations_string)
        self._function = eval(self._function_string)


    def _create_input(self, model: Model) -> None:
        if self._kind is EvaluatorKind.STEADY:
            self._index = [ qty.id for qty in model._model_source.quantities if qty.kind in QuantityKind.variables() ]
            self._init = self._x[self._index, :]
        else:
            raise NotImplementedError
#)


class EquationParser():
#(

    def __init__(self, ms: ModelSource) -> None:
        self._equations: list[Equation] = ms.equations
        self._name_to_id: dict[str, int] = ms.name_to_id

        self._incidence_by_eqtn: list[set[Incidence]] = []
        self.incidence = list[Incidence]
        self.parsed_equations: list[str] = []
        self._error_log: list[tuple(str, str)] = []
        self._run()


    def _run(self):
        self.parsed_equations = [ self._parse_equation(eqn) for eqn in self._equations ]
        if self._error_log:
            raise UndeclaredName(self._error_log)
        self.incidence = [ j for i in self._incidence_by_eqtn for j in i ]


    def _parse_equation(self, eqn: Equation) -> str:

        def _replace_name_in_human(match: [Match]) -> Equation:
            name = match.group(1)
            qty_id = self._name_to_id.get(name)
            qty_shift = self._resolve_shift_str(match.group(2))
            if qty_id is None:
                self._error_log.append((name, match.string))
            eqn_incidence.add(Incidence(eqn.id, qty_id, qty_shift))
            return f"x[{qty_id},t{qty_shift:+g}]"

        eqn_incidence = set()
        parsed_equation = _QUANTITY_NAME_PATTERN.sub(_replace_name_in_human, eqn.human)
        self._incidence_by_eqtn.append(eqn_incidence)

        parsed_equation = parsed_equation.replace("^", "**")
        if len(lhs_rhs := parsed_equation.split("="))==2:
            parsed_equation = lhs_rhs[0] + "-(" + lhs_rhs[1] + ")" 
        return parsed_equation 


    @staticmethod
    def _resolve_shift_str(shift_str: str) -> int:
        return int(shift_str.replace("{", "",).replace("}", "")) if shift_str is not None else 0


#)



class Variant():
#(
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
#)



class Model():
#(
    def __init__(self):
        self._model_source: Optional(ModelSource) = None
        self._variants: list[Variant] = []
        self._steady_equations: Optional(list[str]) = None
        self._steady_incidence: Optional(list[tuple[int, int, int]]) = None

    def assign(self, name_value: dict[str, float]) -> set[str]:
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
    def steady_evaluator(self) -> Evaluator:
        return Evaluator(self, EvaluatorKind.STEADY)

    def create_steady_vector(self, variant: int=0, missing: Optional[float]=None) -> array:
        steady_vector = array([self._variants[variant]._values], dtype=float).transpose()
        if missing:
            nan_to_num(steady_vector, nan=missing, copy=False)
        return steady_vector

    def _assign_auto_values(self) -> None:
        pos_shocks = self._model_source._get_pos_quantities_by_kind(QuantityKind.shocks())
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

        equation_parser = EquationParser(model_source)
        self._parsed_equations: list[str] = equation_parser.parsed_equations
        self._incidence: list[Incidence] = equation_parser.incidence

        self._assign_auto_values()

        return self
#)

