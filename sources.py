"""
Model source
"""

#[
from __future__ import annotations

from IPython import embed
import re
import collections
from typing import NoReturn, TypeAlias
from collections.abc import Iterable

from .equations import (
    EquationKind, Equation,
)
from .quantities import (
    QuantityKind, Quantity,
)
#]


QuantityInput: TypeAlias = tuple[str, str]
EquationInput: TypeAlias = tuple[str, tuple[str, str]]


class Source:
    """
    """
    #[
    def __init__(self, /) -> NoReturn:
        self.quantities = []
        self.dynamic_equations = []
        self.steady_equations = []
        self.log_variables = []
        self.all_but = []
        self.sealed = False


    def seal(self, /) -> NoReturn:
        _check_unique_names(qty.human for qty in self.quantities)
        self.quantities = _reorder_by_kind(self.quantities)
        self.dynamic_equations = _reorder_by_kind(self.dynamic_equations)
        self.steady_equations = _reorder_by_kind(self.steady_equations)
        self.quantities = _stamp_id(self.quantities)
        self.dynamic_equations = _stamp_id(self.dynamic_equations)
        self.steady_equations = _stamp_id(self.steady_equations)
        self._populate_logly()
        self.sealed = True


    @property
    def num_quantities(self, /) -> int:
        return len(self.quantities)


    @property
    def all_names(self, /) -> Iterable[str]:
        return [ qty.human for qty in self.quantities ]


    def add_parameters(self, names: Iterable[QuantityInput] | None) -> NoReturn:
        self._add_quantities(names, QuantityKind.PARAMETER)


    def add_exogenous_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> NoReturn:
        self._add_quantities(quantity_inputs, QuantityKind.EXOGENOUS_VARIABLE)


    def add_transition_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> NoReturn:
        self._add_quantities(quantity_inputs, QuantityKind.TRANSITION_VARIABLE)


    def add_transition_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> NoReturn:
        self._add_quantities(quantity_inputs, QuantityKind.TRANSITION_SHOCK)


    def add_transition_equations(self, equation_inputs: Iterable[EquationInput] | None) -> NoReturn:
        self._add_equations(equation_inputs, EquationKind.TRANSITION_EQUATION)


    def add_measurement_variables(self, quantity_inputs: Iterable[EquationInput] | None) -> NoReturn:
        self._add_quantities(quantity_inputs, QuantityKind.MEASUREMENT_VARIABLE)


    def add_measurement_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> NoReturn:
        self._add_quantities(quantity_inputs, QuantityKind.MEASUREMENT_SHOCK)


    def add_measurement_equations(self, equation_inputs: Iterable[QuantityInput] | None) -> NoReturn:
        self._add_equations(equation_inputs, EquationKind.MEASUREMENT_EQUATION)


    def add_log_variables(self, log_variable_inputs: tuple[str, Iterable[str]] | None) -> NoReturn:
        flag = log_variable_inputs[0]
        log_variables = log_variable_inputs[1]
        self.all_but.append(flag=="all-but")
        if log_variables:
            self.log_variables += log_variables


    def _add_quantities(self, quantity_inputs: Iterable[QuantityInput] | None, kind: QuantityKind) -> NoReturn:
        if not quantity_inputs:
            return
        offset = self.quantities[-1].entry + 1 if self.quantities else 0
        self.quantities = self.quantities + [
            Quantity(id=None, human=q[1].strip(), kind=kind, logly=None, descript=q[0].strip(), entry=i)
            for i, q in enumerate(quantity_inputs, start=offset)
        ]


    def _add_equations(self, equation_inputs: Iterable[EquationInput] | None, kind: EquationKind, /) -> NoReturn:
        if not equation_inputs:
            return
        offset = self.dynamic_equations[-1].entry + 1 if self.dynamic_equations else 0
        self.dynamic_equations = self.dynamic_equations + [
            Equation(id=None, human=_handle_white_spaces(ein[1][0]), kind=kind, descript=ein[0].strip(), entry=i)
            for i, ein in enumerate(equation_inputs, start=offset)
        ]
        self.steady_equations = self.steady_equations + [
            Equation(id=None, human=_handle_white_spaces(ein[1][1] if ein[1][1] else ein[1][0]), kind=kind, descript=ein[0].strip(), entry=i)
            for i, ein in enumerate(equation_inputs, start=offset)
        ]


    def _populate_logly(self, /) -> NoReturn:
        default_logly = self._resolve_default_logly()
        qid_to_logly = { 
            qty.id: default_logly if qty.human not in self.log_variables else not default_logly
            for qty in self.quantities
            if qty.kind in QuantityKind.LOGLY_VARIABLE
        }
        self.quantities = [
            qty.set_logly(qid_to_logly.get(qty.id, None))
            for qty in self.quantities
        ]


    def _is_logly_consistent(self, /) -> bool:
        return all(self.all_but) or all(not x for x in self.all_but) if self.all_but else True


    def _resolve_default_logly(self, /) -> bool:
        if not self._is_logly_consistent():
            raise Exception("Inconsistent use of !all-but in !log-variables")
        return self.all_but.pop() if self.all_but else False


    @classmethod
    def from_lists(
        cls,
        /,
        transition_variables: Iterable[str], 
        transition_equations: Iterable[str], 
        transition_shocks: Iterable[str] | None = None,
        measurement_variables: Iterable[str] | None = None,
        measurement_equations: Iterable[str] | None = None,
        measurement_shocks: Iterable[str] | None = None,
        parameters: Iterable[str] | None = None,
        log_variables: Iterable[str] | None = None,
        all_but: bool = False,
    ) -> Self:
        """
        """
        self = Source()
        self.add_transition_variables(transition_variables)
        self.add_transition_equations(transition_equations)
        self.add_transition_shocks(transition_shocks)
        self.add_measurement_variables(measurement_variables)
        self.add_measurement_equations(measurement_equations)
        self.add_measurement_shocks(measurement_shocks)
        self.add_parameters(parameters)
        self.all_but = all_but
        self.add_log_variables(log_variables)
        self.seal()
        return self
    #]



def _check_unique_names(names: Iterable[str]) -> NoReturn:
    """
    """
    #[
    name_counter = collections.Counter(names)
    if any(c>1 for c in name_counter.values()):
        duplicates = ( n for n, c in name_counter.items() if c>1 )
        raise Exception("Duplicate names " + ", ".join(duplicates))
    #]


def _handle_white_spaces(x: str) -> str:
    return re.sub(r"[\s\n\r]+", " ", x)


def _reorder_by_kind(items: Iterable, /) -> Iterable:
    return sorted(items, key=lambda x: x.kind.value)


def _stamp_id(items: Iterable, /) -> Iterable:
    return [ i.set_id(_id) for _id, i in enumerate(items) ]

