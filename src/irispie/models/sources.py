"""
Model source
"""

#[
from __future__ import annotations

import re as _re
import collections
from typing import (TypeAlias, Literal, )
from collections.abc import (Iterable, )

from ..parsers import (preparser as _pp, model_source_parser as _pm, )
from .. import (equations as _eq, quantities as _qu, )
#]


QuantityInput: TypeAlias = tuple[str, str]
EquationInput: TypeAlias = tuple[str, tuple[str, str]]


STD_PREFIX = "std_"
_STD_DESCRIPTION = "(Std) "


LOGLY_VARIABLE = (
    _qu.QuantityKind.TRANSITION_VARIABLE
    | _qu.QuantityKind.MEASUREMENT_VARIABLE
    | _qu.QuantityKind.EXOGENOUS_VARIABLE
)


class ModelSource:
    """
    """
    #[
    __slots__ = (
        "quantities", "dynamic_equations", "steady_equations",
        "log_variables", "all_but", "context", "sealed",
    )
    def __init__(self, /) -> None:
        self.quantities = []
        self.dynamic_equations = []
        self.steady_equations = []
        self.log_variables = []
        self.all_but = []
        self.context = None
        self.sealed = False


    def seal(self, /) -> None:
        _check_unique_names(qty.human for qty in self.quantities)
        self._add_stds()
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


    def add_parameters(self, names: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(names, _qu.QuantityKind.PARAMETER)


    def add_exogenous_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _qu.QuantityKind.EXOGENOUS_VARIABLE)


    def add_transition_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _qu.QuantityKind.TRANSITION_VARIABLE)


    def add_transition_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _qu.QuantityKind.TRANSITION_SHOCK)


    def add_transition_equations(self, equation_inputs: Iterable[EquationInput] | None) -> None:
        self._add_equations(equation_inputs, _eq.EquationKind.TRANSITION_EQUATION)


    def add_measurement_variables(self, quantity_inputs: Iterable[EquationInput] | None) -> None:
        self._add_quantities(quantity_inputs, _qu.QuantityKind.MEASUREMENT_VARIABLE)


    def add_measurement_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _qu.QuantityKind.MEASUREMENT_SHOCK)


    def add_measurement_equations(self, equation_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_equations(equation_inputs, _eq.EquationKind.MEASUREMENT_EQUATION)


    def add_all_but(self, all_but_input: str | None) -> None:
        if not all_but_input:
            return
        self.all_but.append(all_but_input)


    def add_log_variables(self, log_variable_inputs: Iterable[str] | None, /, ) -> None:
        if not log_variable_inputs:
            return
        self.log_variables += log_variable_inputs


    def _add_quantities(self, quantity_inputs: Iterable[QuantityInput] | None, kind: _qu.QuantityKind) -> None:
        if not quantity_inputs:
            return
        offset = self.quantities[-1].entry + 1 if self.quantities else 0
        self.quantities = self.quantities + [
            _qu.Quantity(id=None, human=q[1].strip(), kind=kind, logly=None, description=q[0].strip(), entry=i)
            for i, q in enumerate(quantity_inputs, start=offset)
        ]


    def _add_equations(self, equation_inputs: Iterable[EquationInput] | None, kind: _eq.EquationKind, /) -> None:
        if not equation_inputs:
            return
        offset = self.dynamic_equations[-1].entry + 1 if self.dynamic_equations else 0
        self.dynamic_equations = self.dynamic_equations + [
            _eq.Equation(id=None, human=_handle_white_spaces(ein[1][0]), kind=kind, description=ein[0].strip(), entry=i)
            for i, ein in enumerate(equation_inputs, start=offset)
        ]
        self.steady_equations = self.steady_equations + [
            _eq.Equation(id=None, human=_handle_white_spaces(ein[1][1] if ein[1][1] else ein[1][0]), kind=kind, description=ein[0].strip(), entry=i)
            for i, ein in enumerate(equation_inputs, start=offset)
        ]


    def _add_stds(self) -> None:
        def _create_std_input_from_shock(shock):
            description = _STD_DESCRIPTION + (shock.description if shock.description else shock.human)
            human = STD_PREFIX + shock.human
            return(description, human)
        #
        transition_shocks = (q for q in self.quantities if q.kind in _qu.QuantityKind.TRANSITION_SHOCK)
        transition_std_inputs = (_create_std_input_from_shock(i) for i in transition_shocks)
        self._add_quantities(transition_std_inputs, _qu.QuantityKind.TRANSITION_STD)
        #
        measurement_shocks = (q for q in self.quantities if q.kind in _qu.QuantityKind.MEASUREMENT_SHOCK)
        measurement_std_inputs = (_create_std_input_from_shock(i) for i in measurement_shocks)
        self._add_quantities(measurement_std_inputs, _qu.QuantityKind.MEASUREMENT_STD)


    def _populate_logly(self, /) -> None:
        default_logly = self._resolve_default_logly()
        qid_to_logly = { 
            qty.id: default_logly if qty.human not in self.log_variables else not default_logly
            for qty in self.quantities
            if qty.kind in LOGLY_VARIABLE
        }
        self.quantities = [
            qty.set_logly(qid_to_logly.get(qty.id, None))
            for qty in self.quantities
        ]


    def _is_logly_consistent(self, /) -> bool:
        return all(a=="all-but" for a in self.all_but) or all(a=="" for a in self.all_but) if self.all_but else True


    def _resolve_default_logly(self, /) -> bool:
        if not self._is_logly_consistent():
            raise Exception("Inconsistent use of !all-but in !log-variables")
        return self.all_but.pop()=="all-but" if self.all_but else False


    @classmethod
    def from_lists(
        cls,
        /,
        transition_variables: Iterable[QuantityInput],
        transition_equations: Iterable[EquationInput], 
        transition_shocks: Iterable[QuantityInput] | None = None,
        measurement_variables: Iterable[QuantityInput] | None = None,
        measurement_equations: Iterable[EquationInput] | None = None,
        measurement_shocks: Iterable[QuantityInput] | None = None,
        parameters: Iterable[QuantityInput] | None = None,
        exogenous_variables: Iterable[QuantityInput] | None = None,
        log_variables: Iterable[str] | None = None,
        seal: bool = True,
    ) -> Self:
        """
        """
        self = ModelSource()
        self.add_transition_variables(transition_variables)
        self.add_transition_equations(transition_equations)
        self.add_transition_shocks(transition_shocks)
        self.add_measurement_variables(measurement_variables)
        self.add_measurement_equations(measurement_equations)
        self.add_measurement_shocks(measurement_shocks)
        self.add_parameters(parameters)
        self.add_exogenous_variables(exogenous_variables)
        self.add_log_variables(log_variables)
        if seal: self.seal()
        return self

    @classmethod
    def from_string(
        cls,
        source_string: str,
        /,
        context: dict | None = None,
        save_preparsed: str = "",
        seal: bool = True,
    ) -> tuple[Self, dict]:
        """
        """
        preparsed_string, preparser_info = _pp.from_string(
            source_string, context, save_preparsed=save_preparsed, 
        )
        parsed_content = _pm.from_string(preparsed_string)
        #
        self = ModelSource()
        self.add_transition_variables(parsed_content.get("transition-variables"))
        self.add_transition_shocks(parsed_content.get("transition-shocks"))
        self.add_transition_equations(parsed_content.get("transition-equations"))
        self.add_measurement_variables(parsed_content.get("measurement-variables"))
        self.add_measurement_shocks(parsed_content.get("measurement-shocks"))
        self.add_measurement_equations(parsed_content.get("measurement-equations"))
        self.add_parameters(parsed_content.get("parameters"))
        self.add_exogenous_variables(parsed_content.get("exogenous-variables"))
        self.add_log_variables(parsed_content.get("log-variables"))
        for i in parsed_content.get("all-but", []):
            self.add_all_but(i)
        self.context = preparser_info["context"]
        #
        if seal: 
            self.seal()
        #
        return self, preparser_info
    #]


def _check_unique_names(names: Iterable[str], /) -> None:
    """
    """
    #[
    name_counter = collections.Counter(names)
    if any(c>1 for c in name_counter.values()):
        duplicates = ( n for n, c in name_counter.items() if c>1 )
        raise Exception("Duplicate names " + ", ".join(duplicates))
    #]


def _handle_white_spaces(x: str, /) -> str:
    return _re.sub(r"[\s\n\r]+", "", x)


def _reorder_by_kind(items: Iterable, /) -> Iterable:
    return sorted(items, key=lambda x: (x.kind.value, x.entry))


def _stamp_id(items: Iterable, /) -> Iterable:
    return [ i.set_id(_id) for _id, i in enumerate(items) ]


