"""
Model source
"""


#[
from __future__ import annotations

import re as _re
from typing import (Self, Type, TypeAlias, Literal, Protocol, )
from collections.abc import (Iterable, )

from . import pages as _pages
from . import equations as _equations
from . import quantities as _quantities
from .parsers import preparser as _preparser
from .parsers import models as _models
from .parsers import common as _common
#]


__all__ = (
    "ModelSource",
)


QuantityInput: TypeAlias = tuple[str, str, tuple[str, ...]]
EquationInput: TypeAlias = tuple[str, tuple[str, str], tuple[str, ...]]


LOGLY_VARIABLE = (
    _quantities.QuantityKind.TRANSITION_VARIABLE
    | _quantities.QuantityKind.MEASUREMENT_VARIABLE
    | _quantities.QuantityKind.EXOGENOUS_VARIABLE
)


LOGLY_VARIABLE_OR_ANY_SHOCK = (
    LOGLY_VARIABLE
    | _quantities.QuantityKind.ANY_SHOCK
)


class SourceMixinProtocol(Protocol, ):
    """
    """
    #[

    @classmethod
    def from_source(
        klass,
        source: ModelSource,
        /,
        **kwargs,
    ) -> Self:
        ...

    #]


def from_string(
    klass: type[SourceMixinProtocol],
    source_string: str,
    /,
    context: dict | None = None,
    save_preparsed: str = "",
    **kwargs,
) -> SourceMixinProtocol:
    """Create a new object from model source string"""
    #[
    source, info = ModelSource.from_string(
        source_string, context=context, save_preparsed=save_preparsed,
    )
    return klass.from_source(source, context=context, **kwargs, )
    #]


def from_file(
    klass: type[SourceMixinProtocol],
    source_files: str | Iterable[str],
    /,
    **kwargs,
) -> SourceMixinProtocol:
    """Create a new object from source file(s)"""
    #[
    source_string = _combine_source_files(source_files, )
    return from_string(klass, source_string, **kwargs, )
    #]


class ModelSource:
    """
    """
    #[

    __slots__ = (
        "quantities",
        "dynamic_equations",
        "steady_equations",
        "log_variables",
        "all_but",
        "context",
    )

    def __init__(self, /) -> None:
        self.quantities = []
        self.dynamic_equations = []
        self.steady_equations = []
        self.log_variables = []
        self.all_but = []
        self.context = None

    @property
    def num_quantities(self, /) -> int:
        return len(self.quantities)

    @property
    def all_names(self, /) -> Iterable[str]:
        return [ qty.human for qty in self.quantities ]

    def add_parameters(self, names: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(names, _quantities.QuantityKind.PARAMETER)

    def add_exogenous_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _quantities.QuantityKind.EXOGENOUS_VARIABLE)

    def add_transition_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _quantities.QuantityKind.TRANSITION_VARIABLE)

    def add_anticipated_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _quantities.QuantityKind.ANTICIPATED_SHOCK)

    def add_unanticipated_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _quantities.QuantityKind.UNANTICIPATED_SHOCK)

    def add_transition_equations(self, equation_inputs: Iterable[EquationInput] | None) -> None:
        self._add_equations(equation_inputs, _equations.EquationKind.TRANSITION_EQUATION)

    def add_measurement_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _quantities.QuantityKind.MEASUREMENT_VARIABLE)

    def add_measurement_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, _quantities.QuantityKind.MEASUREMENT_SHOCK)

    def add_measurement_equations(self, equation_inputs: Iterable[EquationInput] | None) -> None:
        self._add_equations(equation_inputs, _equations.EquationKind.MEASUREMENT_EQUATION)

    def add_all_but(self, all_but_input: str | None) -> None:
        if not all_but_input:
            return
        self.all_but.append(all_but_input)

    def add_log_variables(self, log_variable_inputs: Iterable[str] | None, /, ) -> None:
        if not log_variable_inputs:
            return
        self.log_variables += log_variable_inputs

    def _add_quantities(self, quantity_inputs: Iterable[QuantityInput] | None, kind: _quantities.QuantityKind) -> None:
        """
        """
        if not quantity_inputs:
            return
        start_entry = self.quantities[-1].entry + 1 if self.quantities else 0
        self.quantities = self.quantities + [
            _quantities.Quantity(
                id=None,
                human=q[1].strip(),
                kind=kind,
                logly=None,
                description=q[0].strip(),
                entry=i,
                attributes=_conform_attributes(q[2], )
            )
            for i, q in enumerate(quantity_inputs, start=start_entry)
        ]

    def _add_equations(self, equation_inputs: Iterable[EquationInput] | None, kind: _equations.EquationKind, /) -> None:
        """
        """
        if not equation_inputs:
            return
        #
        def _create_equations(equation_inputs, kind, start_entry, human_func):
            return [
                _equations.Equation(
                    id=None,
                    human=human_func(ein),
                    kind=kind,
                    description=ein[0].strip(),
                    entry=i,
                    attributes=_conform_attributes(ein[2], ),
                )
                for i, ein in enumerate(equation_inputs, start=start_entry)
            ]
        #
        def _human_func_dynamic(ein):
            return _handle_white_spaces(ein[1][0])
        #
        def _human_func_steady(ein):
            return _handle_white_spaces(ein[1][1] if ein[1][1] else ein[1][0])
        #
        start_entry = self.dynamic_equations[-1].entry + 1 if self.dynamic_equations else 0
        self.dynamic_equations += _create_equations(equation_inputs, kind, start_entry, _human_func_dynamic)
        self.steady_equations += _create_equations(equation_inputs, kind, start_entry, _human_func_steady)

    def populate_logly(self, /) -> None:
        """
        """
        default_logly = self._resolve_default_logly()
        listed_logly = not default_logly
        for qty in self.quantities:
            if qty.kind not in LOGLY_VARIABLE:
                continue
            qty.logly = listed_logly if qty.human in self.log_variables else default_logly

    def _is_logly_consistent(self, /) -> bool:
        return all(a=="all-but" for a in self.all_but) or all(a=="" for a in self.all_but) if self.all_but else True

    def _resolve_default_logly(self, /) -> bool:
        if not self._is_logly_consistent():
            raise Exception("Inconsistent use of !all-but in !log-variables")
        return self.all_but.pop()=="all-but" if self.all_but else False

    @classmethod
    def from_lists(
        klass,
        /,
        transition_variables: Iterable[QuantityInput],
        transition_equations: Iterable[EquationInput], 
        anticipated_shocks: Iterable[QuantityInput] | None = None,
        unanticipated_shocks: Iterable[QuantityInput] | None = None,
        measurement_variables: Iterable[QuantityInput] | None = None,
        measurement_equations: Iterable[EquationInput] | None = None,
        measurement_shocks: Iterable[QuantityInput] | None = None,
        parameters: Iterable[QuantityInput] | None = None,
        exogenous_variables: Iterable[QuantityInput] | None = None,
        log_variables: Iterable[str] | None = None,
    ) -> Self:
        """
        """
        self = klass()
        self.add_transition_variables(transition_variables, )
        self.add_transition_equations(transition_equations, )
        self.add_anticipated_shocks(anticipated_shocks, )
        self.add_unanticipated_shocks(unanticipated_shocks, )
        self.add_measurement_variables(measurement_variables, )
        self.add_measurement_equations(measurement_equations, )
        self.add_measurement_shocks(measurement_shocks, )
        self.add_parameters(parameters, )
        self.add_exogenous_variables(exogenous_variables, )
        self.add_log_variables(log_variables, )
        self.populate_logly()
        return self

    @classmethod
    def from_string(
        klass,
        source_string: str,
        /,
        context: dict | None = None,
        save_preparsed: str = "",
    ) -> tuple[Self, dict]:
        """
        """
        self = klass()
        self.context = context
        preparsed_string, preparser_info = _preparser.from_string(
            source_string, context=self.context, save_preparsed=save_preparsed, 
        )
        parsed_content = _models.from_string(preparsed_string)
        #
        self.add_transition_variables(parsed_content.get("transition-variables"))
        self.add_anticipated_shocks(parsed_content.get("anticipated-shocks"))
        self.add_unanticipated_shocks(parsed_content.get("unanticipated-shocks"))
        self.add_transition_equations(parsed_content.get("transition-equations"))
        self.add_measurement_variables(parsed_content.get("measurement-variables"))
        self.add_measurement_shocks(parsed_content.get("measurement-shocks"))
        self.add_measurement_equations(parsed_content.get("measurement-equations"))
        self.add_parameters(parsed_content.get("parameters"))
        self.add_exogenous_variables(parsed_content.get("exogenous-variables"))
        self.add_log_variables(parsed_content.get("log-variables"))
        for i in parsed_content.get("all-but", []):
            self.add_all_but(i)
        self.populate_logly()
        #
        return self, preparser_info

    #]


def _handle_white_spaces(x: str, /) -> str:
    return _re.sub(r"[\s\n\r]+", "", x)


def _conform_attributes(attributes: str | Iterable[str] | None, /, ) -> set:
    """
    """
    #[
    if not attributes:
        return set()
    if isinstance(attributes, str):
        attributes = set((attributes, ))
    return set(a.strip() for a in attributes)
    #]


def _combine_source_files(
    source_files: str | Iterable[str],
    /,
    joint: str = "\n\n",
) -> str:
    """
    """
    #[
    if isinstance(source_files, str):
        source_files = [source_files]
    source_strings = []
    for f in source_files:
        with open(f, "r") as fid:
            source_strings.append(fid.read(), )
    return "\n\n".join(source_strings, )
    #]

