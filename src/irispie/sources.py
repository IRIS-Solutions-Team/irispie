"""
Model source
"""


#[

from __future__ import annotations

import re as _re
from typing import Self, Any, Type, TypeAlias, Literal, Protocol, NoReturn
from collections.abc import Iterable
import documark as _dm

from . import equations as _equations
from .equations import Equation, EquationKind
from . import quantities as _quantities
from .quantities import QuantityKind, Quantity
from . import wrongdoings as _wd
from .parsers import preparser as _preparser
from .parsers import models as _models
from .parsers import common as _common

#]


__all__ = (
    "ModelSource",
)


Description: TypeAlias = str
Human: TypeAlias = str
Attributes: TypeAlias = tuple[str, ...]
QuantityInput: TypeAlias = tuple[Description, Human, Attributes]
EquationInput: TypeAlias = tuple[Description, tuple[Human, Human], Attributes]


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
    source_string = _combine_source_files_into_string(source_files, )
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
        "context",
        "description",
    )

    def __init__(self, **kwargs, ) -> None:
        self.quantities = kwargs.get("quantities", [])
        self.dynamic_equations = kwargs.get("dynamic_equations", [])
        self.steady_equations = kwargs.get("steady_equations", [])
        self.context = kwargs.get("context", {})
        self.description = kwargs.get("description", "")

    @property
    def num_quantities(self, /, ) -> int:
        return len(self.quantities)

    @property
    def all_names(self, /, ) -> Iterable[str]:
        return [ qty.human for qty in self.quantities ]

    def _add_parameters(self, names: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(names, QuantityKind.PARAMETER)

    def _add_exogenous_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, QuantityKind.EXOGENOUS_VARIABLE)

    def _add_transition_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, QuantityKind.TRANSITION_VARIABLE)

    def _add_anticipated_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, QuantityKind.ANTICIPATED_SHOCK)

    def _add_unanticipated_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, QuantityKind.UNANTICIPATED_SHOCK)

    def _add_transition_equations(self, equation_inputs: Iterable[EquationInput] | None) -> None:
        self._add_equations(equation_inputs, EquationKind.TRANSITION_EQUATION)

    def _add_measurement_variables(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, QuantityKind.MEASUREMENT_VARIABLE)

    def _add_measurement_shocks(self, quantity_inputs: Iterable[QuantityInput] | None) -> None:
        self._add_quantities(quantity_inputs, QuantityKind.MEASUREMENT_SHOCK)

    def _add_measurement_equations(self, equation_inputs: Iterable[EquationInput] | None) -> None:
        self._add_equations(equation_inputs, EquationKind.MEASUREMENT_EQUATION)

    def _add_steady_autovalues(self, equation_inputs: Iterable[EquationInput] | None) -> None:
        self._add_equations(equation_inputs, EquationKind.STEADY_AUTOVALUES, )

    def _add_all_but(self, all_but_input: str | None) -> None:
        if not all_but_input:
            return
        self.all_but.append(all_but_input)

    def _add_log_variables(self, log_variable_inputs: Iterable[str] | None, /, ) -> None:
        if not log_variable_inputs:
            return
        self.log_variables += log_variable_inputs

    def _add_quantities(self, quantity_inputs: Iterable[QuantityInput] | None, kind: QuantityKind) -> None:
        """
        """
        if not quantity_inputs:
            return
        start_entry = self.quantities[-1].entry + 1 if self.quantities else 0
        self.quantities = self.quantities + [
            Quantity(
                id=None,
                human=q[1].strip(),
                kind=kind,
                logly=None,
                description=q[0].strip(),
                entry=i,
                attributes=_conform_attributes(q[2], )
            )
            for i, q in enumerate(quantity_inputs, start=start_entry, )
        ]

    def _add_equations(self, equation_inputs: Iterable[EquationInput] | None, kind: EquationKind, /, ) -> None:
        """
        """
        if not equation_inputs:
            return
        #
        def _create_equations(equation_inputs, kind, start_entry, human_func):
            return [
                Equation(
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
        equation_inputs = tuple(equation_inputs)
        start_entry = self.dynamic_equations[-1].entry + 1 if self.dynamic_equations else 0
        self.dynamic_equations += _create_equations(equation_inputs, kind, start_entry, _human_func_dynamic)
        self.steady_equations += _create_equations(equation_inputs, kind, start_entry, _human_func_steady)

    def _verify_log_variables(self, log_variables: Iterable[str] | None) -> None:
        """
        """
        if not log_variables:
            return
        loggables = _extract_loggable_names(self.quantities, )
        illegal_log_variables = set(log_variables) - loggables
        if illegal_log_variables:
            raise _wd.IrisPieCritical((
                f"Illegal name(s) on the {_models.LOG_VARIABLES_KEYWORD} list",
                *illegal_log_variables,
            ))

    def _populate_logly(
        self,
        log_variables: Iterable[str] | None,
        all_but: bool | False,
    ) -> None:
        """
        """
        log_variables = set(log_variables) if log_variables else set()
        unlisted_logly = bool(all_but)
        listed_logly = not unlisted_logly
        def is_logly(human: str, /, ) -> bool:
            return listed_logly if human in log_variables else unlisted_logly
        for qty in self.quantities:
            if qty.kind not in QuantityKind.LOGGABLE_VARIABLE:
                continue
            qty.logly = is_logly(qty.human, )

    @classmethod
    def from_lists(
        klass,
        transition_variables: Iterable[QuantityInput],
        transition_equations: Iterable[EquationInput],
        anticipated_shocks: Iterable[QuantityInput] | None = None,
        unanticipated_shocks: Iterable[QuantityInput] | None = None,
        measurement_variables: Iterable[QuantityInput] | None = None,
        measurement_equations: Iterable[EquationInput] | None = None,
        measurement_shocks: Iterable[QuantityInput] | None = None,
        steady_autovalues: Iterable[EquationInput] | None = None,
        parameters: Iterable[QuantityInput] | None = None,
        exogenous_variables: Iterable[QuantityInput] | None = None,
        log_variables: Iterable[str] | None = None,
        all_but: bool = False,
        context: dict | None = None,
    ) -> Self:
        """
        """
        self = klass()
        #
        self._add_transition_variables(transition_variables, )
        self._add_transition_equations(transition_equations, )
        self._add_anticipated_shocks(anticipated_shocks, )
        self._add_unanticipated_shocks(unanticipated_shocks, )
        self._add_measurement_variables(measurement_variables, )
        self._add_parameters(parameters, )
        self._add_exogenous_variables(exogenous_variables, )
        self._verify_log_variables(log_variables, )
        self._populate_logly(log_variables, all_but, )
        #
        self._add_measurement_equations(measurement_equations, )
        self._add_measurement_shocks(measurement_shocks, )
        self._add_steady_autovalues(steady_autovalues, )
        #
        self.context = (dict(context) if context else {}) | {"__builtins__": {}}
        #
        return self

    @classmethod
    def from_string(
        klass,
        source_string: str,
        *,
        context: dict | None = None,
        save_preparsed: str = "",
    ) -> tuple[Self, dict[str, Any]]:
        """
        """
        context = (dict(context) if context else {}) | {"__builtins__": {}}
        preparsed_string, preparser_info = _preparser.from_string(
            source_string, context=context, save_preparsed=save_preparsed, 
        )
        parsed_content = _models.from_string(preparsed_string)
        all_but = _is_all_but_present(parsed_content.get("all-but", ), )
        log_variables = set(parsed_content.get("log-variables", ()), )
        #
        self = klass.from_lists(
            transition_variables=parsed_content.get("transition-variables", ),
            transition_equations=parsed_content.get("transition-equations", ),
            anticipated_shocks=parsed_content.get("anticipated-shocks", ),
            unanticipated_shocks=parsed_content.get("unanticipated-shocks", ),
            measurement_variables=parsed_content.get("measurement-variables", ),
            measurement_equations=parsed_content.get("measurement-equations", ),
            measurement_shocks=parsed_content.get("measurement-shocks", ),
            steady_autovalues=parsed_content.get("steady-autovalues", ),
            parameters=parsed_content.get("parameters", ),
            exogenous_variables=parsed_content.get("exogenous-variables", ),
            log_variables=log_variables,
            all_but=all_but,
            context=context,
        )
        #
        return self, preparser_info

    @classmethod
    def from_file(
        klass,
        source_files: str | Iterable[str],
        /,
        **kwargs,
    ) -> Self:
        """Create a new ModelSource from source file(s)"""
        source_string = _combine_source_files_into_string(source_files, )
        return klass.from_string(source_string, **kwargs, )

    #]


def _handle_white_spaces(x: str, /, ) -> str:
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


def _combine_source_files_into_string(
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


def _verify_all_but(all_but: list[str] | None, /, ) -> None | NoReturn:
    if all_but is None:
        return
    if all(i == _models.ALL_BUT_KEYWORD for i in all_but):
        return
    if all(i == "" for i in all_but):
        return
    raise Exception("Inconsistent use of !all-but in !log-variables")


def _is_all_but_present(all_but: Iterable[str] | None, /, ) -> bool:
    if not all_but:
        return False
    all_but = tuple(all_but)
    _verify_all_but(all_but, )
    return all_but[0] == _models.ALL_BUT_KEYWORD


def _extract_loggable_names(quantities: Iterable[Quantity], /, ) -> set[str]:
    """
    """
    return set(
        i.human for i in quantities
        if i.kind in QuantityKind.LOGGABLE_VARIABLE
    )

