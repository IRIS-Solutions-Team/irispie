"""
"""


#[
from __future__ import annotations

from typing import (Self, Callable, NoReturn, )
from numbers import (Number, )
import copy as _cp

from ..conveniences import descriptions as _descriptions
from ..fords import descriptors as _descriptors
from ..equators import plain as _equators
from .. import equations as _equations
from .. import quantities as _quantities
from .. import wrongdoings as _wrongdoings
from .. import makers as _makers

from . import _flags
#]


_DEFAULT_STD_LINEAR = 1
_DEFAULT_STD_NONLINEAR = 0.01


_PLAIN_EQUATOR_EQUATION = (
    _equations.EquationKind.TRANSITION_EQUATION
    | _equations.EquationKind.MEASUREMENT_EQUATION
)


class Invariant(
    _descriptions.DescriptionMixin,
):
    """
    Invariant part of a Simultaenous object
    """
    #[

    _DEFAULT_STD_NAME_FORMAT = "std_{}"
    _DEFAULT_STD_DESCRIPTION_FORMAT = "(Std) {}"

    __slots__ = (
        "quantities",
        "dynamic_equations",
        "steady_equations",
        "dynamic_descriptor",
        "steady_descriptor",
        "_flags",
        "_context",
        "_shock_qid_to_std_qid",
        "_plain_dynamic_equator",
        "_plain_steady_equator",
        "_min_shift",
        "_max_shift",
        "_default_std",
        "_description",
    )

    def __init__(
        self,
        source,
        /,
        check_syntax: bool = True,
        std_name_format: str | None = None,
        std_description_format: str | None = None,
        autodeclare_as: str | None = None,
        default_std: Number | None = None,
        description: str | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        self.set_description(description, )
        self._flags = _flags.Flags.from_kwargs(**kwargs, )
        self._default_std = _resolve_default_std(default_std, self._flags, )
        self._context = source.context or {}
        std_name_format = std_name_format or self._DEFAULT_STD_NAME_FORMAT
        std_description_format = std_description_format or self._DEFAULT_STD_DESCRIPTION_FORMAT
        #
        self.quantities = tuple(source.quantities)
        self.dynamic_equations = tuple(source.dynamic_equations)
        self.steady_equations = tuple(source.steady_equations)
        #
        # Create std_ parameters for unanticipated and measurement shocks
        if self._flags.is_stochastic:
            unanticipated_shocks = _quantities.generate_quantities_of_kind(self.quantities, _quantities.QuantityKind.UNANTICIPATED_SHOCK, )
            measurement_shocks = _quantities.generate_quantities_of_kind(self.quantities, _quantities.QuantityKind.MEASUREMENT_SHOCK, )
            #
            self.quantities += tuple(_generate_stds(
                unanticipated_shocks,
                _quantities.QuantityKind.UNANTICIPATED_STD,
                std_name_format,
                std_description_format,
                entry=len(self.quantities),
            ))
            self.quantities += tuple(_generate_stds(
                measurement_shocks,
                _quantities.QuantityKind.MEASUREMENT_STD,
                std_name_format,
                std_description_format,
                entry=len(self.quantities),
            ))
        #
        # Look up undeclared names; autodeclare these names or throw an error
        undeclared_names = _collect_undeclared_names(
            self.quantities,
            self.dynamic_equations + self.steady_equations,
        )
        if undeclared_names and autodeclare_as is None:
            raise _wrongdoings.IrisPieCritical([
                "These names are used in equations but not declared:",
                *undeclared_names,
                ])
        self.quantities += tuple(_generate_quantities_for_undeclared_names(
            undeclared_names, autodeclare_as, entry=len(self.quantities),
        ))
        #
        # Verify uniqueness of all quantity names
        _quantities.check_unique_names(self.quantities)
        #
        # Number of transition equations must equal number of transition variables
        _check_numbers_of_variables_equations(
            self.quantities,
            self.dynamic_equations,
            _quantities.QuantityKind.TRANSITION_VARIABLE,
            _equations.EquationKind.TRANSITION_EQUATION,
        )
        #
        # Number of measurement equations must equal number of measurement variables
        _check_numbers_of_variables_equations(
            self.quantities,
            self.dynamic_equations,
            _quantities.QuantityKind.MEASUREMENT_VARIABLE,
            _equations.EquationKind.MEASUREMENT_EQUATION,
        )
        #
        self.quantities = _quantities.reorder_by_kind(self.quantities, )
        self.dynamic_equations = _equations.reorder_by_kind(self.dynamic_equations, )
        self.steady_equations = _equations.reorder_by_kind(self.steady_equations, )
        _quantities.stamp_id(self.quantities, )
        _equations.stamp_id(self.dynamic_equations, )
        _equations.stamp_id(self.steady_equations, )
        #
        self._shock_qid_to_std_qid = (
            _create_shock_qid_to_std_qid(self.quantities, std_name_format, )
            if self._flags.is_stochastic
            else {}
        )
        #
        name_to_qid = _quantities.create_name_to_qid(self.quantities, )
        _equations.finalize_dynamic_equations(self.dynamic_equations, name_to_qid, )
        _equations.finalize_steady_equations(self.steady_equations, name_to_qid, )
        #
        if check_syntax:
            _check_syntax(self.dynamic_equations, self._context, )
            _check_syntax(self.steady_equations, self._context, )
        #
        self._min_shift, self._max_shift = None, None
        self._populate_min_max_shifts()
        #
        self.dynamic_descriptor = _descriptors.Descriptor(self.dynamic_equations, self.quantities, self._context, )
        self.steady_descriptor = _descriptors.Descriptor(self.steady_equations, self.quantities, self._context, )
        #
        # Create a function to evaluate LHS–RHS in dynamic equations
        self._plain_dynamic_equator = _equators.PlainEquator(
            _equations.generate_equations_of_kind(self.dynamic_equations, _PLAIN_EQUATOR_EQUATION, ),
            context=self._context,
        )
        #
        # Create a function to evaluate LHS–RHS in steady equations
        self._plain_steady_equator = _equators.PlainEquator(
            _equations.generate_equations_of_kind(self.steady_equations, _PLAIN_EQUATOR_EQUATION, ),
            context=self._context,
        )

    @property
    def num_transition_equations(self, /, ) -> int:
        """
        """
        return sum(
            1 for e in self.dynamic_equations
            if e.kind is _equations.EquationKind.TRANSITION_EQUATION
        )

    @property
    def num_measurement_equations(self, /, ) -> int:
        """
        """
        return sum(
            1 for e in self.dynamic_equations
            if e.kind is _equations.EquationKind.MEASUREMENT_EQUATION
        )

    def _populate_min_max_shifts(self, /, ) -> None:
        """
        """
        self._min_shift = _equations.get_min_shift_from_equations(
            self.dynamic_equations + self.steady_equations,
        )
        self._max_shift = _equations.get_max_shift_from_equations(
            self.dynamic_equations + self.steady_equations,
        )

    #]


def _check_syntax(equations, function_context, /, ):
    """
    Try all equations at once; if this fails, do equation by equation to catch the troublemakers
    """
    try:
        eval(_equations.create_equator_func_string(equations, ), )
    except:
        _catch_troublemakers(equations, function_context, )
    #]


def _catch_troublemakers(equations, function_context, /, ):
    """
    Catch the troublemakers
    """
    #[
    fail = [
        eqn.human for eqn in equations
        if not _success_creating_lambda(eqn, function_context)
    ]
    if fail:
        message = ["Syntax error in these equations"] + fail
        raise _wrongdoings.IrisPieCritical(message, )
    #]


def _success_creating_lambda(equation, function_context):
    """
    """
    #[
    try:
        eval(_equations.create_equator_func_string([equation.xtring]))
        return True
    except Exception as ex:
        return False
    #]


def _create_shock_qid_to_std_qid(
    quantities: Iterable[_quantities.Quantity],
    std_name_format: str,
    /,
) -> dict[int, int]:
    """
    """
    #[
    name_to_qid = _quantities.create_name_to_qid(quantities, )
    qid_to_name = _quantities.create_qid_to_name(quantities, )
    kind = _quantities.QuantityKind.STOCHASTIC_SHOCK
    all_shock_qids = tuple(_quantities.generate_qids_by_kind(quantities, kind))
    return {
        shock_qid: name_to_qid[std_name_format.format(qid_to_name[shock_qid], )]
        for shock_qid in all_shock_qids
    }
    #]


def _generate_stds(
    shocks: tuple[_quantities.Quantity, ...],
    std_kind: _quantities.QuantityKind,
    std_name_format: str,
    std_description_format: str,
    /,
    *,
    entry: int,
) -> tuple[_quantities.Quantity, ...]:
    """
    """
    #[
    return (
        _quantities.Quantity(
            id=None,
            human=std_name_format.format(shock.human, ),
            kind=std_kind,
            logly=None,
            description=std_description_format.format(shock.description or shock.human, ),
            entry=entry,
        )
        for shock in shocks
    )
    #]


def _collect_undeclared_names(
    quantities: tuple[_quantities.Quantity, ...],
    equations: tuple[_equations.Equation, ...],
    /,
) -> tuple[str, ...]:
    """
    """
    #[
    all_names = set(_quantities.generate_all_quantity_names(quantities, ))
    all_names_in_equations = set(_equations.generate_all_names_from_equations(equations, ))
    return tuple(all_names_in_equations - all_names)
    #]


def _generate_quantities_for_undeclared_names(
    undeclared_names: set[str],
    autodeclare_as: str | None,
    /,
    *,
    entry: int,
) -> tuple[_quantities.Quantity, ...] | NoReturn:
    """
    """
    #[
    if not undeclared_names:
        return tuple()
    autodeclare_kind = _quantities.QuantityKind.from_keyword(autodeclare_as, )
    return (
        _quantities.Quantity(
            id=None,
            human=name,
            kind=autodeclare_kind,
            logly=None,
            description=None,
            entry=entry,
        )
        for name in undeclared_names
    )
    #]


def _check_numbers_of_variables_equations(
    quantities: tuple[_quantities.Quantity, ...],
    equations: tuple[_equations.Equation, ...],
    quantity_kind: _quantities.QuantityKind,
    equation_kind: _equations.EquationKind,
    /,
) -> None:
    """
    """
    #[
    num_variables = _quantities.count_quantities_of_kind(quantities, quantity_kind, )
    num_equations = _equations.count_equations_of_kind(equations, equation_kind, )
    if num_variables != num_equations:
        raise _wrongdoings.IrisPieCritical([
            "Inconsistent numbers of variables and equations",
            f"Number of {quantity_kind.human.lower()}s: {num_variables}",
            f"Number of {equation_kind.human.lower()}s: {num_equations}",
        ])
    #]


def _resolve_default_std(
    custom_default_std: Number | None,
    flags: _flags.Flags,
) -> Number:
    """
    """
    #[
    if custom_default_std is not None:
        return custom_default_std
    elif flags.is_linear:
        return _DEFAULT_STD_LINEAR
    else:
        return _DEFAULT_STD_NONLINEAR
    #]

