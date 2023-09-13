"""
"""


#[
from __future__ import annotations

from typing import (Self, Callable, )

from .. import equations as _equations
from .. import quantities as _quantities
from .. import wrongdoings as _wrongdoings
from ..fords import descriptors as _descriptors
from ..equators import plain as _equators

from . import _flags
#]


_PLAIN_EQUATOR_EQUATION = (
    _equations.EquationKind.TRANSITION_EQUATION
    | _equations.EquationKind.MEASUREMENT_EQUATION
)


class Invariant:
    """
    Invariant part of a Model object
    """
    #[
    __slots__ = (
        "_flags",
        "_function_context",
        "_quantities",
        "_dynamic_equations",
        "_steady_equations",
        "_dynamic_descriptor",
        "_steady_descriptor",
        "_min_shift",
        "_max_shift",
    )
    def __init__(
        self,
        model_source,
        /,
        context: dict | None = None,
        check_syntax: bool = True,
        **kwargs,
    ) -> Self:
        """
        """
        self._flags = _flags.Flags.from_kwargs(**kwargs, )
        #
        self._populate_function_context(context)
        #
        self._quantities = _cp.deepcopy(model_source.quantities, )
        self._dynamic_equations = _cp_deepcopy(model_source.dynamic_equations, )
        self._steady_equations = _cp.deepcopy(model_source.steady_equations, )
        #
        name_to_qid = _quantities.create_name_to_qid(self._quantities, )
        _equations.finalize_dynamic_equations(self._dynamic_equations, name_to_qid, )
        _equations.finalize_steady_equations(self._steady_equations, name_to_qid, )
        #
        if check_syntax:
            _check_syntax(self._dynamic_equations, self._function_context, )
            _check_syntax(self._steady_equations, self._function_context, )
        #
        self._dynamic_descriptor = _descriptors.Descriptor(self._dynamic_equations, self._quantities, self._function_context, )
        self._steady_descriptor = _descriptors.Descriptor(self._steady_equations, self._quantities, self._function_context, )
        #
        self._plain_equator_for_dynamic_equations = _equators.PlainEquator(
            _equations.generate_equations_of_kind( self._dynamic_equations, _PLAIN_EQUATOR_EQUATION, ),
            custom_functions=self._function_context,
        )
        #
        self._plain_equator_for_steady_equations = _equators.PlainEquator(
            _equations.generate_equations_of_kind(self._steady_equations, _PLAIN_EQUATOR_EQUATION, ),
            custom_functions=self._function_context,
        )
        #
        self._min_shift, self._max_shift = None, None
        self._populate_min_max_shifts()

    def _populate_min_max_shifts(self, /, ) -> None:
        """
        """
        self._min_shift = _equations.get_min_shift_from_equations(
            self._dynamic_equations + self._steady_equations,
        )
        self._max_shift = _equations.get_max_shift_from_equations(
            self._dynamic_equations + self._steady_equations,
        )

    def _populate_function_context(
        self,
        context: dict | None,
        /,
    ) -> None:
        """
        """
        self._function_context = {
            k: v for k, v in context.items()
            if isinstance(v, Callable)
        } if context else None
    #]


def _check_syntax(equations, function_context, /, ):
    """
    Try all equations at once; if this fails, do equation by equation to # catch the troublemakers
    """
    try:
        eval(_equations.create_equator_func_string(equations), )
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
        _wrongdoings.throw("error", message)
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

