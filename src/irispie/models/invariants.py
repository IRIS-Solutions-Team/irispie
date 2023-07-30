"""
"""


#[
from typing import (Self, Callable, )

from .. import (equations as _eq, quantities as _qu, wrongdoings as _wd, )
from ..fords import (descriptors as _fd, )
from ..equators import (plain as _ep, )
from . import (facade as _mf, flags as _mg, )
#]


_PLAIN_EQUATOR_EQUATION = (
    _eq.EquationKind.TRANSITION_EQUATION
    | _eq.EquationKind.MEASUREMENT_EQUATION
)


class Invariant:
    """
    Invariant part of a Model object
    """
    #[
    def __init__(
        self,
        model_source,
        /,
        context: dict | None = None,
        needs_check_syntax: bool = True,
        **kwargs,
    ) -> Self:
        """
        """
        self._flags = _mg.Flags.from_kwargs(**kwargs, )
        #
        self._populate_function_context(context)
        #
        self._quantities = model_source.quantities[:]
        self._dynamic_equations = model_source.dynamic_equations[:]
        self._steady_equations = model_source.steady_equations[:]
        #
        name_to_qid = _qu.create_name_to_qid(self._quantities, )
        _eq.finalize_dynamic_equations(self._dynamic_equations, name_to_qid, )
        _eq.finalize_steady_equations(self._steady_equations, name_to_qid, )
        if needs_check_syntax:
            _check_syntax(self._dynamic_equations, self._function_context, )
            _check_syntax(self._steady_equations, self._function_context, )
        #
        self._dynamic_descriptor = _fd.Descriptor(self._dynamic_equations, self._quantities, self._function_context, )
        self._steady_descriptor = _fd.Descriptor(self._steady_equations, self._quantities, self._function_context, )
        #
        self._plain_equator_for_dynamic_equations = _ep.PlainEquator(
            _eq.generate_equations_of_kind( self._dynamic_equations, _PLAIN_EQUATOR_EQUATION, ),
            custom_functions=self._function_context,
        )
        #
        self._plain_equator_for_steady_equations = _ep.PlainEquator(
            _eq.generate_equations_of_kind(self._steady_equations, _PLAIN_EQUATOR_EQUATION, ),
            custom_functions=self._function_context,
        )
        #
        self._populate_min_max_shifts()

    def _populate_min_max_shifts(self, /, ) -> None:
        """
        """
        self._min_shift = _eq.get_min_shift_from_equations(
            self._dynamic_equations + self._steady_equations,
        )
        self._max_shift = _eq.get_max_shift_from_equations(
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
        eval(_eq.create_equator_func_string(equations), )
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
        _wd.throw("error", message)
    #]


def _success_creating_lambda(equation, function_context):
    """
    """
    #[
    try:
        eval(_eq.create_equator_func_string([equation.xtring]))
        return True
    except Exception as ex:
        print(ex)
        breakpoint()
        return False
    #]

