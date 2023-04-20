"""
"""


#[
from __future__ import (annotations, )

from typing import (Self, NoReturn, Callable, )

from .. import (equations as eq_, quantities as qu_, evaluators as ev_, wrongdoings as wd_, )
from ..models import (facade as mf_, evaluators as me_, flags as mg_, )
from ..fords import (descriptors as fd_, )
#]


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
        self._flags = mg_.ModelFlags.from_kwargs(**kwargs, )
        #
        self._populate_function_context(context)
        #
        self._quantities = model_source.quantities[:]
        self._dynamic_equations = model_source.dynamic_equations[:]
        self._steady_equations = model_source.steady_equations[:]
        #
        name_to_qid = qu_.create_name_to_qid(self._quantities, )
        eq_.finalize_dynamic_equations(self._dynamic_equations, name_to_qid, )
        eq_.finalize_steady_equations(self._steady_equations, name_to_qid, )
        if needs_check_syntax:
            _check_syntax(self._dynamic_equations, self._function_context, )
            _check_syntax(self._steady_equations, self._function_context, )
        #
        self._dynamic_descriptor = fd_.Descriptor(self._dynamic_equations, self._quantities, self._function_context, )
        self._steady_descriptor = fd_.Descriptor(self._steady_equations, self._quantities, self._function_context, )
        #
        dynamic_equations_for_plain_evaluator = eq_.generate_equations_of_kind(self._dynamic_equations, me_.STEADY_EVALUATOR_EQUATION, )
        self._plain_evaluator_for_dynamic_equations = ev_.PlainEvaluator(dynamic_equations_for_plain_evaluator, self._function_context, )
        #
        steady_equations_for_plain_evaluator = eq_.generate_equations_of_kind(self._steady_equations, me_.STEADY_EVALUATOR_EQUATION, )
        self._plain_evaluator_for_steady_equations = ev_.PlainEvaluator(steady_equations_for_plain_evaluator, self._function_context, )
        #
        self._populate_min_max_shifts()

    def _populate_min_max_shifts(self, /, ) -> NoReturn:
        """
        """
        self._min_shift = eq_.get_min_shift_from_equations(
            self._dynamic_equations + self._steady_equations,
        )
        self._max_shift = eq_.get_max_shift_from_equations(
            self._dynamic_equations + self._steady_equations,
        )

    def _populate_function_context(
        self,
        context: dict | None,
        /,
    ) -> NoReturn:
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
        eval(eq_.create_evaluator_func_string(equations), )
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
        wd_.throw("error", message)
    #]


def _success_creating_lambda(equation, function_context):
    """
    """
    #[
    try:
        eval(eq_.create_evaluator_func_string([equation.xtring]))
        return True
    except Exception as ex:
        breakpoint()
        return False
    #]

