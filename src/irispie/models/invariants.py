"""
"""


#[
from __future__ import annotations

from typing import (Self, Callable, )
import copy as _cp

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
        "quantities",
        "dynamic_equations",
        "steady_equations",
        "preprocessor",
        "postprocessor",
        "_flags",
        "_function_context",
        "shock_qid_to_std_qid",
        "dynamic_descriptor",
        "steady_descriptor",
        "_plain_equator_for_dynamic_equations",
        "_plain_equator_for_steady_equations",
        "_min_shift",
        "_max_shift",
    )
    def __init__(
        self,
        source,
        /,
        context: dict | None = None,
        check_syntax: bool = True,
        **kwargs,
    ) -> Self:
        """
        """
        self._flags = _flags.Flags.from_kwargs(**kwargs, )
        self._populate_function_context(context)
        #
        self.quantities = _cp.deepcopy(source.quantities)
        self.dynamic_equations = _cp.deepcopy(source.dynamic_equations)
        self.steady_equations = _cp.deepcopy(source.steady_equations)
        self.preprocessor = None
        self.postprocessor = None
        #
        _add_stds(self.quantities, _quantities.QuantityKind.TRANSITION_SHOCK, _quantities.QuantityKind.TRANSITION_STD, )
        _add_stds(self.quantities, _quantities.QuantityKind.MEASUREMENT_SHOCK, _quantities.QuantityKind.MEASUREMENT_STD, )
        _quantities.check_unique_names(self.quantities)
        #
        quantities = _quantities.reorder_by_kind(self.quantities, )
        dynamic_equations = _equations.reorder_by_kind(self.dynamic_equations, )
        steady_equations = _equations.reorder_by_kind(self.steady_equations, )
        _quantities.stamp_id(self.quantities, )
        _equations.stamp_id(self.dynamic_equations, )
        _equations.stamp_id(self.steady_equations, )
        #
        self.shock_qid_to_std_qid = _create_shock_qid_to_std_qid(self.quantities, )
        #
        name_to_qid = _quantities.create_name_to_qid(self.quantities, )
        _equations.finalize_dynamic_equations(self.dynamic_equations, name_to_qid, )
        _equations.finalize_steady_equations(self.steady_equations, name_to_qid, )
        #
        if check_syntax:
            _check_syntax(self.dynamic_equations, self._function_context, )
            _check_syntax(self.steady_equations, self._function_context, )
        #
        self.dynamic_descriptor = _descriptors.Descriptor(self.dynamic_equations, self.quantities, self._function_context, )
        self.steady_descriptor = _descriptors.Descriptor(self.steady_equations, self.quantities, self._function_context, )
        #
        self._plain_equator_for_dynamic_equations = _equators.PlainEquator(
            _equations.generate_equations_of_kind( self.dynamic_equations, _PLAIN_EQUATOR_EQUATION, ),
            custom_functions=self._function_context,
        )
        #
        self._plain_equator_for_steady_equations = _equators.PlainEquator(
            _equations.generate_equations_of_kind(self.steady_equations, _PLAIN_EQUATOR_EQUATION, ),
            custom_functions=self._function_context,
        )
        #
        self._min_shift, self._max_shift = None, None
        self._populate_min_max_shifts()

    def _populate_min_max_shifts(self, /, ) -> None:
        """
        """
        self._min_shift = _equations.get_min_shift_from_equations(
            self.dynamic_equations + self.steady_equations,
        )
        self._max_shift = _equations.get_max_shift_from_equations(
            self.dynamic_equations + self.steady_equations,
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


def _create_shock_qid_to_std_qid(
    quantities: Iterable[_quantities.Quantity],
    /,
) -> dict[int, int]:
    """
    """
    name_to_qid = _quantities.create_name_to_qid(quantities, )
    qid_to_name = _quantities.create_qid_to_name(quantities, )
    kind = _quantities.QuantityKind.SHOCK
    all_shock_qids = tuple(_quantities.generate_qids_by_kind(quantities, kind))
    return {
        shock_qid: name_to_qid[_create_std_name(qid_to_name[shock_qid], )]
        for shock_qid in all_shock_qids
    }


def _add_stds(
    quantities: Iterable[_quantities.Quantity],
    shock_kind: _quantities.QuantityKind,
    std_kind: _quantities.QuantityKind,
    /,
) -> None:
    """
    """
    shocks = (q for q in quantities if q.kind in shock_kind)
    for std_qid, shock in enumerate(shocks, start=len(quantities)):
        std_human = _create_std_name(shock.human, )
        std_logly = False
        std_description = _create_std_description(shock.description, shock.human, )
        std_entry = len(quantities)
        std_quantity = _quantities.Quantity(std_qid, std_human, std_kind, std_logly, std_description, std_entry, )
        quantities.append(std_quantity, )


_STD_PREFIX = "std_"
_STD_DESCRIPTION = "(Std) "


def _create_std_name(
    shock_name: str,
    /,
) -> str:
    """
    """
    return _STD_PREFIX + shock_name


def _create_std_description(
    shock_description: str,
    shock_human: str,
    /,
) -> str:
    """
    """
    return _STD_DESCRIPTION + (shock_description if shock_description else shock_human)

