"""
"""


#[
from __future__ import annotations

import copy as _cp
import functools as _ft
import itertools as _it
import numpy as _np
import re as _re

from ..conveniences import descriptions as _descriptions
from ..fords import descriptors as _descriptors
from ..equators import plain as _equators
from .. import equations as _equations
from ..equations import EquationKind, Equation
from .. import quantities as _quantities
from ..quantities import QuantityKind, Quantity
from .. import wrongdoings as _wrongdoings
from .. import makers as _makers
from .. import contexts as _contexts
from ..incidences.main import ZERO_SHIFT_TOKEN_PATTERN
from .. import sources as _sources
from .. sources import ModelSource

from ._flags import Flags
from . import _tolerance

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self, Callable, NoReturn, Any
    from numbers import Real

#]


_DEFAULT_STD_LINEAR = 1
_DEFAULT_STD_NONLINEAR = 0.01


_PLAIN_EQUATOR_EQUATION = (
    EquationKind.TRANSITION_EQUATION
    | EquationKind.MEASUREMENT_EQUATION
)


_DEFAULT_STD_NAME_FORMAT = "std_{}"
_DEFAULT_STD_DESCRIPTION_FORMAT = "(Std) {}"


_STD_PREFIX = "std_"
_STD_DESCRIPTION_PREFIX = "(Std) "


def is_std_name(name: str, /, ) -> bool:
    return name.startswith(_STD_PREFIX)


def std_name_from_shock_name(shock_name: str, /, ) -> str:
    return f"{_STD_PREFIX}{shock_name}"


def shock_name_from_std_name(std_name: str, /, ) -> str:
    if not is_std_name(std_name):
        raise ValueError(f"Invalid std name: {std_name}")
    return std_name.removeprefix(_STD_PREFIX, )


def std_description_from_shock_description(shock_description: str, /, ) -> str:
    return f"{_STD_DESCRIPTION_PREFIX}{shock_description}"


class Invariant(
    _tolerance.InlayForInvariant,
    _descriptions.DescriptionMixin,
):
    """
    Invariant part of a Simultaenous object
    """
    #[

    _serialized_slots = (
        "quantities",
        "dynamic_equations",
        "steady_equations",
        "shock_qid_to_std_qid",
        "tolerance",
        "_context",
        "_default_std",
        "_flags",
        "_min_shift",
        "_max_shift",
        "__description__",
    )

    _derived_slots = (
        "dynamic_descriptor",
        "steady_descriptor",
        "update_steady_autovalues_in_variant",
        "_plain_dynamic_equator",
        "_plain_steady_equator",
    )

    __slots__ = _serialized_slots + _derived_slots

    def __init__(self, **kwargs, ) -> None:
        """
        """
        for n in self._serialized_slots:
            setattr(self, n, kwargs.get(n, ), )
        for n in self._derived_slots:
            setattr(self, n, None, )

    @classmethod
    def from_source(
        klass,
        source: ModelSource,
        #
        check_syntax: bool = True,
        autodeclare_as: str | None = None,
        default_std: Real | None = None,
        description: str | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass()
        self.reset_tolerance()
        self.set_description(
            description if description is not None
            else source.description
        )
        self._flags = Flags.from_kwargs(**kwargs, )
        self._default_std = _resolve_default_std(default_std, self._flags, )
        self._context = (dict(source.context) or {}) | {"__builtins__": None}
        #
        self.quantities = tuple(source.quantities)
        self.dynamic_equations = tuple(_equations.generate_equations_of_kind(
            source.dynamic_equations,
            kind=EquationKind.ENDOGENOUS_EQUATION | EquationKind.STEADY_AUTOVALUES,
        ))
        self.steady_equations = tuple(_equations.generate_equations_of_kind(
            source.steady_equations,
            kind=EquationKind.ENDOGENOUS_EQUATION | EquationKind.STEADY_AUTOVALUES,
        ))
        #
        # Create std_ parameters for all shocks
        if self._flags.is_stochastic:
            get_shocks = _ft.partial(_quantities.generate_quantities_of_kind, self.quantities, )
            unanticipated_shocks = get_shocks(kind=_quantities.UNANTICIPATED_SHOCK, )
            anticipated_shocks = get_shocks(kind=_quantities.ANTICIPATED_SHOCK, )
            measurement_shocks = get_shocks(kind=_quantities.MEASUREMENT_SHOCK, )
            #
            entry = len(self.quantities)
            self.quantities += tuple(
                _create_std_for_shock(i, QuantityKind.UNANTICIPATED_STD, entry, )
                for i in unanticipated_shocks
            )
            self.quantities += tuple(
                _create_std_for_shock(i, QuantityKind.ANTICIPATED_STD, entry, )
                for i in anticipated_shocks
            )
            self.quantities += tuple(
                _create_std_for_shock(i, QuantityKind.MEASUREMENT_STD, entry, )
                for i in measurement_shocks
            )
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
        # Count of transition equations must equal count of transition variables
        _check_numbers_of_variables_equations(
            self.quantities,
            self.dynamic_equations,
            QuantityKind.TRANSITION_VARIABLE,
            EquationKind.TRANSITION_EQUATION,
        )
        #
        # Number of measurement equations must equal number of measurement variables
        _check_numbers_of_variables_equations(
            self.quantities,
            self.dynamic_equations,
            QuantityKind.MEASUREMENT_VARIABLE,
            EquationKind.MEASUREMENT_EQUATION,
        )
        #
        self.quantities = _quantities.reorder_by_kind(self.quantities, )
        self.dynamic_equations, self.steady_equations = _reorder_equations_by_kind(self.dynamic_equations, self.steady_equations, )
        _quantities.stamp_id(self.quantities, )
        _equations.stamp_id(self.dynamic_equations, )
        _equations.stamp_id(self.steady_equations, )
        #
        # For stochastic models, create a mapping from shock qid to std qid
        self.shock_qid_to_std_qid = (
            _create_shock_qid_to_std_qid(self.quantities, )
            if not self._flags.is_deterministic else {}
        )
        #
        # Finalize equations by replacing names with qid pointers
        name_to_qid = _quantities.create_name_to_qid(self.quantities, )
        _equations.finalize_equations(self.dynamic_equations, name_to_qid, )
        _equations.finalize_equations(self.steady_equations, name_to_qid, )
        #
        if check_syntax:
            _check_syntax(self.dynamic_equations, self._context, )
            _check_syntax(self.steady_equations, self._context, )
        #
        self._populate_min_max_shifts()
        self._populate_derived_attributes()
        #
        return self

    def _populate_derived_attributes(self, /, ) -> None:
        """
        """
        #
        # Descriptors for first-order systems
        self.dynamic_descriptor = _descriptors.Descriptor(self.dynamic_equations, self.quantities, self._context, )
        self.steady_descriptor = _descriptors.Descriptor(self.steady_equations, self.quantities, self._context, )
        #
        # Evaluators of LHS-minus-RHS in dynamic and steady equations
        self._plain_dynamic_equator = _equators.PlainEquator(
            _equations.generate_equations_of_kind(self.dynamic_equations, _PLAIN_EQUATOR_EQUATION, ),
            context=self._context,
        )
        self._plain_steady_equator = _equators.PlainEquator(
            _equations.generate_equations_of_kind(self.steady_equations, _PLAIN_EQUATOR_EQUATION, ),
            context=self._context,
        )
        #
        # Steady autovalue updater
        steady_autovalues = tuple(_equations.generate_equations_of_kind(
            self.steady_equations,
            kind=EquationKind.STEADY_AUTOVALUES,
        ))
        _create_steady_autovalue_updater(self, steady_autovalues, )

    def copy(self, /, ) -> Self:
        """
        """
        # TODO: Optimize
        return _cp.deepcopy(self, )

    @property
    def num_transition_equations(self, /, ) -> int:
        """
        """
        return sum(
            1 for e in self.dynamic_equations
            if e.kind is EquationKind.TRANSITION_EQUATION
        )

    @property
    def num_measurement_equations(self, /, ) -> int:
        """
        """
        return sum(
            1 for e in self.dynamic_equations
            if e.kind is EquationKind.MEASUREMENT_EQUATION
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

    def __getstate__(self, /, ) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in self._serialized_slots
        }

    def __setstate__(self, state: dict[str, Any], ) -> None:
        for k in self._serialized_slots:
            setattr(self, k, state[k])
        self._populate_derived_attributes()

    def to_portable(self, /, ) -> dict[str, Any]:
        return {
            "description": str(self.get_description()),
            "flags": self._flags.to_portable(),
            "quantities": _quantities.to_portable(self.quantities, ),
            "equations": _equations.to_portable(self.dynamic_equations, self.steady_equations, ),
            "context": _contexts.to_portable(self._context, ),
        }

    @classmethod
    def from_portable(klass, portable: dict[str, Any], /, ) -> Self:
        description = str(portable["description"], )
        flags = portable["flags"]
        quantities = _quantities.from_portable(portable["quantities"], )
        dynamic_equations, steady_equations = _equations.from_portable(portable["equations"], )
        context = _contexts.from_portable(portable["context"], )
        source = ModelSource(
            quantities=quantities,
            dynamic_equations=dynamic_equations,
            steady_equations=steady_equations,
            context=_contexts.from_portable(portable["context"], ),
            description=description,
            **flags,
        )
        return klass.from_source(source, check_syntax=False, )

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
    quantities: Iterable[Quantity],
    /,
) -> dict[int, int]:
    """
    """
    #[
    name_to_qid = _quantities.create_name_to_qid(quantities, )
    qid_to_name = _quantities.create_qid_to_name(quantities, )
    kind = QuantityKind.ANY_SHOCK
    all_shock_qids = tuple(_quantities.generate_qids_by_kind(quantities, kind))
    return {
        shock_qid: name_to_qid[std_name_from_shock_name(qid_to_name[shock_qid])]
        for shock_qid in all_shock_qids
    }
    #]


def _create_std_for_shock(
    shock: Quantity,
    std_kind: QuantityKind,
    entry: int,
) -> Quantity:
    """
    """
    #[
    std_name = std_name_from_shock_name(shock.human, )
    std_description = std_description_from_shock_description(shock.description or shock.human, )
    return Quantity(
        id=None,
        human=std_name,
        kind=std_kind,
        logly=None,
        description=std_description,
        entry=entry,
    )
    #]


def _collect_undeclared_names(
    quantities: tuple[Quantity, ...],
    equations: tuple[Equation, ...],
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
) -> tuple[Quantity, ...] | NoReturn:
    """
    """
    #[
    if not undeclared_names:
        return tuple()
    autodeclare_kind = QuantityKind.from_keyword(autodeclare_as, )
    return (
        Quantity(
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
    quantities: tuple[Quantity, ...],
    equations: tuple[Equation, ...],
    quantity_kind: QuantityKind,
    equation_kind: EquationKind,
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
    custom_default_std: Real | None,
    flags: Flags,
) -> Real:
    """
    """
    #[
    if custom_default_std is not None:
        return float(custom_default_std)
    elif flags.is_linear:
        return _DEFAULT_STD_LINEAR
    else:
        return _DEFAULT_STD_NONLINEAR
    #]


_STEADY_AUTOVALUE_PATTERN = _re.compile(
    r"-\(" + ZERO_SHIFT_TOKEN_PATTERN + r"\)(.*)"
)


def _create_steady_autovalue_updater(
    self,
    steady_autovalues: Iterable[Equation],
    /,
) -> tuple[int, str]:
    """
    """
    #[
    lhs_qids = []
    rhs_xtrings = []
    for i in steady_autovalues:
        match = _STEADY_AUTOVALUE_PATTERN.match(i.xtring, )
        lhs_qid = int(match.group(1))
        rhs_xtring = match.group(2)
        lhs_qids.append(lhs_qid, )
        rhs_xtrings.append(rhs_xtring, )

    if not lhs_qids:
        self.update_steady_autovalues_in_variant = None
        return

    joined_rhs_xtrings = "(" + "  ,  ".join(rhs_xtrings, ) + " , )"
    func, func_str, *_ = _makers.make_function(
        "__equator",
        _equators.EQUATOR_ARGS,
        joined_rhs_xtrings,
        self._context,
    )
    num_columns = self._max_shift - self._min_shift + 1 
    shift_in_first_column = self._min_shift
    t = -self._min_shift
    qid_to_logly = _quantities.create_qid_to_logly(self.quantities, )

    def update_steady_autovalues(variant, ):
        steady_array = variant.create_steady_array(
            qid_to_logly,
            num_columns=num_columns,
            shift_in_first_column=shift_in_first_column,
        )
        values = _np.vstack(func(steady_array, t, ), )
        variant.update_levels_from_array(values, lhs_qids, )

    self.update_steady_autovalues_in_variant = update_steady_autovalues

    #]


def _reorder_equations_by_kind(
    dynamic_equations: tuple[Equation],
    steady_equations: tuple[Equation],
) -> tuple[tuple[Equation], tuple[Equation]]:
    """
    Reorder dynamic and steady equations at the same time to guarantee the same order for both
    """
    #[
    zipped = zip(dynamic_equations, steady_equations)
    zipped = sorted(zipped, key=lambda x: x[0].kind.value, )
    return zip(*zipped)
    #]

