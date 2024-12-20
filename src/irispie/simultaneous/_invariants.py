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
from ..incidences.main import ZERO_SHIFT_TOKEN_PATTERN
from .. import sources as _sources

from ._flags import Flags

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self, Callable, NoReturn, Any
    from numbers import Real
    from .. sources import ModelSource

#]


_DEFAULT_STD_LINEAR = 1
_DEFAULT_STD_NONLINEAR = 0.01


_PLAIN_EQUATOR_EQUATION = (
    EquationKind.TRANSITION_EQUATION
    | EquationKind.MEASUREMENT_EQUATION
)


_DEFAULT_STD_NAME_FORMAT = "std_{}"
_DEFAULT_STD_DESCRIPTION_FORMAT = "(Std) {}"


class Invariant(
    _descriptions.DescriptionMixin,
):
    r"""
    ................................................................................
    ==Class: Invariant==

    Represents the invariant portion of a Simultaneous object, managing quantities 
    and equations to ensure the model's consistency and validity. This class is a 
    critical component in mathematical modeling, especially for simulations that 
    involve dynamic and steady-state equations.

    The `Invariant` class initializes key attributes, processes model inputs, and 
    ensures that equations are correctly associated with their respective variables. 
    It also supports syntax validation and the generation of auxiliary quantities 
    and parameters.

    Inherits:
        - _descriptions.DescriptionMixin: Provides description handling capabilities.

    Attributes:
        - `_serialized_slots`: Persistent data elements.
        - `_derived_slots`: Derived and computed data attributes.
        - `__slots__`: Combines all defined slots for memory optimization.
    ................................................................................
    """
    #[

    _serialized_slots = (
        "quantities",
        "dynamic_equations",
        "steady_equations",
        "shock_qid_to_std_qid",
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
        r"""
        ................................................................................
        ==Method: __init__==

        Initializes an `Invariant` instance, setting up serialized and derived slots 
        with the provided keyword arguments. Attributes that are not explicitly 
        provided are initialized with default values.

        ### Input arguments ###
        ???+ input "kwargs"
            A set of key-value pairs corresponding to attribute names and their values. 
            Any unspecified attributes default to `None`.

        ### Example ###
        ```python
            inv = Invariant(quantities=[], dynamic_equations=[], steady_equations=[])
        ```
        ................................................................................
        """
        for n in self._serialized_slots:
            setattr(self, n, kwargs.get(n, ), )
        for n in self._derived_slots:
            setattr(self, n, None, )

    @classmethod
    def from_source(
        klass,
        source: ModelSource,
        /,
        check_syntax: bool = True,
        std_name_format: str = _DEFAULT_STD_NAME_FORMAT,
        std_description_format: str = _DEFAULT_STD_DESCRIPTION_FORMAT,
        autodeclare_as: str | None = None,
        default_std: Real | None = None,
        description: str | None = None,
        **kwargs,
    ) -> Self:
        r"""
        ................................................................................
        ==Method: from_source==

        Creates an `Invariant` instance from a `ModelSource` object. Processes model 
        attributes, equations, and context settings. Supports validation and 
        auto-declaration of undeclared names.

        ### Input arguments ###
        ???+ input "source: ModelSource"
            Source object containing model definitions, quantities, and equations.
        ???+ input "check_syntax: bool = True"
            Whether to validate syntax during initialization.
        ???+ input "std_name_format: str = _DEFAULT_STD_NAME_FORMAT"
            Format string for naming standard deviations.
        ???+ input "std_description_format: str = _DEFAULT_STD_DESCRIPTION_FORMAT"
            Format string for describing standard deviations.
        ???+ input "autodeclare_as: str | None = None"
            Optional keyword to auto-declare undeclared names.
        ???+ input "default_std: Real | None = None"
            Custom default value for standard deviations.
        ???+ input "description: str | None = None"
            Description for the `Invariant` instance.
        ???+ input "**kwargs"
            Additional parameters for internal configurations.

        ### Returns ###
        ???+ returns "Self"
            A new `Invariant` instance initialized with the provided `ModelSource`.

        ### Example ###
        ```python
            inv = Invariant.from_source(model_source, check_syntax=True)
        ```
        ................................................................................
        """
        self = klass()
        self.set_description(description, )
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
            self.quantities += tuple(_generate_stds_for_shocks(
                unanticipated_shocks,
                QuantityKind.UNANTICIPATED_STD,
                std_name_format,
                std_description_format,
                entry=len(self.quantities),
            ))
            self.quantities += tuple(_generate_stds_for_shocks(
                anticipated_shocks,
                QuantityKind.ANTICIPATED_STD,
                std_name_format,
                std_description_format,
                entry=len(self.quantities),
            ))
            self.quantities += tuple(_generate_stds_for_shocks(
                measurement_shocks,
                QuantityKind.MEASUREMENT_STD,
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
            _create_shock_qid_to_std_qid(self.quantities, std_name_format, )
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
        #
        self._populate_derived_attributes()
        #
        return self

    def _populate_derived_attributes(self, /, ) -> None:
        r"""
        ................................................................................
        ==Method: _populate_derived_attributes==

        Populates derived attributes of the `Invariant` instance, including equation 
        descriptors, plain equators, and steady autovalue updaters. This method 
        prepares the instance for further computations and ensures internal coherence.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            inv._populate_derived_attributes()
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Method: copy==

        Creates a deep copy of the `Invariant` instance. This method is useful when 
        an identical but independent copy of the object is required. All attributes, 
        including both serialized and derived slots, are duplicated.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "Self"
            A new `Invariant` instance that is a deep copy of the original.

        ### Example ###
        ```python
            inv_copy = inv.copy()
        ```
        ................................................................................
        """
        # TODO: Optimize
        return _cp.deepcopy(self, )

    @property
    def num_transition_equations(self, /, ) -> int:
        r"""
        ................................................................................
        ==Property: num_transition_equations==

        Calculates and returns the number of transition equations in the `Invariant` 
        instance. Transition equations are identified based on their kind.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "int"
            The count of transition equations.

        ### Example ###
        ```python
            num_transitions = inv.num_transition_equations
        ```
        ................................................................................
        """
        return sum(
            1 for e in self.dynamic_equations
            if e.kind is EquationKind.TRANSITION_EQUATION
        )

    @property
    def num_measurement_equations(self, /, ) -> int:
        r"""
        ................................................................................
        ==Property: num_measurement_equations==

        Calculates and returns the number of measurement equations in the `Invariant` 
        instance. Measurement equations are identified based on their kind.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "int"
            The count of measurement equations.

        ### Example ###
        ```python
            num_measurements = inv.num_measurement_equations
        ```
        ................................................................................
        """
        return sum(
            1 for e in self.dynamic_equations
            if e.kind is EquationKind.MEASUREMENT_EQUATION
        )

    def _populate_min_max_shifts(self, /, ) -> None:
        r"""
        ................................................................................
        ==Method: _populate_min_max_shifts==

        Computes the minimum and maximum shifts across all equations in the `Invariant` 
        instance. These values are used to manage time indexing in dynamic and steady 
        equations.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            inv._populate_min_max_shifts()
        ```
        ................................................................................
        """
        self._min_shift = _equations.get_min_shift_from_equations(
            self.dynamic_equations + self.steady_equations,
        )
        self._max_shift = _equations.get_max_shift_from_equations(
            self.dynamic_equations + self.steady_equations,
        )

    def __getstate__(self, /, ) -> dict[str, Any]:
        r"""
        ................................................................................
        ==Method: __getstate__==

        Serializes the `Invariant` instance into a dictionary format, capturing all 
        serialized slots. This method is primarily used for object persistence or 
        debugging.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "dict[str, Any]"
            A dictionary containing the serialized state of the `Invariant` instance.

        ### Example ###
        ```python
            state = inv.__getstate__()
        ```
        ................................................................................
        """
        return {
            k: getattr(self, k)
            for k in self._serialized_slots
        }

    def __setstate__(self, state: dict[str, Any], ) -> None:
        r"""
        ................................................................................
        ==Method: __setstate__==

        Restores the `Invariant` instance from a serialized state dictionary. This 
        method reinitializes all serialized attributes and populates derived attributes 
        to ensure consistency.

        ### Input arguments ###
        ???+ input "state: dict[str, Any]"
            A dictionary containing the serialized state of an `Invariant` instance.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            inv.__setstate__(state)
        ```
        ................................................................................
        """
        for k in self._serialized_slots:
            setattr(self, k, state[k])
        self._populate_derived_attributes()

    def _serialize_to_portable(self, /, ) -> dict[str, Any]:
        r"""
        ................................................................................
        ==Method: _serialize_to_portable==

        Converts the `Invariant` instance into a portable format. This serialized 
        dictionary can be used for sharing or storing the instance data.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "dict[str, Any]"
            A dictionary in portable format, representing the `Invariant` instance.

        ### Example ###
        ```python
            portable = inv._serialize_to_portable()
        ```
        ................................................................................
        """
        return _sources.serialize_to_portable(self, )

    #]


def _check_syntax(equations, function_context, /, ):
    r"""
    ................................................................................
    ==Function: _check_syntax==

    Validates the syntax of a set of equations within a given function context. This 
    function attempts to evaluate all equations collectively and isolates problematic 
    equations if a syntax error is detected.

    ### Input arguments ###
    ???+ input "equations"
        A collection of equations to validate.
    ???+ input "function_context"
        The execution context used for evaluating the equations.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _check_syntax(equations, context)
    ```
    ................................................................................
    """
    try:
        eval(_equations.create_equator_func_string(equations, ), )
    except:
        _catch_troublemakers(equations, function_context, )
    #]


def _catch_troublemakers(equations, function_context, /, ):
    r"""
    ................................................................................
    ==Function: _catch_troublemakers==

    Identifies and reports equations that fail syntax validation. Attempts to evaluate 
    each equation individually to pinpoint the ones causing errors.

    ### Input arguments ###
    ???+ input "equations"
        A collection of equations to evaluate.
    ???+ input "function_context"
        The execution context used for evaluating the equations.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _catch_troublemakers(equations, context)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Function: _success_creating_lambda==

    Tests whether a single equation can be successfully compiled into a lambda 
    function. Returns a boolean result indicating success or failure.

    ### Input arguments ###
    ???+ input "equation"
        The equation to test.
    ???+ input "function_context"
        The execution context for evaluating the equation.

    ### Returns ###
    ???+ returns "bool"
        `True` if the equation can be compiled successfully, `False` otherwise.

    ### Example ###
    ```python
        is_valid = _success_creating_lambda(equation, context)
    ```
    ................................................................................
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
    std_name_format: str,
    /,
) -> dict[int, int]:
    r"""
    ................................................................................
    ==Function: _create_shock_qid_to_std_qid==

    Maps each shock quantity ID (qid) to its corresponding standard deviation quantity 
    ID based on the given name format.

    ### Input arguments ###
    ???+ input "quantities: Iterable[Quantity]"
        The collection of quantities to process.
    ???+ input "std_name_format: str"
        The format string for standard deviation names.

    ### Returns ###
    ???+ returns "dict[int, int]"
        A dictionary mapping shock qids to standard deviation qids.

    ### Example ###
    ```python
        mapping = _create_shock_qid_to_std_qid(quantities, "std_{}")
    ```
    ................................................................................
    """
    #[
    name_to_qid = _quantities.create_name_to_qid(quantities, )
    qid_to_name = _quantities.create_qid_to_name(quantities, )
    kind = QuantityKind.ANY_SHOCK
    all_shock_qids = tuple(_quantities.generate_qids_by_kind(quantities, kind))
    return {
        shock_qid: name_to_qid[std_name_format.format(qid_to_name[shock_qid], )]
        for shock_qid in all_shock_qids
    }
    #]


def _generate_stds_for_shocks(
    shocks: Iterable[Quantity, ...],
    std_kind: QuantityKind,
    std_name_format: str,
    std_description_format: str,
    /,
    *,
    entry: int,
) -> tuple[Quantity, ...]:
    r"""
    ................................................................................
    ==Function: _generate_stds_for_shocks==

    Generates standard deviation quantities for a collection of shocks. Each new 
    quantity is assigned a specific kind, name, and description.

    ### Input arguments ###
    ???+ input "shocks: Iterable[Quantity, ...]"
        The collection of shock quantities.
    ???+ input "std_kind: QuantityKind"
        The kind of standard deviation quantities to generate.
    ???+ input "std_name_format: str"
        The format string for naming standard deviations.
    ???+ input "std_description_format: str"
        The format string for describing standard deviations.
    ???+ input "entry: int"
        The starting entry index for the generated quantities.

    ### Returns ###
    ???+ returns "tuple[Quantity, ...]"
        A tuple of generated standard deviation quantities.

    ### Example ###
    ```python
        stds = _generate_stds_for_shocks(shocks, std_kind, "std_{}", "(Std) {}")
    ```
    ................................................................................
    """
    #[
    return (
        Quantity(
            id=None,
            human=std_name_format.format(shock.human, ),
            kind=std_kind,
            logly=None,
            description=std_description_format.format(shock.description or shock.human, ),
            entry=entry,
        ) for shock in shocks
    )
    #]


def _collect_undeclared_names(
    quantities: tuple[Quantity, ...],
    equations: tuple[Equation, ...],
    /,
) -> tuple[str, ...]:
    r"""
    ................................................................................
    ==Function: _collect_undeclared_names==

    Identifies names used in equations that are not declared as quantities. This is 
    critical for ensuring that all variables and parameters are properly defined.

    ### Input arguments ###
    ???+ input "quantities: tuple[Quantity, ...]"
        A tuple of declared quantities.
    ???+ input "equations: tuple[Equation, ...]"
        A tuple of equations to check.

    ### Returns ###
    ???+ returns "tuple[str, ...]"
        A tuple of undeclared names found in the equations.

    ### Example ###
    ```python
        undeclared = _collect_undeclared_names(quantities, equations)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Function: _generate_quantities_for_undeclared_names==

    Creates quantities for names that are used in equations but not declared. This 
    function supports automatic declaration of such names.

    ### Input arguments ###
    ???+ input "undeclared_names: set[str]"
        The set of undeclared names to be processed.
    ???+ input "autodeclare_as: str | None"
        The kind of quantity to auto-declare the names as.
    ???+ input "entry: int"
        The starting entry index for the generated quantities.

    ### Returns ###
    ???+ returns "tuple[Quantity, ...]"
        A tuple of generated quantities.

    ### Example ###
    ```python
        quantities = _generate_quantities_for_undeclared_names(undeclared, "parameter", entry=10)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Function: _check_numbers_of_variables_equations==

    Validates that the number of variables of a specific kind matches the number of 
    equations of a corresponding kind. Raises an error if the counts do not match.

    ### Input arguments ###
    ???+ input "quantities: tuple[Quantity, ...]"
        A tuple of quantities to check.
    ???+ input "equations: tuple[Equation, ...]"
        A tuple of equations to check.
    ???+ input "quantity_kind: QuantityKind"
        The kind of quantities to count.
    ???+ input "equation_kind: EquationKind"
        The kind of equations to count.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _check_numbers_of_variables_equations(quantities, equations, QuantityKind.TRANSITION_VARIABLE, EquationKind.TRANSITION_EQUATION)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Function: _resolve_default_std==

    Determines the default standard deviation to use based on the provided custom 
    value or the model's linearity status.

    ### Input arguments ###
    ???+ input "custom_default_std: Real | None"
        A custom default standard deviation value. If `None`, the default is resolved 
        based on the `Flags`.
    ???+ input "flags: Flags"
        Flags that determine whether the model is linear or nonlinear.

    ### Returns ###
    ???+ returns "Real"
        The resolved default standard deviation.

    ### Example ###
    ```python
        default_std = _resolve_default_std(None, flags)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Function: _create_steady_autovalue_updater==

    Generates a function to update steady-state autovalues in a variant instance. The 
    function processes a collection of steady-state equations to derive updates for 
    their corresponding quantities.

    ### Input arguments ###
    ???+ input "self"
        The `Invariant` instance to which the updater belongs.
    ???+ input "steady_autovalues: Iterable[Equation]"
        A collection of steady-state autovalue equations to process.

    ### Returns ###
    ???+ returns "tuple[int, str]"
        A tuple containing details of the updated autovalue configuration.

    ### Example ###
    ```python
        _create_steady_autovalue_updater(inv, steady_autovalues)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Function: _reorder_equations_by_kind==

    Reorders dynamic and steady-state equations to ensure consistency in their order 
    based on equation kinds. Both dynamic and steady equations are sorted simultaneously.

    ### Input arguments ###
    ???+ input "dynamic_equations: tuple[Equation]"
        A tuple of dynamic equations to reorder.
    ???+ input "steady_equations: tuple[Equation]"
        A tuple of steady-state equations to reorder.

    ### Returns ###
    ???+ returns "tuple[tuple[Equation], tuple[Equation]]"
        A tuple containing the reordered dynamic and steady-state equations.

    ### Example ###
    ```python
        reordered_dynamic, reordered_steady = _reorder_equations_by_kind(dynamic, steady)
    ```
    ................................................................................
    """
    #[
    zipped = zip(dynamic_equations, steady_equations)
    zipped = sorted(zipped, key=lambda x: x[0].kind.value, )
    return zip(*zipped)
    #]

