"""
Simultaneous models
"""


#[

from __future__ import annotations

from typing import Self, Any, TypeAlias, Literal
from types import EllipsisType
from collections.abc import Iterable, Iterator, Callable
from numbers import Number
import numpy as _np
import itertools as _it
import functools as _ft
import documark as _dm

from .. import has_invariant as _has_invariant
from .. import has_variants as _has_variants
from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from ..conveniences import iterators as _iterators
from ..parsers import common as _pc
from ..databoxes import main as _databoxes

from ..fords import solutions as _solutions
from ..fords import steadiers as _fs
from ..fords import descriptors as _descriptors
from ..fords import systems as _systems
from ..fords import kalmans as _kalmans
from ..fords import std_simulators as _std_simulators

from ._invariants import Invariant
from ._variants import Variant
from . import _covariances
from . import _flags
from . import _simulate
from . import _steady
from . import _logly
from . import _get
from . import _pretty
from . import _assigns
from . import _slatable_protocols
from . import _plannable_protocols
from . import _steady_boxable_protocols
from . import _io

#]


__all__ = [
    "Simultaneous",
    "Model",
]


_DEFAULT_SOLUTION_TOLERANCE = 1e-12


@_dm.reference(
    path=("structural_models", "simultaneous.md", ),
    categories={
        "constructor": "Creating new simultaneous models",
        "filtering": "Applying structural filters on models",
        "information": "Getting information about models",
        "parameters": "Manipulating model parameters",
        "property": None,
    },
)
class Simultaneous(
    _has_invariant.HasInvariantMixin,
    _has_variants.HasVariantsMixin,
    _kalmans.Mixin,
    _std_simulators.Mixin,

    _assigns.Inlay,
    _simulate.Inlay,
    _steady.Inlay,
    _logly.Inlay,
    _get.Inlay,
    _pretty.Inlay,
    _covariances.Inlay,
    _slatable_protocols.Inlay,
    _plannable_protocols.Inlay,
    _steady_boxable_protocols.Inlay,
    _io.Inlay,
):
    r"""
    ................................................................................
    ==Simultaneous models==

    This class represents simultaneous models, which are used to model dynamic
    systems that may exhibit complex interdependencies among variables.

    The `Simultaneous` class inherits from several mixins to provide a variety of
    capabilities, including equation resolution, variance computation, simulation,
    and system analysis. Internal attributes and operations ensure a robust
    implementation of simultaneous modeling.

    ### Input arguments ###
    ???+ input "invariant"
        An instance of `Invariant` or `None` specifying the fixed structure of the
        model equations.

    ???+ input "variants"
        A list of `Variant` objects or `None`, representing different versions of
        the model setup.

    ### Returns ###
    ???+ returns "self"
    An instance of the `Simultaneous` class.
    ................................................................................
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    def __init__(
        self,
        invariant: Invariant | None = None,
        variants: list[Variant] | None = None,
    ) -> None:
        r"""
        ................................................................................
        ==Initialize Simultaneous object==

        The constructor sets up the invariant and variants of the simultaneous
        model. It ensures that variants are provided in a usable format, defaulting
        to an empty list if none are supplied.

        ### Input arguments ###
        ???+ input "invariant"
            An instance of `Invariant` or `None` specifying the fixed structure of
            the model equations.

        ???+ input "variants"
            A list of `Variant` objects or `None`, representing different versions
            of the model setup.

        ### Returns ###
        ???+ returns "None"
        This constructor does not return any value.

        ### Example ###
        ```python
            model = Simultaneous(invariant=my_invariant, variants=my_variants)
        ```
        ................................................................................
        """
        self._invariant = invariant
        self._variants = variants or []

    @classmethod
    def skeleton(
        klass,
        other,
        /,
    ) -> Self:
        r"""
        ................................................................................
        ==Create a skeleton copy of an existing Simultaneous model==

        This method creates a new `Simultaneous` model instance with the same
        invariant structure as an existing model. The resulting object shares
        structural attributes but does not duplicate the variants.

        ### Input arguments ###
        ???+ input "klass"
            The class type, typically `Simultaneous`, used to instantiate the skeleton.

        ???+ input "other"
            An existing `Simultaneous` object whose invariant structure will be copied.

        ### Returns ###
        ???+ returns "self"
        A new `Simultaneous` instance with a copied invariant structure.

        ### Example ###
        ```python
            skeleton_model = Simultaneous.skeleton(existing_model)
        ```
        ................................................................................
        """
        self = klass()
        self._invariant = other._invariant
        return self

    @classmethod
    @_dm.reference(category="constructor", call_name="Simultaneous.from_file", )
    def from_file(klass, *args, **kwargs, ) -> _sources.SourceMixinProtocol:
        r"""
        ................................................................................
        ==Create Simultaneous model object from source file or files==

        This method reads and parses one or more source files containing model source
        code and creates a `Simultaneous` model object. It supports context and
        descriptions to allow custom configurations during instantiation.

        ### Input arguments ###
        ???+ input "file_names"
            The name of the model source file or list of file names to combine in
            order.

        ???+ input "context"
            A dictionary supplying values for preparsing commands and custom equation
            functions.

        ???+ input "description"
            A description of the model as a string.

        ### Returns ###
        ???+ returns "self"
        A `Simultaneous` model object created from the source files.

        ### Example ###
        ```python
            model = Simultaneous.from_file(
                file_names=["model1.txt", "model2.txt"],
                context=my_context,
                description="Example model"
            )
        ```
        ................................................................................
        """
        return _sources.from_file(klass, *args, **kwargs, )

    @classmethod
    @_dm.reference(category="constructor", call_name="Simultaneous.from_string",)
    def from_string(klass, *args, **kwargs, ) -> _sources.SourceMixinProtocol:
        r"""
        ................................................................................
        ==Create Simultaneous model from string==

        This method reads and parses a string containing model source code and creates
        a `Simultaneous` model object. It functions similarly to
        `Simultaneous.from_file`.

        ### Input arguments ###
        ???+ input "string"
            The string containing model source code.

        ???+ input "context"
            A dictionary supplying values for preparsing commands and custom equation
            functions.

        ???+ input "description"
            A description of the model as a string.

        ### Returns ###
        ???+ returns "self"
        A `Simultaneous` model object created from the source string.

        ### Example ###
        ```python
            model = Simultaneous.from_string(
                string="model source code",
                context=my_context,
                description="Example model"
            )
        ```
        ................................................................................
        """
        return _sources.from_string(klass, *args, **kwargs, )

    def copy(self, /, ) -> Self:
        r"""
        ................................................................................
        ==Create a quasi-deep copy of the Simultaneous model==

        This method creates a copy of the `Simultaneous` model. The copied object
        includes its own set of variants, which are also independently duplicated.

        ### Returns ###
        ???+ returns "self"
        A new `Simultaneous` object with copied attributes and variants.

        ### Example ###
        ```python
            model_copy = existing_model.copy()
        ```
        ................................................................................
        """
        new = type(self)()
        new._invariant = self._invariant.copy()
        new._variants = [ variant.copy() for variant in self._variants ]
        return new

    def __getitem__(
        self,
        request: str | int,
        /,
    ):
        r"""
        ................................................................................
        ==Access elements of the Simultaneous model==

        This method implements indexed access to the `Simultaneous` model's
        attributes. String indices retrieve values of quantities, while integer
        indices retrieve specific variants.

        ### Input arguments ###
        ???+ input "request"
            A string specifying the quantity name or an integer specifying the variant
            index.

        ### Returns ###
        ???+ returns "value"
        The value associated with the requested quantity or variant.

        ### Example ###
        ```python
            value = model["quantity_name"]
            variant = model[0]
        ```
        ................................................................................
        """
        if isinstance(request, str):
            return self.get_value(request, )
        else:
            return self.get_variant(request, )

    def __setitem__(
        self,
        reference: str,
        value: Any,
        /,
    ) -> None:
        r"""
        ................................................................................
        ==Assign values to quantities in the Simultaneous model==

        This method allows direct assignment of values to specific model quantities
        using their names as string keys. If the key is not a string, an exception is
        raised.

        ### Input arguments ###
        ???+ input "reference"
            A string representing the name of the quantity to which the value is to
            be assigned.

        ???+ input "value"
            The value to be assigned to the specified quantity.

        ### Example ###
        ```python
            model["quantity_name"] = 42
        ```
        ................................................................................
        """
        if isinstance(reference, str):
            self._assign({reference: value}, )
        else:
            raise NotImplementedError("Assigning model variants not implemented yet", )

    def __repr__(self, /, ) -> str:
        r"""
        ................................................................................
        ==String representation of the Simultaneous model==

        This method returns a detailed string representation of the `Simultaneous`
        model object, including key attributes such as description, number of
        variants, equations, and lead/lag information.

        ### Returns ###
        ???+ returns "str"
        A formatted string describing the `Simultaneous` model.

        ### Example ###
        ```python
            print(model)
        ```
        ................................................................................
        """
        indented = " " * 4
        return "\n".join((
            f"",
            f"<{self.__class__.__name__} at {id(self):#x}>",
            f"[Description: \"{self.get_description()}\"]",
            f"[Num variants: {self.num_variants}]",
            f"[Num transition, measurement equations: {self.num_transition_equations}, {self.num_measurement_equations}]",
            f"[Max lag, lead: t{self.max_lag:+g}, t{self.max_lead:+g}]",
            f"",
        ))

    def get_value(
        self,
        name: str,
        /,
    ) -> Any:
        r"""
        ................................................................................
        ==Retrieve the value of a specified quantity==

        This method fetches the value of a model quantity based on its name. If the
        quantity name is invalid, an exception is raised.

        ### Input arguments ###
        ???+ input "name"
            A string specifying the name of the quantity whose value is to be
            retrieved.

        ### Returns ###
        ???+ returns "Any"
        The value associated with the specified quantity.

        ### Example ###
        ```python
            value = model.get_value("quantity_name")
        ```
        ................................................................................
        """
        quantities = self._invariant.quantities
        qids, invalid_names = _quantities.lookup_qids_by_name(quantities, (name, ), )
        if invalid_names:
            raise _wrongdoings.IrisPieCritical(f"Invalid model name \"{invalid_names[0]}\"", )
        return _has_variants.unpack_singleton(
            self._get_values_as_dict("levels", qids, )[name],
            self.is_singleton,
        )

    def change_logly(
        self,
        new_logly: bool,
        some_names: Iterable[str] | None = None,
        /
    ) -> None:
        r"""
        ................................................................................
        ==Change the log-status of specified quantities==

        This method modifies the logarithmic status (log-status) of selected model
        quantities. By default, all quantities with a non-`None` log-status are
        affected.

        ### Input arguments ###
        ???+ input "new_logly"
            A boolean value specifying the new log-status (`True` for logarithmic,
            `False` for non-logarithmic).

        ???+ input "some_names"
            An optional iterable of strings specifying the names of quantities whose
            log-status is to be changed. If `None`, all quantities with a log-status
            are affected.

        ### Example ###
        ```python
            model.change_logly(new_logly=True, some_names=["quantity1", "quantity2"])
        ```
        ................................................................................
        """
        some_names = set(some_names) if some_names else None
        qids = [
            qty.id
            for qty in self._invariant.quantities
            if qty.logly is not None and (some_names is None or qty.human in some_names)
        ]
        self._invariant.quantities = _quantities.change_logly(self._invariant.quantities, new_logly, qids)

    @property
    @_dm.reference(category="property", )
    def is_linear(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the model is declared as linear==

        This property determines whether the `Simultaneous` model is linear based on
        the flags in its invariant structure.

        ### Returns ###
        ???+ returns "bool"
        A boolean value: `True` if the model is linear, `False` otherwise.

        ### Example ###
        ```python
            linear_status = model.is_linear
        ```
        ................................................................................
        """
        return self._invariant._flags.is_linear

    @property
    @_dm.reference(category="property", )
    def is_flat(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the model is declared as flat==

        This property determines whether the `Simultaneous` model is flat, as specified
        by the flags in its invariant structure.

        ### Returns ###
        ???+ returns "bool"
        A boolean value: `True` if the model is flat, `False` otherwise.

        ### Example ###
        ```python
            flat_status = model.is_flat
        ```
        ................................................................................
        """
        return self._invariant._flags.is_flat

    @property
    @_dm.reference(category="property", )
    def is_deterministic(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the model is declared as deterministic==

        This property checks whether the `Simultaneous` model operates under
        deterministic assumptions, based on its invariant flags.

        ### Returns ###
        ???+ returns "bool"
        A boolean value: `True` if the model is deterministic, `False` otherwise.

        ### Example ###
        ```python
            deterministic_status = model.is_deterministic
        ```
        ................................................................................
        """
        return self._invariant._flags.is_deterministic

    @property
    @_dm.reference(category="property", )
    def num_transition_equations(self, /, ) -> int:
        r"""
        ................................................................................
        ==Retrieve the number of transition equations==

        This property returns the total count of transition equations defined in the
        `Simultaneous` model.

        ### Returns ###
        ???+ returns "int"
        An integer representing the number of transition equations in the model.

        ### Example ###
        ```python
            num_transitions = model.num_transition_equations
        ```
        ................................................................................
        """
        return self._invariant.num_transition_equations

    @property
    @_dm.reference(category="property", )
    def num_measurement_equations(self, /, ) -> int:
        r"""
        ................................................................................
        ==Retrieve the number of measurement equations==

        This property returns the total count of measurement equations defined in the
        `Simultaneous` model.

        ### Returns ###
        ???+ returns "int"
        An integer representing the number of measurement equations in the model.

        ### Example ###
        ```python
            num_measurements = model.num_measurement_equations
        ```
        ................................................................................
        """
        return self._invariant.num_measurement_equations

    @property
    @_dm.reference(category="property", )
    def max_lag(self, /, ) -> int:
        r"""
        ................................................................................
        ==Retrieve the maximum lag in the model==

        This property returns the maximum lag (negative or zero) present in the
        `Simultaneous` model's equations.

        ### Returns ###
        ???+ returns "int"
        An integer representing the maximum lag in the model.

        ### Example ###
        ```python
            maximum_lag = model.max_lag
        ```
        ................................................................................
        """
        return self._invariant._min_shift

    @property
    @_dm.reference(category="property", )
    def max_lead(self, /, ) -> int:
        r"""
        ................................................................................
        ==Retrieve the maximum lead in the model==

        This property returns the maximum lead (positive or zero) present in the
        `Simultaneous` model's equations.

        ### Returns ###
        ???+ returns "int"
        An integer representing the maximum lead in the model.

        ### Example ###
        ```python
            maximum_lead = model.max_lead
        ```
        ................................................................................
        """
        return self._invariant._max_shift

    @property
    def solution_vectors(self, /, ) -> _descriptors.SolutionVectors:
        r"""
        ................................................................................
        ==Retrieve solution vectors for the model==

        This property provides access to the solution vectors used in the dynamic
        descriptor of the `Simultaneous` model.

        ### Returns ###
        ???+ returns "SolutionVectors"
        An instance of `_descriptors.SolutionVectors` containing the solution
        vectors.

        ### Example ###
        ```python
            solution_vectors = model.solution_vectors
        ```
        ................................................................................
        """
        return self._invariant.dynamic_descriptor.solution_vectors

    @property
    def quantities(self, /, ) -> tuple[_quantities.Quantity]:
        r"""
        ................................................................................
        ==Access the quantities in the model==

        This property returns a tuple containing all the quantities defined in the
        `Simultaneous` model.

        ### Returns ###
        ???+ returns "tuple[Quantity]"
        A tuple of `_quantities.Quantity` objects representing the quantities in the
        model.

        ### Example ###
        ```python
            quantities = model.quantities
        ```
        ................................................................................
        """
        return self._invariant.quantities

    @property
    def shock_qid_to_std_qid(self, /, ) -> dict[int, int]:
        r"""
        ................................................................................
        ==Map shock quantity IDs to standard deviation quantity IDs==

        This property provides a dictionary that maps shock quantity IDs to their
        corresponding standard deviation quantity IDs.

        ### Returns ###
        ???+ returns "dict[int, int]"
        A dictionary where the keys are shock quantity IDs, and the values are
        standard deviation quantity IDs.

        ### Example ###
        ```python
            mapping = model.shock_qid_to_std_qid
        ```
        ................................................................................
        """
        return self._invariant.shock_qid_to_std_qid

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        r"""
        ................................................................................
        ==Create a mapping from quantity names to IDs==

        This method generates a dictionary mapping the names of quantities to their
        corresponding IDs.

        ### Returns ###
        ???+ returns "dict[str, int]"
        A dictionary where the keys are quantity names, and the values are their
        corresponding IDs.

        ### Example ###
        ```python
            name_to_id = model.create_name_to_qid()
        ```
        ................................................................................
        """
        return _quantities.create_name_to_qid(self._invariant.quantities)

    def create_qid_to_name(self, /, ) -> dict[int, str]:
        r"""
        ................................................................................
        ==Create a mapping from quantity IDs to names==

        This method generates a dictionary mapping the IDs of quantities to their
        corresponding names.

        ### Returns ###
        ???+ returns "dict[int, str]"
        A dictionary where the keys are quantity IDs, and the values are their
        corresponding names.

        ### Example ###
        ```python
            id_to_name = model.create_qid_to_name()
        ```
        ................................................................................
        """
        return _quantities.create_qid_to_name(self._invariant.quantities)

    def create_qid_to_kind(self, /, ) -> dict[int, str]:
        r"""
        ................................................................................
        ==Create a mapping from quantity IDs to their kinds==

        This method generates a dictionary mapping the IDs of quantities to their
        corresponding kinds, such as parameters or shocks.

        ### Returns ###
        ???+ returns "dict[int, str]"
        A dictionary where the keys are quantity IDs, and the values are their kinds.

        ### Example ###
        ```python
            id_to_kind = model.create_qid_to_kind()
        ```
        ................................................................................
        """
        return _quantities.create_qid_to_kind(self._invariant.quantities)

    def create_qid_to_description(self, /, ) -> dict[int, str]:
        r"""
        ................................................................................
        ==Create a mapping from quantity IDs to their descriptions==

        This method generates a dictionary mapping the IDs of quantities to their
        descriptive text.

        ### Returns ###
        ???+ returns "dict[int, str]"
        A dictionary where the keys are quantity IDs, and the values are their
        descriptions.

        ### Example ###
        ```python
            id_to_description = model.create_qid_to_description()
        ```
        ................................................................................
        """
        return _quantities.create_qid_to_description(self._invariant.quantities)

    def create_steady_array(
        self,
        variant: Variant | None = None,
        **kwargs,
    ) -> _np.ndarray:
        r"""
        ................................................................................
        ==Create a steady-state array for the model==

        This method generates an array representing the steady-state values of model
        quantities for a specified variant. If no variant is provided, the default
        variant is used.

        ### Input arguments ###
        ???+ input "variant"
            An optional `Variant` object. If `None`, the default variant of the model
            is used.

        ???+ input "**kwargs"
            Additional keyword arguments passed to the array creation logic.

        ### Returns ###
        ???+ returns "_np.ndarray"
        A NumPy array containing the steady-state values of model quantities.

        ### Example ###
        ```python
            steady_array = model.create_steady_array(variant=my_variant)
        ```
        ................................................................................
        """
        qid_to_logly = self.create_qid_to_logly()
        if variant is None:
            variant = self._variants[0]
        return variant.create_steady_array(qid_to_logly, **kwargs, )

    def create_zero_array(
        self,
        variant: Variant | None = None,
        **kwargs,
    ) -> _np.ndarray:
        r"""
        ................................................................................
        ==Create a zero array for the model==

        This method generates an array where all model quantities are initialized to
        zero for a specified variant. If no variant is provided, the default variant
        is used.

        ### Input arguments ###
        ???+ input "variant"
            An optional `Variant` object. If `None`, the default variant of the model
            is used.

        ???+ input "**kwargs"
            Additional keyword arguments passed to the array creation logic.

        ### Returns ###
        ???+ returns "_np.ndarray"
        A NumPy array where all values are zero.

        ### Example ###
        ```python
            zero_array = model.create_zero_array(variant=my_variant)
        ```
        ................................................................................
        """
        qid_to_logly = self.create_qid_to_logly()
        if variant is None:
            variant = self._variants[0]
        return variant.create_zero_array(qid_to_logly, **kwargs, )

    def create_some_array(
        self,
        /,
        deviation: bool,
        **kwargs,
    ) -> _np.ndarray:
        r"""
        ................................................................................
        ==Create an array with steady-state or zero values==

        This method generates an array based on the `deviation` flag. If `deviation`
        is `True`, a zero array is created. Otherwise, a steady-state array is
        returned.

        ### Input arguments ###
        ???+ input "deviation"
            A boolean value. If `True`, a zero array is created. If `False`, a
            steady-state array is returned.

        ???+ input "**kwargs"
            Additional keyword arguments passed to the array creation logic.

        ### Returns ###
        ???+ returns "_np.ndarray"
        A NumPy array with either steady-state or zero values, depending on the
        `deviation` flag.

        ### Example ###
        ```python
            some_array = model.create_some_array(deviation=False)
        ```
        ................................................................................
        """
        if not deviation:
            return self.create_steady_array(**kwargs, )
        else:
            return self.create_zero_array(**kwargs, )

    def _enforce_assignment_rules(self, variant, /, ) -> None:
        r"""
        ................................................................................
        ==Enforce assignment rules for model variants==

        This private method ensures that specific rules are applied to a `Variant`
        during assignment operations. These include resetting shock levels to zero
        and removing changes from non-loggable quantities.

        ### Input arguments ###
        ???+ input "variant"
            A `Variant` object for which assignment rules are to be enforced.

        ### Example ###
        ```python
            model._enforce_assignment_rules(my_variant)
        ```
        ................................................................................
        """
        #
        # Reset levels of shocks to zero
        name_to_qid = self.create_name_to_qid()
        shock_qids = _quantities.generate_qids_by_kind(
            self._invariant.quantities, _quantities.QuantityKind.ANY_SHOCK,
        )
        zero_shocks = { i: 0 for i in shock_qids }
        variant.update_values_from_dict(zero_shocks, )
        #
        # Remove changes from quantities that are not loggable
        non_loggables = _quantities.generate_qids_by_kind(
            self._invariant.quantities, ~_sources.LOGGABLE_VARIABLE,
        )
        assign_non_loggables = { qid: (..., _np.nan, ) for qid in non_loggables }
        variant.update_values_from_dict(assign_non_loggables, )

    def systemize(
        self,
        /,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> Iterable[_systems.System]:
        r"""
        ................................................................................
        ==Create unsolved first-order systems for each model variant==

        This method constructs a first-order system representation for each variant
        of the `Simultaneous` model. The resulting systems are not solved.

        ### Input arguments ###
        ???+ input "unpack_singleton"
            A boolean value specifying whether a single variant system should be
            unpacked. Defaults to `True`.

        ???+ input "**kwargs"
            Additional keyword arguments passed to the system creation logic.

        ### Returns ###
        ???+ returns "Iterable[_systems.System]"
        An iterable of `_systems.System` objects representing the unsolved systems.

        ### Example ###
        ```python
            systems = model.systemize(unpack_singleton=True)
        ```
        ................................................................................
        """
        model_flags = self.resolve_flags(**kwargs, )
        system = [
            self._systemize(variant, self._invariant.dynamic_descriptor, model_flags, )
            for variant in self._variants
        ]
        return _has_variants.unpack_singleton(
            system, self.is_singleton,
            unpack_singleton=unpack_singleton,
        )

    def _systemize(
        self,
        variant: Variant,
        descriptor: _descriptors.Descriptor,
        model_flags: flags.Flags,
        /,
    ) -> _systems.System:
        r"""
        ................................................................................
        ==Create an unsolved first-order system for a specific variant==

        This private method generates an unsolved first-order system for a given
        variant of the `Simultaneous` model. The system incorporates model flags and
        descriptors to construct the necessary equations.

        ### Input arguments ###
        ???+ input "variant"
            The `Variant` object for which the system is to be constructed.

        ???+ input "descriptor"
            A `_descriptors.Descriptor` object representing the model's dynamic
            descriptor.

        ???+ input "model_flags"
            A `_flags.Flags` object defining the configuration and properties of the
            model.

        ### Returns ###
        ???+ returns "_systems.System"
        An unsolved `_systems.System` object representing the model variant.

        ### Example ###
        ```python
            system = model._systemize(variant=my_variant, descriptor=desc, model_flags=flags)
        ```
        ................................................................................
        """
        min_shift = self._invariant._min_shift
        max_shift = self._invariant._max_shift
        num_columns = -min_shift + 1 + max_shift
        qid_to_logly = self.create_qid_to_logly()
        #
        if model_flags.is_linear:
            data_array = variant.create_zero_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=min_shift, )
            data_array_lagged = None
        else:
            data_array = variant.create_steady_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=min_shift, )
            data_array_lagged = variant.create_steady_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=min_shift-1, )
        #
        column_offset = -min_shift
        return _systems.System(
            descriptor, data_array,
            model_flags, data_array_lagged, column_offset,
        )

    def solve(
        self,
        /,
        clip_small: bool = False,
        return_info: bool = False,
        unpack_singleton: bool = True,
        tolerance: Real = _DEFAULT_SOLUTION_TOLERANCE,
        **kwargs,
    ) -> dict[str, Any]:
        r"""
        ................................................................................
        ==Solve the first-order system for all model variants==

        This method calculates the first-order solution for each variant within the
        `Simultaneous` model. It handles unit root exceptions and returns solution
        information if requested.

        ### Input arguments ###
        ???+ input "clip_small"
            A boolean specifying whether to clip small values in the solution.

        ???+ input "return_info"
            A boolean specifying whether to return solution information.

        ???+ input "unpack_singleton"
            A boolean specifying whether to unpack single-variant results.

        ???+ input "tolerance"
            A real number specifying the tolerance for solution convergence.

        ???+ input "**kwargs"
            Additional keyword arguments for resolving flags or configuration.

        ### Returns ###
        ???+ returns "dict[str, Any]"
        A dictionary containing solution information if `return_info` is `True`.
        Otherwise, no value is returned.

        ### Example ###
        ```python
            solution_info = model.solve(return_info=True)
        ```
        ................................................................................
        """
        model_flags = self.resolve_flags(**kwargs, )
        out_info = [
            self._solve(
                self_v,
                vid,
                model_flags,
                clip_small=clip_small,
                tolerance=tolerance,
            )
            for vid, self_v in enumerate(self._variants, )
        ]
        if return_info:
            out_info = _has_variants.unpack_singleton(
                out_info, self.is_singleton,
                unpack_singleton=unpack_singleton,
            )
            return out_info
        else:
            return

    def _solve(
        self,
        variant: Variant,
        vid: int,
        model_flags: flags.Flags,
        /,
        clip_small: bool,
        tolerance: Real,
    ) -> None:
        r"""
        ................................................................................
        ==Calculate the first-order solution for a single model variant==

        This private method solves the first-order system for one variant of the
        `Simultaneous` model. It handles unit root exceptions and assigns the
        computed solution to the variant.

        ### Input arguments ###
        ???+ input "variant"
            The `Variant` object for which the solution is calculated.

        ???+ input "vid"
            An integer specifying the index of the variant.

        ???+ input "model_flags"
            A `_flags.Flags` object defining the configuration of the model.

        ???+ input "clip_small"
            A boolean specifying whether to clip small values in the solution.

        ???+ input "tolerance"
            A real number specifying the tolerance for solution convergence.

        ### Example ###
        ```python
            model._solve(variant=my_variant, vid=0, model_flags=flags, clip_small=False, tolerance=1e-12)
        ```
        ................................................................................
        """
        variant_header = f"[Variant {vid}]"
        system = self._systemize(
            variant,
            self._invariant.dynamic_descriptor,
            model_flags,
        )
        try:
            variant.solution = _solutions.Solution.from_system(
                self._invariant.dynamic_descriptor,
                system,
                clip_small=clip_small,
                tolerance=tolerance,
            )
        except _solutions.UnitRootException:
            raise _wrongdoings.IrisPieCritical(
                f"{variant_header} Inconsistency in classification of unit roots; "
                "modify (increase) the tolerance level",
            )
        info = {}
        #
        return info

    def _choose_plain_equator(
        self,
        equation_switch: Literal["dynamic", "steady", ],
        /,
    ) -> Callable | None:
        r"""
        ................................................................................
        ==Select plain equator based on equation type==

        This private method selects a plain equator based on the specified equation
        type (`dynamic` or `steady`).

        ### Input arguments ###
        ???+ input "equation_switch"
            A string literal specifying the type of equation: `"dynamic"` or `"steady"`.

        ### Returns ###
        ???+ returns "Callable | None"
        A callable object representing the plain equator or `None` if not found.

        ### Example ###
        ```python
            equator = model._choose_plain_equator(equation_switch="dynamic")
        ```
        ................................................................................
        """
        match equation_switch:
            case "dynamic":
                return self._invariant._plain_dynamic_equator
            case "steady":
                return self._invariant._plain_steady_equator

    def reset_stds(self, /, ) -> None:
        r"""
        ................................................................................
        ==Reset standard deviations of shocks to default values==

        This method initializes the standard deviations of all shocks in the model
        to their default values, as specified in the invariant structure.

        ### Example ###
        ```python
            model.reset_stds()
        ```
        ................................................................................
        """
        std_names = _quantities.generate_quantity_names_by_kind(
            self._invariant.quantities,
            _quantities.QuantityKind.ANY_STD,
        )
        dict_to_assign = {
            k: self._invariant._default_std
            for k in std_names
        }
        self.assign(**dict_to_assign, )

    @classmethod
    def from_source(
        klass,
        source: _sources.ModelSource,
        /,
        **kwargs,
    ) -> Self:
        r"""
        ................................................................................
        ==Create a Simultaneous model from a source==

        This class method creates a `Simultaneous` model object from a model source.
        It initializes the invariant and creates the initial variant.

        ### Input arguments ###
        ???+ input "source"
            A `_sources.ModelSource` object containing the source information for the
            model.

        ???+ input "**kwargs"
            Additional keyword arguments for the model creation.

        ### Returns ###
        ???+ returns "Self"
        A `Simultaneous` model object created from the source.

        ### Example ###
        ```python
            model = Simultaneous.from_source(source=my_source)
        ```
        ................................................................................
        """
        self = klass()
        self._invariant = Invariant.from_source(source, **kwargs, )
        initial_variant = Variant.from_source(
            self._invariant.quantities,
            self._invariant._flags.is_flat,
        )
        self._variants = [ initial_variant ]
        self.reset_stds()
        self._enforce_assignment_rules(self._variants[0], )
        return self

    def get_context(self, /, ) -> dict[str, Any]:
        r"""
        ................................................................................
        ==Retrieve the model's context==

        This method returns the context associated with the `Simultaneous` model,
        which includes preparsing definitions and custom function setups.

        ### Returns ###
        ???+ returns "dict[str, Any]"
        A dictionary containing the context of the model.

        ### Example ###
        ```python
            context = model.get_context()
        ```
        ................................................................................
        """
        return self._invariant._context

    def resolve_flags(self, /, **kwargs, ) -> _flags.Flags:
        r"""
        ................................................................................
        ==Resolve model flags with updates from keyword arguments==

        This method updates and resolves the `Flags` object for the model using
        specified keyword arguments. The updated flags reflect the dynamic behavior
        and configuration of the model.

        ### Input arguments ###
        ???+ input "**kwargs"
            Keyword arguments to update and resolve the flags.

        ### Returns ###
        ???+ returns "_flags.Flags"
        A `Flags` object representing the resolved flags for the model.

        ### Example ###
        ```python
            resolved_flags = model.resolve_flags(some_flag=True)
        ```
        ................................................................................
        """
        return _flags.Flags.update_from_kwargs(self.get_flags(), **kwargs)

    def __getstate__(self, /, ):
        r"""
        ................................................................................
        ==Serialize the model's state==

        This method serializes the `Simultaneous` model's state for pickling or other
        storage purposes.

        ### Returns ###
        ???+ returns "dict"
        A dictionary representing the serialized state of the model.

        ### Example ###
        ```python
            state = model.__getstate__()
        ```
        ................................................................................
        """
        return {
            "_invariant": self._invariant,
            "_variants": self._variants,
        }

    def __setstate__(self, state, /, ):
        r"""
        ................................................................................
        ==Restore the model's state==

        This method restores the state of the `Simultaneous` model from a serialized
        representation.

        ### Input arguments ###
        ???+ input "state"
            A dictionary containing the serialized state of the model.

        ### Example ###
        ```python
            model.__setstate__(state)
        ```
        ................................................................................
        """
        self._invariant = state["_invariant"]
        self._variants = state["_variants"]

    def _serialize_to_portable(self, /, ) -> dict[str, Any]:
        r"""
        ................................................................................
        ==Serialize the model to a portable format==

        This private method converts the `Simultaneous` model into a dictionary
        representation suitable for storage or transport.

        ### Returns ###
        ???+ returns "dict[str, Any]"
        A dictionary containing the serialized representation of the model.

        ### Example ###
        ```python
            serialized_model = model._serialize_to_portable()
        ```
        ................................................................................
        """
        qid_to_name = self.create_qid_to_name()
        return {
            "source": self._invariant._serialize_to_portable(),
            "variants": [v._serialize_to_portable(qid_to_name, ) for v in self._variants],
        }

    #]


Model = Simultaneous

