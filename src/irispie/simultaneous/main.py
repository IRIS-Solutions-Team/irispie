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
from . import _covariances as _covariances
from . import _flags as _flags
from . import _simulate as _simulate
from . import _steady as _steady
from . import _logly as _logly
from . import _get as _get
from . import _pretty as _pretty
from . import _assigns as _assigns
from . import _slatable_protocols as _slatable_protocols
from . import _plannable_protocols as _plannable_protocols
from . import _io as _io

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
    _io.Inlay,
):
    """
················································································

`Simultaneous` models
======================

················································································
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
        """
        """
        self._invariant = invariant
        self._variants = variants or []

    @classmethod
    def skeleton(
        klass,
        other,
        /,
    ) -> Self:
        """
        """
        self = klass()
        self._invariant = other._invariant
        return self

    @classmethod
    @_dm.reference(category="constructor", call_name="Simultaneous.from_file", )
    def from_file(klass, *args, **kwargs, ) -> _sources.SourceMixinProtocol:
        """
················································································

==Create `Simultaneous` model object from source file or files==

```
self = Simultaneous.from_file(
    file_names,
    /,
    context=None,
    description="",
)
```

Read and parse one or more source files specified by `file_names` (a string
or a list of strings) with model source code, and create a `Simultaneous`
model object.


### Input arguments ###


???+ input "file_names"
    The name of the model source file from which the `Simultaneous` model object
    will be created, or a list of file names; if multiple file names are
    specified, they will all combined together in the given order.

???+ input "context"
    Dictionary supplying the values used in preparsing commands, and the
    definition of non-standard functions used in the equations.

???+ input "description"
    Desscription of the model specified as a text string.


### Returns ###


???+ returns "self"
`Simultaneous` model object created from the `file_names`.

················································································
        """
        return _sources.from_file(klass, *args, **kwargs, )

    @classmethod
    @_dm.reference(category="constructor", call_name="Simultaneous.from_string",)
    def from_string(klass, *args, **kwargs, ) -> _sources.SourceMixinProtocol:
        """
················································································

==Create `Simultaneous` model from string==

```
self = Simultaneous.from_string(
    string,
    /,
    context=None,
    description="",
)
```

Read and parse a text `string` with a model source code, and create a
`Simultaneous` model object. Otherwise, this function behaves the same way as
[`Simultaneous.from_file`](#simultaneousfrom_file).


### Input arguments ###


???+ input "string"

    Text string from which the `Simultaneous` model object will be created.

See [`Simultaneous.from_file`](#simultaneousfrom_file) for other input arguments.


### Returns ###

See [`Simultaneous.from_file`](simultaneousfrom_file) for return values.

················································································
        """
        return _sources.from_string(klass, *args, **kwargs, )

    def copy(self, /, ) -> Self:
        """
        Create a quasi-deep copy of this model
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
        """
        Implement self[i] for variants and self[name] for quantitie values
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
        """
        """
        if isinstance(reference, str):
            self._assign({reference: value}, )
        else:
            raise NotImplementedError("Assigning model variants not implemented yet", )

    def __repr__(self, /, ) -> str:
        """
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
        """
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
        """
        Change the log-status of some model quantities
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
        """==True for models declared as linear=="""
        return self._invariant._flags.is_linear

    @property
    @_dm.reference(category="property", )
    def is_flat(self, /, ) -> bool:
        """==True for models declared as flat=="""
        return self._invariant._flags.is_flat

    @property
    @_dm.reference(category="property", )
    def is_deterministic(self, /, ) -> bool:
        """==True for models declared as deterministic=="""
        return self._invariant._flags.is_deterministic

    @property
    @_dm.reference(category="property", )
    def num_transition_equations(self, /, ) -> int:
        """==Number of transition equations=="""
        return self._invariant.num_transition_equations

    @property
    @_dm.reference(category="property", )
    def num_measurement_equations(self, /, ) -> int:
        """==Number of measurement equations=="""
        return self._invariant.num_measurement_equations

    @property
    @_dm.reference(category="property", )
    def max_lag(self, /, ) -> int:
        """==Maximul lag in the model (negative or zero)=="""
        return self._invariant._min_shift

    @property
    @_dm.reference(category="property", )
    def max_lead(self, /, ) -> int:
        """==Maximul lead in the model (positive or zero)=="""
        return self._invariant._max_shift

    @property
    def solution_vectors(self, /, ) -> _descriptors.SolutionVectors:
        """
        """
        return self._invariant.dynamic_descriptor.solution_vectors

    @property
    def quantities(self, /, ) -> tuple[_quantities.Quantity]:
        """==Tuple of model quantities=="""
        return self._invariant.quantities

    @property
    def shock_qid_to_std_qid(self, /, ) -> dict[int, int]:
        """==Dictionary mapping shock quantity id to standard deviation quantity id=="""
        return self._invariant.shock_qid_to_std_qid

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        return _quantities.create_name_to_qid(self._invariant.quantities)

    def create_qid_to_name(self, /, ) -> dict[int, str]:
        return _quantities.create_qid_to_name(self._invariant.quantities)

    def create_qid_to_kind(self, /, ) -> dict[int, str]:
        return _quantities.create_qid_to_kind(self._invariant.quantities)

    def create_qid_to_description(self, /, ) -> dict[int, str]:
        """
        Create a dictionary mapping from quantity id to quantity descriptor
        """
        return _quantities.create_qid_to_description(self._invariant.quantities)

    def create_steady_array(
        self,
        /,
        variant: Variant | None = None,
        **kwargs,
    ) -> _np.ndarray:
        """
        """
        qid_to_logly = self.create_qid_to_logly()
        if variant is None:
            variant = self._variants[0]
        return variant.create_steady_array(qid_to_logly, **kwargs, )

    def create_zero_array(
        self,
        /,
        variant: Variant | None = None,
        **kwargs,
    ) -> _np.ndarray:
        """
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
        return {
            True: self.create_zero_array, False: self.create_steady_array,
        }[deviation](**kwargs)

    def _enforce_assignment_rules(self, variant, /, ) -> None:
        """
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
        """
        Create unsolved first-order system for each variant
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
        """
        Create unsolved first-order system for one variant
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
        """
        Calculate first-order solution for each variant within this model
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
        """
        Calculate first-order solution for one variant of this model
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
        """
        """
        match equation_switch:
            case "dynamic":
                return self._invariant._plain_dynamic_equator
            case "steady":
                return self._invariant._plain_steady_equator

    def reset_stds(self, /, ) -> None:
        """
        Initialize standard deviations of shocks to default values
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
        """
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
        """
        """
        return self._invariant._context

    def resolve_flags(self, /, **kwargs, ) -> _flags.Flags:
        return _flags.Flags.update_from_kwargs(self.get_flags(), **kwargs)

    def __getstate__(self, /, ):
        """
        """
        return {
            "_invariant": self._invariant,
            "_variants": self._variants,
        }

    def __setstate__(self, state, /, ):
        """
        """
        self._invariant = state["_invariant"]
        self._variants = state["_variants"]

    def _serialize_to_portable(self, /, ) -> dict[str, Any]:
        """
        """
        qid_to_name = self.create_qid_to_name()
        return {
            "source": self._invariant._serialize_to_portable(),
            "variants": [v._serialize_to_portable(qid_to_name, ) for v in self._variants],
        }

    #]


Model = Simultaneous

