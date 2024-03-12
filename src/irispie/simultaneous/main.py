"""
Simultaneous models
"""


#[
from __future__ import annotations

from typing import (Self, Any, TypeAlias, Literal, )
from types import EllipsisType
from collections.abc import (Iterable, Iterator, Callable, )
from numbers import (Number, )
import copy as _co
import numpy as _np
import itertools as _it
import functools as _ft

from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from .. import has_invariant as _has_invariant
from .. import has_variants as _has_variants
from .. import pages as _pages
from ..conveniences import iterators as _iterators
from ..parsers import common as _pc
from ..databoxes import main as _databoxes
from ..fords import solutions as _solutions
from ..fords import steadiers as _fs
from ..fords import descriptors as _descriptors
from ..fords import systems as _systems
from ..fords import kalmans as _kalmans

from . import invariants as _invariants
from . import variants as _variants
from . import _covariances as _covariances
from . import _flags as _flags
from . import _simulate as _simulate
from . import _steady as _steady
from . import _get as _get
from . import _assigns as _assigns
#]


__all__ = [
    "Simultaneous",
    "Model",
]


_SIMULATE_CAN_BE_EXOGENIZED = _quantities.QuantityKind.ENDOGENOUS_VARIABLE
_SIMULATE_CAN_BE_ENDOGENIZED = _quantities.QuantityKind.EXOGENOUS_VARIABLE | _quantities.QuantityKind.ANY_SHOCK


@_pages.reference(
    path=("structural_models", "simultaneous_models", "reference.md", ),
    categories={
        "constructor": "Creating new simultaneous models",
        "property": None,
    },
)
class Simultaneous(
    _assigns.AssignMixin,
    _has_invariant.HasInvariantMixin,
    _has_variants.HasVariantsMixin,
    _simulate.SimulateMixin,
    _steady.SteadyInlay,
    _covariances.CoverianceMixin,
    _kalmans.KalmanMixin,
    _get.GetMixin,
):
    """
················································································

`Simultaneous` model objects
==============================

················································································
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    def __init__(
        self,
        /,
    ) -> None:
        """
        """
        self._invariant = None
        self._variants = []

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
    @_pages.reference(category="constructor", call_name="Simultaneous.from_file", )
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
    @_pages.reference(category="constructor", call_name="Simultaneous.from_string",)
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
        Create a deep copy of this model
        """
        return _co.deepcopy(self)

    def __getitem__(
        self,
        request,
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
        reference,
        value: Any,
        /,
    ) -> None:
        """
        """
        if isinstance(reference, str):
            self.assign(**{reference: value}, )

    def __repr__(self, /, ) -> str:
        """
        """
        indented = " " * 4
        min_shift, max_shift = self.get_min_max_shifts()
        return "\n".join((
            f"",
            f"{self.__class__.__name__} model",
            f"Description: \"{self.get_description()}\"",
            f"|",
            f"| Num of variants: {self.num_variants}",
            f"| Num of equations [transition, measurement]: [{self.num_transition_equations}, {self.num_measurement_equations}]",
            f"| Time shifts [min, max]: [{min_shift:+g}, {max_shift:+g}]",
            f"|",
        ))

    def get_value(
        self,
        name: str,
        /,
    ) -> Any:
        """
        """
        names = (name, )
        quantities = self._invariant.quantities
        qids, invalid_names = _quantities.lookup_qids_by_name(quantities, names, )
        if invalid_names:
            raise _wrongdoings.IrisPieError(f"Invalid model name \"{invalid_names[0]}\"", )
        else:
            singleton_dict = self._get_values("levels", qids, )
            return singleton_dict[name]

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
    @_pages.reference(category="property", )
    def is_linear(self, /, ) -> bool:
        """==True for models declared as linear=="""
        return self._invariant._flags.is_linear

    @property
    @_pages.reference(category="property", )
    def is_flat(self, /, ) -> bool:
        """==True for models declared as flat=="""
        return self._invariant._flags.is_flat

    @property
    @_pages.reference(category="property", )
    def is_deterministic(self, /, ) -> bool:
        """==True for models declared as deterministic=="""
        return self._invariant._flags.is_deterministic

    @property
    @_pages.reference(category="property", )
    def num_transition_equations(self, /, ) -> int:
        """==Number of transition equations=="""
        return self._invariant.num_transition_equations

    @property
    @_pages.reference(category="property", )
    def num_measurement_equations(self, /, ) -> int:
        """==Number of measurement equations=="""
        return self._invariant.num_measurement_equations

    @property
    @_pages.reference(category="property", )
    def max_lag(self, /, ) -> int:
        """==Maximul lag in the model (negative or zero)=="""
        return self._invariant._min_shift

    @property
    @_pages.reference(category="property", )
    def max_lead(self, /, ) -> int:
        """==Maximul lead in the model (positive or zero)=="""
        return self._invariant._max_shift

    @property
    def _solution_vectors(self, /, ) -> _descriptors.SolutionVectors:
        """
        """
        return self._invariant.dynamic_descriptor.solution_vectors

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

    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        """
        Create a dictionary mapping from quantity id to quantity log-status
        """
        return _quantities.create_qid_to_logly(self._invariant.quantities)

    def create_steady_array(
        self,
        /,
        variant: _variants.Variant | None = None,
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
        variant: _variants.Variant | None = None,
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
        #
        name_to_qid = self.create_name_to_qid()
        shock_qids = _quantities.generate_qids_by_kind(
            self._invariant.quantities, _quantities.QuantityKind.ANY_SHOCK,
        )
        zero_shocks = { i: 0 for i in shock_qids }
        variant.update_values_from_dict(zero_shocks, )
        #
        # Remove changes from quantities that are not logly variables
        #
        nonloglies = _quantities.generate_qids_by_kind(
            self._invariant.quantities, ~_sources.LOGLY_VARIABLE,
        )
        assign_nonloglies = { qid: (..., _np.nan, ) for qid in nonloglies }
        variant.update_values_from_dict(assign_nonloglies, )

    def systemize(
        self,
        /,
        **kwargs,
    ) -> Iterable[_systems.System]:
        """
        Create unsolved first-order system for each variant
        """
        model_flags = self._invariant._flags.update_from_kwargs(**kwargs, )
        return tuple(
            self._systemize(variant, self._invariant.dynamic_descriptor, model_flags, )
            for variant in self._variants
        )

    def _systemize(
        self,
        variant: _variants.Variant,
        descriptor: _descriptors.Descriptor,
        model_flags: flags.Flags,
        /,
    ) -> _systems.System:
        """
        Create unsolved first-order system for one variant
        """
        min_shift, max_shift = self.get_min_max_shifts()
        num_columns = -min_shift + 1 + max_shift
        qid_to_logly = self.create_qid_to_logly()
        #
        if model_flags.is_linear:
            data_array = variant.create_zero_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=min_shift, )
            data_array_lagged = None
            steady_array = variant.create_steady_array(qid_to_logly, num_columns=1, ).reshape(-1)
        else:
            data_array = variant.create_steady_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=min_shift, )
            data_array_lagged = variant.create_steady_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=min_shift-1, )
            steady_array = data_array[:, -min_shift]
        #
        column_offset = -min_shift
        return _systems.System(
            descriptor, data_array, steady_array,
            model_flags, data_array_lagged,
            column_offset,
        )

    def solve(
        self,
        /,
        clip_small: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Calculate first-order solution for each variant within this model
        """
        model_flags = self._invariant._flags.update_from_kwargs(**kwargs, )
        for variant in self._variants:
            self._solve(variant, model_flags, clip_small=clip_small, )
        info = {}
        return info

    def _solve(
        self,
        variant: _variants.Variant,
        model_flags: flags.Flags,
        /,
        clip_small: bool,
    ) -> None:
        """
        Calculate first-order solution for one variant of this model
        """
        system = self._systemize(variant, self._invariant.dynamic_descriptor, model_flags, )
        variant.solution = _solutions.Solution(self._invariant.dynamic_descriptor, system, clip_small=clip_small, )

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

    def _assign_default_stds(self, /, ) -> None:
        """
        Initialize standard deviations of shocks to default values
        """
        std_names = _quantities.generate_quantity_names_by_kind(self._invariant.quantities, _quantities.QuantityKind.ANY_STD, )
        dict_to_assign = { k: self._invariant._default_std for k in std_names }
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
        self._invariant = _invariants.Invariant(source, **kwargs, )
        initial_variant = _variants.Variant(
            self._invariant.quantities,
            self._invariant._flags.is_flat,
        )
        self._variants = [ initial_variant ]
        self._assign_default_stds()
        self._enforce_assignment_rules(self._variants[0], )
        return self

    def get_context(self, /, ) -> dict[str, Any]:
        """
        """
        return self._invariant._context

    #
    # ===== Implement SlatableProtocol =====
    #

    def get_min_max_shifts(self, ) -> tuple[int, int]:
        """
        """
        return self._invariant._min_shift, self._invariant._max_shift

    def get_databox_names(self, /, ) -> tuple[str, ...]:
        """
        """
        qid_to_name = self.create_qid_to_name()
        return tuple(qid_to_name[qid] for qid in sorted(qid_to_name))

    def get_fallbacks(self, /, ) -> dict[str, Number]:
        """
        """
        shocks = _quantities.generate_quantity_names_by_kind(self._invariant.quantities, _quantities.QuantityKind.ANY_SHOCK)
        return { name: 0 for name in shocks }

    def get_overwrites(self, /, ) -> dict[str, Any]:
        """
        """
        return self.get_parameters_stds()

    def get_scalar_names(self, /, ) -> tuple[str, ...]:
        """
        """
        return _quantities.generate_quantity_names_by_kind(
            self._invariant.quantities,
            _quantities.QuantityKind.PARAMETER | _quantities.QuantityKind.ANY_STD,
        )

    #
    # ===== Implement PlannableSimulateProtocol =====
    #

    @property
    def simulate_can_be_exogenized(self, /, ) -> tuple[str, ...]:
        return tuple(_quantities.generate_quantity_names_by_kind(
            self._invariant.quantities, _SIMULATE_CAN_BE_EXOGENIZED,
        ))

    @property
    def simulate_can_be_endogenized(self, /, ) -> tuple[str, ...]:
        return tuple(_quantities.generate_quantity_names_by_kind(
            self._invariant.quantities, _SIMULATE_CAN_BE_ENDOGENIZED,
        ))

    simulate_can_be_when_data = ()

    #]


Model = Simultaneous

