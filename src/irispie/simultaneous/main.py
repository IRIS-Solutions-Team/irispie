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
from .. import iter_variants as _iter_variants
from ..conveniences import iterators as _iterators
from ..conveniences import files as _files
from ..parsers import common as _pc
from ..databoxes import main as _databoxes
from ..fords import solutions as _solutions
from ..fords import steadiers as _fs
from ..fords import descriptors as _descriptors
from ..fords import systems as _systems

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


class Simultaneous(
    _sources.SourceMixin,
    _assigns.AssignMixin,
    _iter_variants.IterVariantsMixin,
    _simulate.SimulateMixin,
    _steady.SteadyMixin,
    _covariances.CoverianceMixin,
    _get.GetMixin,
    _files.FromFileMixin,
):
    """

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
        self._variants = None

    def copy(self) -> Self:
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
    def is_linear(self, /, ) -> bool:
        """
        True for models declared as linear
        """
        return self._invariant._flags.is_linear

    @property
    def is_flat(self, /, ) -> bool:
        """
        True for models declared as flat
        """
        return self._invariant._flags.is_flat

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

    def _enforce_autovalues(self, variant, /, ) -> None:
        """
        """
        #
        # Reset levels of shocks to zero, remove changes
        #
        name_to_qid = self.create_name_to_qid()
        autovalues = self.get_autovalues()
        autovalues = { name_to_qid[k]: v for k, v in autovalues.items() }
        variant.update_values_from_dict(autovalues, )
        #
        # Remove changes from quantities that are not logly variables
        #
        nonloglies = _quantities.generate_qids_by_kind(self._invariant.quantities, ~_sources.LOGLY_VARIABLE, )
        assign_nonloglies = { qid: (..., _np.nan) for qid in nonloglies }
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
        **kwargs,
    ) -> None:
        """
        Calculate first-order solution for each variant within this model
        """
        model_flags = self._invariant._flags.update_from_kwargs(**kwargs, )
        for variant in self._variants:
            self._solve(variant, model_flags, )

    def _solve(
        self,
        variant: _variants.Variant,
        model_flags: flags.Flags,
        /,
    ) -> None:
        """
        Calculate first-order solution for one variant of this model
        """
        system = self._systemize(variant, self._invariant.dynamic_descriptor, model_flags, )
        variant.solution = _solutions.Solution(self._invariant.dynamic_descriptor, system, )

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
        std_names = _quantities.generate_quantity_names_by_kind(self._invariant.quantities, _quantities.QuantityKind.STD, )
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
        self._enforce_autovalues(self._variants[0], )
        self._assign_default_stds()
        return self

    #
    # Implement SlatableProtocol
    # =============================
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

    def get_autovalues(self, /, ) -> dict[str, Number]:
        """
        """
        shocks = _quantities.generate_quantity_names_by_kind(self._invariant.quantities, _quantities.QuantityKind.SHOCK)
        return { name: 0 for name in shocks }
    #]


Model = Simultaneous

