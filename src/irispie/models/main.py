"""
"""


#[
from __future__ import annotations

from typing import (Self, Any, TypeAlias, Literal, )
from collections.abc import (Iterable, Callable, )
from numbers import (Number, )
import copy as _co
import numpy as _np
import itertools as _it
import functools as _ft

from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from .. import dates as _dates
from ..parsers import common as _pc
from ..databanks import main as _databanks
from ..fords import solutions as _solutions
from ..fords import steadiers as _fs
from ..fords import descriptors as _descriptors
from ..fords import systems as _systems
from .. import wrongdoings as _wrongdoings

from . import variants as _variants
from . import invariants as _invariants
from . import _covariances as _covariances
from . import _flags as _flags
from . import _simulate as _simulate
from . import _steady as _steady
from . import _get as _get
from . import _processor as _processor
#]


__all__ = [
    "Model"
]


class Model(
    _sources.SourceMixin,
    _simulate.SimulateMixin,
    _steady.SteadyMixin,
    _covariances.CoverianceMixin,
    _get.GetMixin,
    _processor.ProcessorMixin,
):
    """

    """
    #[
    __slots__ = (
        "_invariant", "_variants",
    )

    def __init__(
        self,
        /,
    ) -> None:
        """
        """
        self._invariant = None
        self._variants = None

    def assign(
        self,
        /,
        **kwargs,
    ) -> None:
        """
        """
        name_to_qid = self.create_name_to_qid()
        qid_to_name = self.create_qid_to_name()
        #
        qid_to_value = _rekey_dict(kwargs, name_to_qid, )
        for vid, variant in enumerate(self._variants, ):
            qid_to_value_variant = _extract_dict_variant(qid_to_value, vid, qid_to_name, )
            variant.update_values_from_dict(qid_to_value_variant, )
            self._enforce_auto_values(variant, )

    def copy(self) -> Self:
        """
        Create a deep copy of the Model object
        """
        return _co.deepcopy(self)

    def __getitem__(
        self,
        request,
        /,
    ):
        """
        """
        if isinstance(request, str):
            return self._getitem_value(request, )
        else:
            return self._getitem_variant(variant, )

    def _getitem_value(
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

    def _getitem_variant(
        self,
        variants,
        /,
    ) -> Self:
        new = self.from_self()
        index_variants = resolve_variant(self, variants, )
        new._variants = [ self._variants[i] for i in index_variants ]
        return new

    def alter_num_variants(
        self,
        new_num: int,
        /,
    ) -> Self:
        """
        Alter (expand, shrink) the number of alternative parameter variants in this model object
        """
        if new_num < self.num_variants:
            self._shrink_num_variants(new_num, )
        elif new_num > self.num_variants:
            self._expand_num_variants(new_num, )
        return self

    @property
    def is_singleton(self, /, ) -> bool:
        """
        True for Models with only one variant
        """
        return self.num_variants == 1

    def change_logly(
        self,
        new_logly: bool,
        some_names: Iterable[str] | None = None,
        /
    ) -> None:
        """
        Change the log-status of some Model quantities
        """
        some_names = set(some_names) if some_names else None
        qids = [
            qty.id
            for qty in self._invariant.quantities
            if qty.logly is not None and (some_names is None or qty.human in some_names)
        ]
        self._invariant.quantities = _quantities.change_logly(self._invariant.quantities, new_logly, qids)

    @property
    def num_variants(self, /, ) -> int:
        """
        Number of alternative variants within this Model
        """
        return len(self._variants)

    @property
    def is_linear(self, /, ) -> bool:
        """
        True for Models declared as linear
        """
        return self._invariant._flags.is_linear

    @property
    def is_flat(self, /, ) -> bool:
        """
        True for Models declared as flat
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
        variant: _variants.Variant|None = None,
        **kwargs,
    ) -> _np.ndarray:
        qid_to_logly = self.create_qid_to_logly()
        if variant is None:
            variant = self._variants[0]
        return variant.create_steady_array(qid_to_logly, **kwargs, )

    def create_zero_array(
        self,
        /,
        variant: _variants.Variant|None = None,
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

    def _enforce_auto_values(self, variant, /, ) -> None:
        """
        """
        #
        # Reset levels of shocks to zero, remove changes
        #
        assign_shocks = {
            qid: (0, _np.nan)
            for qid in _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.SHOCK)
        }
        variant.update_values_from_dict(assign_shocks)
        #
        # Remove changes from quantities that are not logly variables
        #
        assign_non_logly = {
            qid: (..., _np.nan)
            for qid in _quantities.generate_qids_by_kind(self._invariant.quantities, ~_sources.LOGLY_VARIABLE)
        }
        variant.update_values_from_dict(assign_non_logly)

    def _shrink_num_variants(self, new_num: int, /, ) -> None:
        """
        """
        if new_num<1:
            raise Exception('Number of variants must be one or more')
        self._variants = self._variants[0:new_num]

    def _expand_num_variants(self, new_num: int, /, ) -> None:
        """
        """
        for i in range(self.num_variants, new_num):
            self._variants.append(_co.deepcopy(self._variants[-1]))

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
        ac = descriptor.aldi_context
        num_columns = ac.shape_data[1]
        qid_to_logly = self.create_qid_to_logly()
        #
        if model_flags.is_linear:
            data_array = variant.create_zero_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=ac.min_shift, )
            data_array_lagged = None
            steady_array = variant.create_steady_array(qid_to_logly, num_columns=1, ).reshape(-1)
        else:
            data_array = variant.create_steady_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=ac.min_shift, )
            data_array_lagged = variant.create_steady_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=ac.min_shift-1, )
            steady_array = data_array[:, -ac.min_shift]
        #
        return _systems.System(descriptor, data_array, steady_array, model_flags, data_array_lagged, )

    def solve(
        self,
        /,
        **kwargs,
    ) -> None:
        """
        Calculate first-order solution for each Variant within this Model
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
        Calculate first-order solution for one Variant of this Model
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
                return self._invariant._plain_equator_for_dynamic_equations
            case "steady":
                return self._invariant._plain_equator_for_steady_equations

    def _assign_default_stds(
        self,
        default_std: Number | None = None,
        /,
    ) -> None:
        """
        """
        default_std = default_std if default_std is not None else _DEFAULT_STD_RESOLUTION(self.get_flags())
        dict_to_assign = {
            k: default_std
            for k in _quantities.generate_quantity_names_by_kind(self._invariant.quantities, _quantities.QuantityKind.STD, )
        }
        self.assign(**dict_to_assign, )

    @classmethod
    def from_source(
        cls,
        source: _sources.AlgebraicSource,
        /,
        default_std: int | None = None,
        context: dict | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        self = cls()
        self._invariant = _invariants.Invariant(
            source,
            context=context,
            **kwargs,
        )
        self._variants = [ _variants.Variant(self._invariant.quantities, ) ]
        for v in self._variants:
            self._enforce_auto_values(v, )
        self._assign_default_stds(default_std, )
        return self

    def from_self(self, ) -> Self:
        """
        Create a new Model object with pointers to invariant and variants of this Model object
        """
        new = type(self)()
        new._invariant = self._invariant
        new._variants = self._variants
        return new

    #
    # Implement SimulatableProtocol
    #

    def get_min_max_shifts(self) -> tuple[int, int]:
        """
        """
        return self._invariant._min_shift, self._invariant._max_shift

    def get_databank_names(self, plan, /, ) -> tuple[str, ...]:
        qid_to_name = self.create_qid_to_name()
        return tuple(qid_to_name[qid] for qid in range(len(qid_to_name)))
    #]


_DEFAULT_STD_LINEAR = 1
_DEFAULT_STD_NONLINEAR = 0.01
_DEFAULT_STD_RESOLUTION = lambda flags: {
    True: _DEFAULT_STD_LINEAR,
    False: _DEFAULT_STD_NONLINEAR,
}[flags.is_linear]


def _rekey_dict(dict_to_rekey: dict, old_key_to_new_key: dict, /, garbage_key=None, ) -> dict:
    #[
    new_dict = {
        old_key_to_new_key.get(key, garbage_key): value
        for key, value in dict_to_rekey.items()
    }
    if garbage_key in new_dict:
        del new_dict[garbage_key]
    return new_dict
    #]


def resolve_variant(self, variants, /, ) -> Iterable[int]:
    #[
    if isinstance(variants, Number):
        return [variants, ]
    elif variants is Ellipsis:
        return range(self.num_variants)
    elif isinstance(variants, slice):
        return range(*variants.indices(self.num_variants))
    else:
        return [v for v in variants]
    #]


def _extract_dict_variant(
    qid_to_value: dict[int, Any],
    vid: int,
    qid_to_name: dict[int, str],
    /,
) -> dict[int, tuple[Number|Ellipsis, Number|Ellipsis]]:
    """
    """
    def _extract_value_variant(
        value: list | tuple | Number,
        name: str,
        /,
    ) -> tuple[Number|Ellipsis, Number|Ellipsis]:
        """
        """
        if isinstance(value, Iterable, ) and not isinstance(value, str, ) and not isinstance(value, tuple, ):
            value = list(value, )[vid]
        if isinstance(value, Number, ) or value is Ellipsis:
            value = (value, Ellipsis, )
        if not isinstance(value, tuple, ) or len(value) != 2:
            raise _wrongdoings.IrisPieError(
                f"Invalid type of value assigned to this Model quantity: \"{name}\"",
            )
        return value
        #
    return {
        qid: _extract_value_variant(value, qid_to_name[qid], )
        for qid, value in qid_to_value.items()
    }
