"""
"""


#[
from __future__ import annotations

from typing import (Self, TypeAlias, Literal, )
from collections.abc import (Iterable, Callable, )
from numbers import (Number, )
import copy as _co
import numpy as _np
import itertools as _it
import functools as _ft

from .. import equations as _equations
from .. import quantities as _quantities
from .. import wrongdoings as _wd
from .. import sources as _sources
from .. import dates as _dates
from ..parsers import common as _pc
from ..databanks import main as _databanks
from ..fords import solutions as _sl
from ..fords import steadiers as _fs
from ..fords import descriptors as _de
from ..fords import systems as _sy

from . import variants as _variants
from . import invariants as _invariants
from . import _flags
from . import _simulate
from . import _steady
from . import _get
#]


#[
__all__ = [
    "Model"
]


_SteadySolverReturn: TypeAlias = tuple[
    _np.ndarray|None, Iterable[int]|None,
    _np.ndarray|None, Iterable[int]|None,
]


_EquationSwitch: TypeAlias = Literal["dynamic"] | Literal["steady"]
#]


class Model(
    _sources.SourceMixin,
    _simulate.SimulateMixin,
    _steady.SteadyMixin,
    _get.GetMixin,
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
    ) -> Self:
        """
        """
        garbage_key = None
        qid_to_value = _rekey_dict(kwargs, _quantities.create_name_to_qid(self._invariant._quantities))
        for v in self._variants:
            v.update_values_from_dict(qid_to_value)
        #
        self._enforce_auto_values()
        return self

    def assign_from_databank(
        self,
        databank: _databanks.Databank,
        /,
    ) -> Self:
        """
        """
        return self.assign(**databank.__dict__)

    def copy(self) -> Self:
        """
        Create a deep copy of the Model object
        """
        return _co.deepcopy(self)

    def __getitem__(self, variants):
        new = self.from_self()
        index_variants = resolve_variant(self, variants)
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
            for qty in self._invariant._quantities
            if qty.logly is not None and (some_names is None or qty.human in some_names)
        ]
        self._invariant._quantities = _quantities.change_logly(self._invariant._quantities, new_logly, qids)

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
        return _quantities.create_name_to_qid(self._invariant._quantities)

    def create_qid_to_name(self, /, ) -> dict[int, str]:
        return _quantities.create_qid_to_name(self._invariant._quantities)

    def create_qid_to_kind(self, /, ) -> dict[int, str]:
        return _quantities.create_qid_to_kind(self._invariant._quantities)

    def create_qid_to_descriptor(self, /, ) -> dict[int, str]:
        """
        Create a dictionary mapping from quantity id to quantity descriptor
        """
        return _quantities.create_qid_to_descriptor(self._invariant._quantities)

    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        """
        Create a dictionary mapping from quantity id to quantity log-status
        """
        return _quantities.create_qid_to_logly(self._invariant._quantities)

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

    def _enforce_auto_values(self, /, ) -> None:
        """
        """
        #
        # Reset levels of shocks to zero, remove changes
        #
        assign_shocks = {
            qid: (0, _np.nan)
            for qid in _quantities.generate_qids_by_kind(self._invariant._quantities, _quantities.QuantityKind.SHOCK)
        }
        self._variants[0].update_values_from_dict(assign_shocks)
        #
        # Remove changes from quantities that are not logly variables
        #
        assign_non_logly = {
            qid: (..., _np.nan)
            for qid in  _quantities.generate_qids_by_kind(self._invariant._quantities, ~sources.LOGLY_VARIABLE)
        }
        self._variants[0].update_values_from_dict(assign_non_logly)

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
    ) -> Iterable[_sy.System]:
        """
        Create unsolved first-order system for each variant
        """
        model_flags = self._invariant._flags.update_from_kwargs(**kwargs, )
        return [
            self._systemize(variant, self._invariant._dynamic_descriptor, model_flags, )
            for variant in self._variants
        ]

    def _systemize(
        self,
        variant: _variants.Variant,
        descriptor: _de.Descriptor,
        model_flags: flags.Flags,
        /,
    ) -> _sy.System:
        """
        Create unsolved first-order system for one variant
        """
        ac = descriptor.aldi_context
        num_columns = ac.shape_data[1]
        qid_to_logly = self.create_qid_to_logly()
        #
        if model_flags.is_linear:
            data_array = variant.create_zero_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=ac.min_shift)
            steady_array = variant.create_steady_array(qid_to_logly, num_columns=1, ).reshape(-1)
        else:
            data_array = variant.create_steady_array(qid_to_logly, num_columns=num_columns, )
            steady_array = data_array[:, -ac.min_shift]
        #
        return _sy.System(descriptor, data_array, steady_array, )

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
        system = self._systemize(variant, self._invariant._dynamic_descriptor, model_flags, )
        variant.solution = _sl.Solution(self._invariant._dynamic_descriptor, system, )

    def _choose_plain_equator(
        self,
        equation_switch: _EquationSwitch,
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
        self.assign(**{ k: default_std for k in _quantities.generate_quantity_names_by_kind(self._invariant._quantities, _quantities.QuantityKind.STD) })

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
        #
        self._invariant = _invariants.Invariant(
            source,
            context=context,
            **kwargs,
        )
        #
        self._variants = [ _variants.Variant(self._invariant._quantities) ]
        #
        self._enforce_auto_values()
        self._assign_default_stds(default_std)
        #
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

    def get_databank_names(self, /, ) -> list[str]:
        qid_to_name = self.create_qid_to_name()
        return [ qid_to_name[qid] for qid in range(len(qid_to_name)) ]
    #]


_DEFAULT_STD_LINEAR = 1
_DEFAULT_STD_NONLINEAR = 0.01
_DEFAULT_STD_RESOLUTION = lambda flags: {
    True: _DEFAULT_STD_LINEAR,
    False: _DEFAULT_STD_NONLINEAR,
}[flags.is_linear]


def _rekey_dict(dict_to_rekey: dict, old_key_to_new_key: dict, /, garbage_key=None) -> dict:
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


