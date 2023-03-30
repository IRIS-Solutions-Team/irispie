"""
"""

#[
from __future__ import annotations
# from IPython import embed

from typing import (Self, NoReturn, TypeAlias, Literal, )
from numbers import Number
from collections.abc import Iterable, Callable

import enum
import copy
import scipy
import numpy as np_
import itertools
import functools
import operator

from . import (variants as va_, )
from .. import (sources as so_, )
from .. import (parsers as pa_, )

from .. import (equations as eq_, )
from .. import (quantities as qu_, )
from ..fords import (descriptors as de_, )
from ..fords import (systems as sy_, )
from .. import (exceptions as ex_, )
from ..dataman import (databanks as db_, dates as da_)
from .. import (evaluators as ev_, )
from . import (getters as ge_, )
from ..models import (simulations as si_, evaluators as me_, )
from ..fords import (solutions as sl_, steadiers as fs_, )
#]


__all__ = [
    "Model"
]


class _Default: ...
_default = _Default()


SteadySolverReturn: TypeAlias = tuple[
    np_.ndarray|None, Iterable[int]|None, 
    np_.ndarray|None, Iterable[int]|None,
]

EquationSwitch: TypeAlias = Literal["dynamic"] | Literal["steady"]


class ModelFlags(enum.IntFlag, ):
    """
    """
    #[
    LINEAR = enum.auto()
    FLAT = enum.auto()
    DEFAULT = 0

    @property
    def is_linear(self, /, ) -> bool:
        return ModelFlags.LINEAR in self

    @property
    def is_flat(self, /, ) -> bool:
        return ModelFlags.FLAT in self

    @classmethod
    def update_from_kwargs(cls, self, /, **kwargs) -> Self:
        linear = kwargs.get("linear") if kwargs.get("linear") is not None else self.is_linear
        flat = kwargs.get("flat") if kwargs.get("flat") is not None else self.is_flat
        return cls.from_kwargs(linear=linear, flat=flat)

    @classmethod
    def from_kwargs(cls: type, **kwargs, ) -> Self:
        self = cls.DEFAULT
        if kwargs.get("linear"):
            self |= cls.LINEAR
        if kwargs.get("flat"):
            self |= cls.FLAT
        return self
    #]


_DEFAULT_STD_LINEAR = 1
_DEFAULT_STD_NONLINEAR = 0.01


class Model(si_.SimulationMixin, me_.SteadyEvaluatorMixin, ge_.GetterMixin):
    """
    """
    #[
    def assign(
        self: Self,
        /,
        **kwargs, 
    ) -> Self:
        """
        """
        try:
            qid_to_value = _rekey_dict(kwargs, qu_.create_name_to_qid(self._quantities))
        except KeyError as _KeyError:
            raise ex_.UnknownName(_KeyError.args[0])
        for v in self._variants:
            v.update_values_from_dict(qid_to_value)
        #
        self._enforce_auto_values()
        return self

    def assign_from_databank(
        self, 
        databank: db_.Databank,
        /,
    ) -> Self:
        """
        """
        return self.assign(**databank.__dict__)

    @property
    def copy(self) -> Self:
        return copy.deepcopy(self)

    def __getitem__(self, variants):
        variants = resolve_variant(self, variants)
        self_copy = copy.copy(self)
        self_copy._variants = [ self._variants[i] for i in variants ]
        return self_copy

    def change_num_variants(self, new_num: int, /, ) -> Self:
        """
        Change number of alternative parameter variants in model object
        """
        if new_num<self.num_variants:
            self._shrink_num_variants(new_num, )
        elif new_num>self.num_variants:
            self._expand_num_variants(new_num, )
        return self

    def change_logly(
        self,
        new_logly: bool,
        names: Iterable[str] | None = None,
        /
    ) -> NoReturn:
        names = set(names) if names else None
        qids = [ 
            qty.id 
            for qty in self._quantities 
            if qty.logly is not None and (names is None or qty.human in names)
        ]
        self._quantities = qu_.change_logly(self._quantities, new_logly, qids)

    @property
    def num_variants(self, /, ) -> int:
        return len(self._variants)

    @property
    def is_linear(self, /, ) -> bool:
        return self._flags.is_linear

    @property
    def is_flat(self, /, ) -> bool:
        return self._flags.is_flat

    def create_steady_evaluator(self, /, ) -> ev_.SteadyEvaluator:
        """
        """
        equations = eq_.generate_equations_of_kind(self._steady_equations, eq_.EquationKind.STEADY_EVALUATOR)
        quantities = qu_.generate_quantities_of_kind(self._quantities, qu_.QuantityKind.STEADY_EVALUATOR)
        return self._create_steady_evaluator(self._variants[0], equations, quantities)

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        return qu_.create_name_to_qid(self._quantities)

    def create_qid_to_name(self, /, ) -> dict[int, str]:
        return qu_.create_qid_to_name(self._quantities)

    def create_qid_to_kind(self, /, ) -> dict[int, str]:
        return qu_.create_qid_to_kind(self._quantities)

    def create_qid_to_descript(self, /, ) -> dict[int, str]:
        return qu_.create_qid_to_descript(self._quantities)

    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        return qu_.create_qid_to_logly(self._quantities)

    def get_ordered_names(self, /, ) -> list[str]:
        qid_to_name = self.create_qid_to_name()
        return [ qid_to_name[qid] for qid in range(len(qid_to_name)) ]

    def create_steady_array(
        self,
        /,
        variant: va_.Variant|None = None,
        **kwargs,
    ) -> np_.ndarray:
        qid_to_logly = self.create_qid_to_logly()
        if variant is None:
            variant = self._variants[0]
        return variant.create_steady_array(qid_to_logly, **kwargs, )

    def create_zero_array(
        self,
        /,
        variant: va_.Variant|None = None,
        **kwargs,
    ) -> np_.ndarray:
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
    ) -> np_.ndarray:
        return {
            True: self.create_zero_array, False: self.create_steady_array,
        }[deviation](**kwargs)

    def _enforce_auto_values(self: Self, /, ) -> NoReturn:
        """
        """
        #
        # Reset levels of shocks to zero, remove changes
        #
        assign_shocks = { 
            qid: (0, np_.nan) 
            for qid in qu_.generate_qids_by_kind(self._quantities, qu_.QuantityKind.SHOCK)
        }
        self._variants[0].update_values_from_dict(assign_shocks)
        #
        # Remove changes from quantities that are not logly variables
        #
        assign_non_logly = { 
            qid: (..., np_.nan) 
            for qid in  qu_.generate_qids_by_kind(self._quantities, ~qu_.QuantityKind.LOGLY_VARIABLE)
        }
        self._variants[0].update_values_from_dict(assign_non_logly)

    def _shrink_num_variants(self, new_num: int, /, ) -> NoReturn:
        if new_num<1:
            raise Exception('Number of variants must be one or more')
        self._variants = self._variants[0:new_num]

    def _expand_num_variants(self, new_num: int, /, ) -> NoReturn:
        for i in range(self.num_variants, new_num):
            self._variants.append(copy.deepcopy(self._variants[-1]))


    # def systemize(
        # self,
        # /,
        # _variant: int | ... = ...,
        # linear: bool | None = None,
        # flat: bool | None = None,
    # ) -> Self:
        # """
        # """
        # model_flags = ModelFlags.update_from_kwargs(self._flags, linear=linear, flat=flat)
        # for v in resolve_variant(self, _variant):
            # self._systemize(v, model_flags)
        # return self

    def systemize(self, /, ) -> Iterable[sy_.System]:
        """
        Unsolved first-order systems one for each variant
        """
        return [ 
            self._systemize(variant, self._dynamic_descriptor) 
            for variant in self._variants
        ]

    def _systemize(
        self,
        variant: va_.Variant,
        descriptor: de_.Descriptor,
        /,
    ) -> sy_.System:
        """
        Unsolved first-order system for one variant
        """
        num_columns = descriptor.aldi_context.shape_data[1]
        logly_context = self.create_qid_to_logly()
        value_context = self.create_zero_array(variant, num_columns=num_columns)
        return sy_.System.for_descriptor(descriptor, logly_context, value_context)

    def solve(
        self,
        /,
    ) -> NoReturn:
        for variant in self._variants:
            self._solve(variant)

    def _solve(
        self,
        variant: va_.Variant,
        /,
    ) -> NoReturn:
        system = self._systemize(variant, self._dynamic_descriptor)
        variant.solution = sl_.Solution.for_model(self._dynamic_descriptor, system, )

    def steady(
        self,
        /,
        **kwargs, 
    ) -> dict:
        """
        """
        solver = self._choose_steady_solver(**kwargs)
        for v in self._variants:
            levels, qids_levels, changes, qids_changes = solver(v)
            v.update_levels_from_array(levels, qids_levels)
            v.update_changes_from_array(changes, qids_changes)

    def check_steady(
        self,
        /,
        equation_switch: EquationSwitch = "dynamic",
        tolerance: float = 1e-12,
        details: bool = False,
    ) -> tuple[bool, Iterable[bool], Iterable[Number], Iterable[np_.ndarray]]:
        """
        Verify steady state against dynamic or steady equations
        """
        # FIXME: Evaluate at two different times
        qid_to_logly = self.create_qid_to_logly()
        evaluator = {
            "dynamic": self._plain_evaluator_for_dynamic_equations,
            "steady": self._plain_evaluator_for_steady_equations,
        }[equation_switch]
        steady_arrays = (
            v.create_steady_array(
                qid_to_logly,
                num_columns=evaluator.min_num_columns + 1,
                shift_in_first_column=evaluator.min_shift,
            )
            for v in self._variants
        )
        t_zero = -evaluator.min_shift
        dis = [ 
            np_.hstack((evaluator.eval(x, t_zero, x), evaluator.eval(x, t_zero+1, x)))
            for x in steady_arrays
        ]
        #
        max_abs_dis = [ np_.max(np_.abs(d)) for d in dis ]
        status = [ d < tolerance for d in max_abs_dis ]
        all_status = all(status)
        if not all_status:
            raise Exception("Invalid steady state")
        return (all_status, status, max_abs_dis, dis) if details else all_status

    def _apply_delog_on_vector(
        self,
        vector: np_.ndarray,
        qids: Iterable[int],
        /,
    ) -> np_.ndarray:
        """
        """
        qid_to_logly = self.create_qid_to_logly()
        logly_index = [ qid_to_logly[qid] for qid in qids ]
        if any(logly_index):
            vector[logly_index] = np_.exp(vector[logly_index])
        return vector

    def _steady_linear(
        self, 
        variant: Variant,
        /,
        algorithm: Callable,
    ) -> SteadySolverReturn:
        """
        """
        #
        # Calculate first-order system for steady equations for this variant
        sys = self._systemize(variant, self._steady_descriptor)
        #
        # Calculate steady state for this variant
        Xi, Y, dXi, dY = algorithm(sys)
        levels = np_.hstack(( Xi.flat, Y.flat ))
        changes = np_.hstack(( dXi.flat, dY.flat ))
        #
        # Extract only tokens with zero shift
        tokens = list(itertools.chain(
            self._steady_descriptor.system_vectors.transition_variables,
            self._steady_descriptor.system_vectors.measurement_variables,
        ))
        #
        # [True, False, True, ... ] True for tokens with zero shift
        zero_shift_index = [ not t.shift for t in tokens ]
        #
        # List of qids with zero shifts only
        qids = [ t.qid for t in itertools.compress(tokens, zero_shift_index) ]
        #
        # Extract steady levels for quantities with zero shift
        levels = levels[zero_shift_index]
        levels = self._apply_delog_on_vector(levels, qids)
        #
        # Extract steady changes for quantities with zero shift
        changes = self._apply_delog_on_vector(changes, qids)
        changes = changes[zero_shift_index]
        #
        return levels, qids, changes, qids

    _steady_linear_flat = functools.partialmethod(_steady_linear, algorithm=fs_.solve_steady_linear_flat)
    _steady_linear_nonflat = functools.partialmethod(_steady_linear, algorithm=fs_.solve_steady_linear_nonflat)

    def _steady_nonlinear_flat(
        self,
        variant: Variant,
        /,
    ) -> SteadySolverReturn:
        """
        """
        return None, None, None, None

    def _steady_nonlinear_nonflat(
        self,
        variant: Variant,
        /,
    ) -> SteadySolverReturn:
        """
        """
        return None, None, None, None

    def _choose_steady_solver(
        self,
        **kwargs,
    ) -> Callable:
        """
        Choose steady solver depending on linear and flat flags
        """
        STEADY_SOLVER = {
            ModelFlags.DEFAULT: self._steady_nonlinear_nonflat,
            ModelFlags.FLAT: self._steady_nonlinear_flat,
            ModelFlags.LINEAR: self._steady_linear_nonflat,
            ModelFlags.LINEAR | ModelFlags.FLAT: self._steady_linear_flat,
        }
        model_flags = ModelFlags.update_from_kwargs(self._flags, **kwargs)
        return STEADY_SOLVER[model_flags]


    def _assign_default_stds(self, default_std, /, ):
        if default_std is None:
            default_std = _DEFAULT_STD_LINEAR if ModelFlags.LINEAR in self._flags else _DEFAULT_STD_NONLINEAR
        self.assign(**{ k: default_std for k in qu_.generate_quantity_names_by_kind(self._quantities, qu_.QuantityKind.STD) })


    def _populate_min_max_shifts(self) -> NoReturn:
        self._min_shift = eq_.get_min_shift_from_equations(
            self._dynamic_equations + self._steady_equations
        )
        self._max_shift = eq_.get_max_shift_from_equations(
            self._dynamic_equations + self._steady_equations
        )

    def _get_min_max_shifts(self) -> tuple[int, int]:
        return self._min_shift, self._max_shift

    def get_extended_range_from_base_range(
        self,
        base_range: Iterable[Dater],
    ) -> Iterable[Dater]:
        base_range = [ t for t in base_range ]
        num_base_periods = len(base_range)
        start_date = base_range[0] + self._min_shift
        end_date = base_range[-1] + self._max_shift
        base_columns = [ c for c in range(-self._min_shift, -self._min_shift+num_base_periods) ]
        return [ t for t in da_.Ranger(start_date, end_date) ], base_columns

    @classmethod
    def from_source(
        cls: type,
        model_source: so_.ModelSource,
        /,
        default_std: int|None = None,
        **kwargs, 
    ) -> Self:
        """
        """
        self = cls()
        self._flags = ModelFlags.from_kwargs(**kwargs, )

        self._quantities = model_source.quantities[:]
        self._dynamic_equations = model_source.dynamic_equations[:]
        self._steady_equations = model_source.steady_equations[:]

        name_to_qid = qu_.create_name_to_qid(self._quantities)
        eq_.finalize_dynamic_equations(self._dynamic_equations, name_to_qid)
        eq_.finalize_steady_equations(self._steady_equations, name_to_qid)

        self._variants = [ va_.Variant(self._quantities) ]
        self._enforce_auto_values()
        self._dynamic_descriptor = de_.Descriptor(self._dynamic_equations, self._quantities)
        self._steady_descriptor = de_.Descriptor(self._steady_equations, self._quantities)

        self._plain_evaluator_for_dynamic_equations = ev_.PlainEvaluator(
            eq_.generate_equations_of_kind(self._dynamic_equations, eq_.EquationKind.STEADY_EVALUATOR),
        )

        self._plain_evaluator_for_steady_equations = ev_.PlainEvaluator(
            eq_.generate_equations_of_kind(self._steady_equations, eq_.EquationKind.STEADY_EVALUATOR),
        )

        self._populate_min_max_shifts()
        self._assign_default_stds(default_std)

        return self


    @classmethod
    def from_string(
        cls,
        source_string: str,
        /,
        context: dict | None = None,
        save_preparsed: str = "",
        **kwargs,
    ) -> Self:
        """
        """
        model_source, info = so_.ModelSource.from_string(
            source_string, context=context, save_preparsed=save_preparsed,
        )
        return Model.from_source(model_source, **kwargs, )


    @classmethod
    def from_file(
        cls,
        source_files: str|Iterable[str],
        /,
        **kwargs,
    ) -> Self:
        """
        """
        source_string = pa_.common.combine_source_files(source_files)
        return Model.from_string(source_string, **kwargs, )
    #]


def _rekey_dict(dict_to_rekey: dict, old_key_to_new_key: dict, /, ) -> dict:
    return { 
        old_key_to_new_key[key]: value 
        for key, value in dict_to_rekey.items()
    }


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


