"""
"""


#[
from __future__ import annotations

from typing import (Self, NoReturn, TypeAlias, Literal, )
from collections.abc import (Iterable, Callable, )
from numbers import (Number, )
import copy as co_
import numpy as np_
import itertools as it_
import functools as ft_

from .. import (equations as eq_, quantities as qu_, exceptions as ex_, sources as so_, evaluators as ev_, wrongdoings as wd_, )
from ..parsers import (common as pc_, )
from ..dataman import (databanks as db_, dates as da_)
from ..models import (simulations as si_, evaluators as me_, getters as ge_, variants as va_, invariants as in_, flags as mg_, )
from ..fords import (solutions as sl_, steadiers as fs_, descriptors as de_, systems as sy_, )
#]


#[
__all__ = [
    "Model"
]


_SteadySolverReturn: TypeAlias = tuple[
    np_.ndarray|None, Iterable[int]|None, 
    np_.ndarray|None, Iterable[int]|None,
]


_EquationSwitch: TypeAlias = Literal["dynamic"] | Literal["steady"]
#]


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Front end
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


class Model(
    si_.SimulationMixin,
    me_.SteadyEvaluatorMixin,
    ge_.GetterMixin,
):
    """
     
    """
    #[
    __slots__ = ["_invariant", "_variants"]

    def assign(
        self: Self,
        /,
        **kwargs, 
    ) -> Self:
        """
        """
        garbage_key = None
        qid_to_value = _rekey_dict(kwargs, qu_.create_name_to_qid(self._invariant._quantities))
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

    def copy(self) -> Self:
        """
        Create a deep copy of the Model object
        """
        return co_.deepcopy(self)

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
    ) -> NoReturn:
        """
        Change the log-status of some Model quantities
        """
        some_names = set(some_names) if some_names else None
        qids = [ 
            qty.id 
            for qty in self._invariant._quantities 
            if qty.logly is not None and (some_names is None or qty.human in some_names)
        ]
        self._invariant._quantities = qu_.change_logly(self._invariant._quantities, new_logly, qids)

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

    def create_steady_evaluator(self, /, ) -> ev_.SteadyEvaluator:
        """
        Create a steady-state Evaluator object for this Model
        """
        equations = eq_.generate_equations_of_kind(self._invariant._steady_equations, eq_.EquationKind.STEADY_EVALUATOR)
        quantities = qu_.generate_quantities_of_kind(self._invariant._quantities, qu_.QuantityKind.STEADY_EVALUATOR)
        return self._create_steady_evaluator(self._variants[0], equations, quantities)

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        return qu_.create_name_to_qid(self._invariant._quantities)

    def create_qid_to_name(self, /, ) -> dict[int, str]:
        return qu_.create_qid_to_name(self._invariant._quantities)

    def create_qid_to_kind(self, /, ) -> dict[int, str]:
        return qu_.create_qid_to_kind(self._invariant._quantities)

    def create_qid_to_descript(self, /, ) -> dict[int, str]:
        return qu_.create_qid_to_descript(self._invariant._quantities)

    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        return qu_.create_qid_to_logly(self._invariant._quantities)

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
            for qid in qu_.generate_qids_by_kind(self._invariant._quantities, qu_.QuantityKind.SHOCK)
        }
        self._variants[0].update_values_from_dict(assign_shocks)
        #
        # Remove changes from quantities that are not logly variables
        #
        assign_non_logly = { 
            qid: (..., np_.nan) 
            for qid in  qu_.generate_qids_by_kind(self._invariant._quantities, ~qu_.QuantityKind.LOGLY_VARIABLE)
        }
        self._variants[0].update_values_from_dict(assign_non_logly)

    def _shrink_num_variants(self, new_num: int, /, ) -> NoReturn:
        """
        """
        if new_num<1:
            raise Exception('Number of variants must be one or more')
        self._variants = self._variants[0:new_num]

    def _expand_num_variants(self, new_num: int, /, ) -> NoReturn:
        """
        """
        for i in range(self.num_variants, new_num):
            self._variants.append(co_.deepcopy(self._variants[-1]))

    def systemize(
        self,
        /,
        **kwargs,
    ) -> Iterable[sy_.System]:
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
        variant: va_.Variant,
        descriptor: de_.Descriptor,
        model_flags: mg_.ModelFlags,
        /,
    ) -> sy_.System:
        """
        Create unsolved first-order system for one variant
        """
        ac = descriptor.aldi_context
        num_columns = ac.shape_data[1]
        qid_to_logly = self.create_qid_to_logly()
        if model_flags.is_linear:
            value_context = variant.create_zero_array(qid_to_logly, num_columns=num_columns, shift_in_first_column=ac.min_shift)
            L = variant.create_steady_array(qid_to_logly, num_columns=1, ).reshape(-1)
        else:
            value_context = variant.create_steady_array(qid_to_logly, num_columns=num_columns, )
            L = value_context[:, -ac.min_shift]
        return sy_.System.from_descriptor(descriptor, qid_to_logly, value_context, L, )

    def solve(
        self,
        /,
        **kwargs,
    ) -> NoReturn:
        """
        Calculate first-order solution for each Variant within this Model
        """
        model_flags = self._invariant._flags.update_from_kwargs(**kwargs, )
        for variant in self._variants:
            self._solve(variant, model_flags, )

    def _solve(
        self,
        variant: va_.Variant,
        model_flags: mg_.ModelFlags,
        /,
    ) -> NoReturn:
        """
        Calculate first-order solution for one Variant of this Model
        """
        system = self._systemize(variant, self._invariant._dynamic_descriptor, model_flags, )
        variant.solution = sl_.Solution.for_model(self._invariant._dynamic_descriptor, system, model_flags, )

    def steady(
        self,
        /,
        **kwargs, 
    ) -> dict:
        """
        Calculate steady state for each Variant within this Model
        """
        model_flags = mg_.ModelFlags.update_from_kwargs(self._invariant._flags, **kwargs)
        solver = self._choose_steady_solver(model_flags)
        for v in self._variants:
            levels, qids_levels, changes, qids_changes = solver(v, model_flags, )
            v.update_levels_from_array(levels, qids_levels, )
            v.update_changes_from_array(changes, qids_changes, )

    def check_steady(
        self,
        /,
        equation_switch: _EquationSwitch = "dynamic",
        details: bool = False,
        when_fails: str = "error",
        tolerance: float = 1e-12,
    ) -> tuple[bool, Iterable[bool], Iterable[Number], Iterable[np_.ndarray]] | bool:
        """
        Verify currently assigned steady state in dynamic or steady equations for each Variant within this Model
        """
        qid_to_logly = self.create_qid_to_logly()
        evaluator = self._choose_plain_evaluator(equation_switch)
        steady_arrays = (
            v.create_steady_array(
                qid_to_logly,
                num_columns=evaluator.min_num_columns + 1,
                shift_in_first_column=evaluator.min_shift,
            ) for v in self._variants
        )
        # REFACTOR
        t_zero = -evaluator.min_shift
        dis = [ 
            np_.hstack((
                evaluator.eval(x, t_zero, x[:, t_zero]),
                evaluator.eval(x, t_zero+1, x[:, t_zero+1]),
            ))
            for x in steady_arrays
        ]
        # REFACTOR
        max_abs_dis = [ np_.max(np_.abs(d)) for d in dis ]
        status = [ d < tolerance for d in max_abs_dis ]
        all_status = all(status)
        if not all_status:
            message = "Invalid steady state"
            wd_.throw(when_fails, message)
        return (all_status, status, max_abs_dis, dis) if details else all_status

    def _choose_plain_evaluator(
        self,
        equation_switch: _EquationSwitch,
        /,
    ) -> Callable | None:
        """
        """
        match equation_switch:
            case "dynamic":
                return self._invariant._plain_evaluator_for_dynamic_equations
            case "steady":
                return self._invariant._plain_evaluator_for_steady_equations

    def _steady_linear(
        self, 
        variant: Variant,
        model_flags: mg_.ModelFlags,
        /,
        algorithm: Callable,
    ) -> _SteadySolverReturn:
        """
        """
        #
        # Calculate first-order system for steady equations for this variant
        sys = self._systemize(variant, self._invariant._steady_descriptor, model_flags, )
        #
        # Calculate steady state for this variant
        Xi, Y, dXi, dY = algorithm(sys)
        levels = np_.hstack(( Xi.flat, Y.flat ))
        changes = np_.hstack(( dXi.flat, dY.flat ))
        #
        # Extract only tokens with zero shift
        tokens = list(it_.chain(
            self._invariant._steady_descriptor.system_vectors.transition_variables,
            self._invariant._steady_descriptor.system_vectors.measurement_variables,
        ))
        #
        # [True, False, True, ... ] True for tokens with zero shift
        zero_shift_index = [ not t.shift for t in tokens ]
        #
        # Extract steady levels for quantities with zero shift
        levels = levels[zero_shift_index]
        changes = changes[zero_shift_index]
        qids = [ t.qid for t in it_.compress(tokens, zero_shift_index) ]
        #
        # Delogarithmize when needed
        qid_to_logly = self.create_qid_to_logly()
        levels = _apply_delog_on_vector(levels, qids, qid_to_logly)
        changes = _apply_delog_on_vector(changes, qids, qid_to_logly)
        #
        return levels, qids, changes, qids

    _steady_linear_flat = ft_.partialmethod(_steady_linear, algorithm=fs_.solve_steady_linear_flat)
    _steady_linear_nonflat = ft_.partialmethod(_steady_linear, algorithm=fs_.solve_steady_linear_nonflat)

    def _steady_nonlinear_flat(
        self,
        variant: Variant,
        /,
    ) -> _SteadySolverReturn:
        """
        """
        return None, None, None, None

    def _steady_nonlinear_nonflat(
        self,
        variant: Variant,
        /,
    ) -> _SteadySolverReturn:
        """
        """
        return None, None, None, None

    def _choose_steady_solver(
        self,
        model_flags: mg_.ModelFlags,
        /,
    ) -> Callable:
        """
        Choose steady solver depending on linear and flat flags
        """
        match (model_flags.is_linear, model_flags.is_flat):
            case (False, False):
                return self._steady_nonlinear_nonflat
            case (False, True):
                return self._steady_nonlinear_flat
            case (True, False):
                return self._steady_linear_nonflat
            case (True, True):
                return self._steady_linear_flat

    def _assign_default_stds(self, default_std, /, ):
        """
        """
        if default_std is None:
            default_std = _DEFAULT_STD_LINEAR if mg_.ModelFlags.LINEAR in self._invariant._flags else _DEFAULT_STD_NONLINEAR
        self.assign(**{ k: default_std for k in qu_.generate_quantity_names_by_kind(self._invariant._quantities, qu_.QuantityKind.STD) })

    def _get_min_max_shifts(self) -> tuple[int, int]:
        """
        """
        return self._invariant._min_shift, self._invariant._max_shift

    def get_extended_range_from_base_range(
        self,
        base_range: Iterable[Dater],
    ) -> Iterable[Dater]:
        """
        """
        base_range = [ t for t in base_range ]
        num_base_periods = len(base_range)
        start_date = base_range[0] + self._invariant._min_shift
        end_date = base_range[-1] + self._invariant._max_shift
        base_columns = [ c for c in range(-self._invariant._min_shift, -self._invariant._min_shift+num_base_periods) ]
        return [ t for t in da_.Ranger(start_date, end_date) ], base_columns

    @classmethod
    def from_source(
        cls,
        model_source: so_.ModelSource,
        /,
        default_std: int | None = None,
        context: dict | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        self = cls()
        #
        self._invariant = in_.Invariant(
            model_source,
            context=context,
            **kwargs,
        )
        #
        self._variants = [ va_.Variant(self._invariant._quantities) ]
        #
        self._enforce_auto_values()
        self._assign_default_stds(default_std)
        #
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
        return Model.from_source(model_source, context=context, **kwargs, )

    @classmethod
    def from_file(
        cls,
        source_files: str | Iterable[str],
        /,
        **kwargs,
    ) -> Self:
        """
        Create a new Model object from model source files
        """
        source_string = pc_.combine_source_files(source_files)
        return Model.from_string(source_string, **kwargs, )

    def from_self(self, ) -> Self:
        """
        Create a new Model object with pointers to invariant and variants of this Model object
        """
        new = type(self)()
        new._invariant = self._invariant
        new._variants = self._variants
        return new
    #]


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Back end
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


_DEFAULT_STD_LINEAR = 1
_DEFAULT_STD_NONLINEAR = 0.01


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


def _apply_delog_on_vector(
    vector: np_.ndarray,
    qids: Iterable[int],
    qid_to_logly: dict[int, bool],
    /,
) -> np_.ndarray:
    """
    Delogarithmize the elements of numpy vector that have True log-status
    """
    logly_index = [ qid_to_logly[qid] for qid in qids ]
    if any(logly_index):
        vector[logly_index] = np_.exp(vector[logly_index])
    return vector

