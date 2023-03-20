"""
"""

#[
from __future__ import annotations

from IPython import embed
from typing import (Self, NoReturn, TypeAlias, Literal, )
from numbers import Number
from collections.abc import Iterable, Callable

import enum
import copy
import scipy
import numpy
import itertools
import functools
import operator

from . import variants
from .. import sources
from .. import parsers

from ..incidence import (Token, )
from ..dataman import databanks
from .. import equations
from ..equations import (Equation, )
from .. import quantities
from ..quantities import (QuantityKind, Quantity, )
from ..fords import descriptors
from ..fords import systems
from .. import exceptions
from ..dataman import databanks
from .. import evaluators
from . import getters
#]


SteadySolverReturn: TypeAlias = tuple[
    numpy.ndarray|None, Iterable[int]|None, 
    numpy.ndarray|None, Iterable[int]|None,
]


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


def _solve_steady_linear_flat(
    sys,
    /,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    #[
    """
    """
    pinv = numpy.linalg.pinv
    lstsq = scipy.linalg.lstsq
    vstack = numpy.vstack
    hstack = numpy.hstack
    A, B, C, F, G, H = sys.A, sys.B, sys.C, sys.F, sys.G, sys.H
    #
    # A @ Xi + B @ Xi{-1} + C = 0
    # F @ Y + G @ Xi + H = 0
    #
    # Xi = -pinv(A + B) @ C
    Xi, *_ = lstsq(-(A + B), C)
    dXi = numpy.zeros(Xi.shape)
    #
    # Y = -pinv(F) @ (G @ Xi + H)
    Y, *_ = lstsq(-F, G @ Xi + H)
    dY = numpy.zeros(Y.shape)
    #
    return Xi, Y, dXi, dY
    #]


def _solve_steady_linear_nonflat(
    sys,
    /,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    #[
    """
    """
    # pinv = numpy.linalg.pinv
    lstsq = numpy.linalg.lstsq
    vstack = numpy.vstack
    hstack = numpy.hstack
    A, B, C, F, G, H = sys.A, sys.B, sys.C, sys.F, sys.G, sys.H
    num_y = F.shape[0]
    k = 1
    #
    # A @ Xi + B @ Xi{-1} + C = 0:
    # -->
    # A @ Xi + B @ (Xi - dXi) + C = 0
    # A @ (Xi + k*dXi) + B @ (Xi + (k-1)*dXi) + C = 0
    #
    AB = vstack((
        hstack(( A + B, 0*A + (0-1)*B )),
        hstack(( A + B, k*A + (k-1)*B )),
    ))
    CC = vstack((
        C,
        C,
    ))
    # Xi_dXi = -pinv(AB) @ CC
    Xi_dXi, *_ = lstsq(-AB, CC, rcond=None)
    #
    # F @ Y + G @ Xi + H = 0:
    # -->
    # F @ Y + G @ Xi + H = 0
    # F @ (Y + k*dY) + G @ (Xi + k*dXi) + H = 0
    #
    FF = vstack((
        hstack(( F, 0*F )),
        hstack(( F, k*F )),
    ))
    GG = vstack((
        hstack(( G, 0*G )),
        hstack(( G, k*G )),
    ))
    HH = vstack((
        H,
        H,
    ))
    # Y_dY = -pinv(FF) @ (GG @ Xi_dXi + HH)
    Y_dY, *_ = lstsq(-FF, GG @ Xi_dXi + HH, rcond=None)
    #
    # Separate levels and changes
    #
    num_xi = A.shape[1]
    num_y = F.shape[1]
    Xi, dXi = (
        Xi_dXi[0:num_xi, ...],
        Xi_dXi[num_xi:, ...],
    )
    Y, dY = (
        Y_dY[0:num_y, ...],
        Y_dY[num_y:, ...]
    )
    #
    return Xi, Y, dXi, dY
    #]


class Model(getters.GetterMixin):
    """
    """
    #[
    def __init__(self):
        self._quantities: list[Quantity] = []
        self._dynamic_equations: list[Equation] = []
        self._steady_equations: list[Equation] = []
        self._variants: list[variants.Variant] = []
        self._min_shift: int|None = None
        self._max_shift: int|None = None

    def assign(
        self: Self,
        /,
        **kwargs, 
    ) -> Self:
        """
        """
        try:
            qid_to_value = _rekey_dict(kwargs, quantities.create_name_to_qid(self._quantities))
        except KeyError as _KeyError:
            raise exceptions.UnknownName(_KeyError.args[0])
        for v in self._variants:
            v.update_values_from_dict(qid_to_value)
        #
        self._enforce_auto_values()
        return self

    def assign_from_databank(
        self, 
        databank: databanks.Databank,
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
        self._quantities = quantities.change_logly(self._quantities, new_logly, qids)

    @property
    def num_variants(self, /, ) -> int:
        return len(self._variants)

    @property
    def is_linear(self, /, ) -> bool:
        return self._flags.is_linear

    @property
    def is_flat(self, /, ) -> bool:
        return self._flags.is_flat

    def create_steady_evaluator(
        self,
        equations: Equations | ... = ... ,
        variant: Variant | ... | None = ... ,
        /,
    ) -> evaluators.SteadyEvaluator:
        evaluator = evaluators.SteadyEvaluator.for_model(self, equations if equations is not ... else self._steady_equations)
        if variant is not None:
            evaluator.update_steady_array(self, variant if variant is not ... else self._variants[0])
        return evaluator

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        return quantities.create_name_to_qid(self._quantities)

    def create_qid_to_name(self, /, ) -> dict[int, str]:
        return quantities.create_qid_to_name(self._quantities)


    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        return quantities.create_qid_to_logly(self._quantities)

    def create_steady_array(
        self,
        /,
        variant: variants.Variant|None = None,
        **kwargs,
    ) -> numpy.ndarray:
        qid_to_logly = self.create_qid_to_logly()
        if variant is None:
            variant = self._variants[0]
        return variant.create_steady_array(qid_to_logly, **kwargs, )

    def create_zero_array(
        self,
        /,
        variant: variants.Variant|None = None,
        **kwargs,
    ) -> numpy.ndarray:
        """
        """
        qid_to_logly = self.create_qid_to_logly()
        if variant is None:
            variant = self._variants[0]
        return variant.create_zero_array(qid_to_logly, **kwargs, )

    def _enforce_auto_values(self: Self, /, ) -> NoReturn:
        # Reset levels of shocks to zero, remove changes
        assign_shocks = { 
            qid: (0, numpy.nan) 
            for qid in quantities.generate_qids_by_kind(self._quantities, QuantityKind.SHOCK)
        }
        self._variants[0].update_values_from_dict(assign_shocks)
        #
        # Remove changes from quantities that are not logly variables
        assign_non_logly = { 
            qid: (..., numpy.nan) 
            for qid in  quantities.generate_qids_by_kind(self._quantities, ~QuantityKind.LOGLY_VARIABLE)
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


    def _systemize(
        self,
        variant: Variant,
        descriptor: descriptors.Descriptor,
        /,
    ) -> systems.System:
        """
        Evaluatoe 
        """
        num_columns = descriptor.system_differn_context.shape_data[1]
        logly_context = self.create_qid_to_logly()
        value_context = self.create_zero_array(variant, num_columns=num_columns)
        return systems.System.for_descriptor(descriptor, logly_context, value_context)

    #def _get_steady_something(
    #    self,
    #    /,
    #    extractor_from_variant: Callable,
    #) -> dict[str, Number|numpy.ndarray]:
    #    """
    #    """
    #    return databanks.Databank._from_dict({ name: extractor(qid) for qid, name in qid_to_name.items() })

    def _solve(
        self,
        variant: variants.Variant,
        /,
    ) -> NoReturn:
        system = self._systemize(variant, self._dynamic_descriptor)
        return system

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
        equations_switch: Literal["dynamic"] | Literal["steady"] = "dynamic",
        /,
        tolerance: float = 1e-12,
        details: bool = False,
    ) -> tuple[bool, Iterable[bool], Iterable[Number], Iterable[numpy.ndarray]]:
        evaluator = self._steady_evaluator_for_dynamic_equations
        dis = [ evaluator.update_steady_array(self, v).eval() for v in self._variants ]
        max_abs_dis = [ numpy.max(numpy.abs(d)) for d in dis ]
        status = [ d < tolerance for d in max_abs_dis ]
        all_status = all(status)
        return (all_status, status, max_abs_dis, dis) if details else all_status

    def _apply_delog(
        self,
        vector: numpy.ndarray,
        qids: Iterable[int],
        /,
    ) -> numpy.ndarray:
        """
        """
        qid_to_logly = self.create_qid_to_logly()
        logly_index = [ qid_to_logly[qid] for qid in qids ]
        if any(logly_index):
            vector[logly_index] = numpy.exp(vector[logly_index])
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
        #
        sys = self._systemize(variant, self._steady_descriptor)
        #
        # Calculate steady state for this variant
        #
        Xi, Y, dXi, dY = algorithm(sys)
        levels = numpy.hstack(( Xi.flat, Y.flat ))
        changes = numpy.hstack(( dXi.flat, dY.flat ))
        #
        # Extract only tokens with zero shift
        #
        tokens = list(itertools.chain(
            self._steady_descriptor.system_vectors.transition_variables,
            self._steady_descriptor.system_vectors.measurement_variables,
        ))
        zero_shift_index = [ not t.shift for t in tokens ]
        qids = [ t.qid for t in itertools.compress(tokens, zero_shift_index) ]
        levels = levels[zero_shift_index]
        levels = self._apply_delog(levels, qids)
        changes = self._apply_delog(changes, qids)
        changes = changes[zero_shift_index]
        #
        return levels, qids, changes, qids


    _steady_linear_flat = functools.partialmethod(_steady_linear, algorithm=_solve_steady_linear_flat)
    _steady_linear_nonflat = functools.partialmethod(_steady_linear, algorithm=_solve_steady_linear_nonflat)


    def _steady_nonlinear_flat(
        self,
        variant: Variant,
        /,
    ) -> SteadySolverReturn:
        return None, None, None, None


    def _steady_nonlinear_nonflat(
        self,
        variant: Variant,
        /,
    ) -> SteadySolverReturn:
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
            default_std = _DEFAULT_STD_LINEAR if ModelFlags.LINEAR not in self._flags else _DEFAULT_STD_NONLINEAR
        self.assign(**{ k: default_std for k in quantities.generate_quantity_names_by_kind(self._quantities, QuantityKind.STD) })


    def _populate_min_max_shifts(self) -> NoReturn:
        self._min_shift = equations.get_min_shift_from_equations(
            self._dynamic_equations + self._steady_equations
        )
        self._max_shift = equations.get_max_shift_from_equations(
            self._dynamic_equations + self._steady_equations
        )


    @classmethod
    def from_source(
        cls: type,
        model_source: sources.ModelSource,
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

        name_to_qid = quantities.create_name_to_qid(self._quantities)
        equations.finalize_dynamic_equations(self._dynamic_equations, name_to_qid)
        equations.finalize_steady_equations(self._steady_equations, name_to_qid)

        self._variants = [ variants.Variant(self._quantities) ]
        self._enforce_auto_values()
        self._dynamic_descriptor = descriptors.Descriptor(self._dynamic_equations, self._quantities)
        self._steady_descriptor = descriptors.Descriptor(self._steady_equations, self._quantities)

        self._steady_evaluator_for_dynamic_equations = self.create_steady_evaluator(self._dynamic_equations, None) 
        self._steady_evaluator_for_steady_equations = self.create_steady_evaluator(self._steady_equations, None) 

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
        model_source, info = sources.ModelSource.from_string(
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
        source_string = parsers.common.combine_source_files(source_files)
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

