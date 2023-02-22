"""
"""

#[
from __future__ import annotations

from IPython import embed
from scipy import linalg
import time
from typing import Self, NoReturn, TypeAlias
from numbers import Number
from collections.abc import Iterable, Callable

import enum
import copy
import numpy
import itertools
import operator

from . import variants
from .. import sources

from ..incidence import (
    Token,
    get_max_shift, get_min_shift,
)
from ..equations import (
    Equation,
    finalize_equations_from_humans,
    generate_all_tokens_from_equations
)
from ..quantities import (
    QuantityKind, Quantity,
    create_name_to_qid, create_qid_to_name, create_qid_to_logly,
    generate_qids_by_kind,
    change_logly
)

from .. import metaford
from .. import systems
from .. import exceptions
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


def _resolve_model_flags(func: Callable, /, ) -> Callable:
    """
    Decorator for resolving model flags
    """
    def wrapper(*args, **kwargs, ) -> Callable:
        model = func(*args, **kwargs, )
        model._flags = ModelFlags.from_kwargs(**kwargs, )
        return model
    return wrapper


class Model:
    """
    """
    #[
    def __init__(self):
        self._quantities: list[Quantity] = []
        self._dynamic_equations: list[Equation] = []
        self._steady_equations: list[Equation] = []
        self._variants: list[variants.Variant] = []

    def assign(
        self: Self,
        /,
        **kwargs, 
    ) -> Self:
        """
        """
        try:
            qid_to_value = _rekey_dict(kwargs, create_name_to_qid(self._quantities))
        except KeyError as _KeyError:
            raise exceptions.UnknownName(_KeyError.args[0])
        for v in self._variants:
            v.update_values_from_dict(qid_to_value)
        return self

    def assign_from_databank(
        self, 
        databank: Databank,
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
        self._quantities = change_logly(self._quantities, new_logly, qids)

    @property
    def num_variants(self, /, ) -> int:
        return len(self._variants)

    @property
    def is_linear(self, /, ) -> bool:
        return self._flags.is_linear

    @property
    def is_flat(self, /, ) -> bool:
        return self._flags.is_flat

    def _get_max_shift(self: Self, /, ) -> int:
        return get_max_shift(self._collect_all_tokens)

    def _get_min_shift(self: Self, /, ) -> int:
        return get_min_shift(self._collect_all_tokens)

    def create_steady_evaluator(self, /, ) -> SteadyEvaluator:
        return SteadyEvaluator(self)

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        return create_name_to_qid(self._quantities)

    def create_qid_to_name(self, /, ) -> dict[int, str]:
        return create_qid_to_name(self._quantities)

    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        return create_qid_to_logly(self._quantities)

    def _create_steady_array(
        self,
        variant: Variant,
        /,
        **kwargs,
    ) -> numpy.ndarray:
        qid_to_logly = self.create.qid_to_logly()
        return variant.create_steady_array(qid_to_logly, **kwargs, )

    def _create_zero_array(
        self,
        variant: Variant,
        /,
        **kwargs,
    ) -> numpy.ndarray:
        """
        """
        qid_to_logly = self.create_qid_to_logly()
        levels = variant.levels
        changes = variant.changes
        update = { qid:(float(logly), float(logly), ) for qid, logly in qid_to_logly.items() if logly is not None }
        levels = variants.update_levels_from_dict(levels, update, )
        changes = variants.update_changes_from_dict(changes, update, )
        return variants.create_steady_array(levels, changes, qid_to_logly, **kwargs, )

    def _assign_auto_values(self: Self, /, ) -> NoReturn:
        assign_shocks = { qid: (0, numpy.nan) for qid in  generate_qids_by_kind(self._quantities, QuantityKind.SHOCK) }
        self._variants[0].update_values_from_dict(assign_shocks)

    def _shrink_num_variants(self, new_num: int, /, ) -> NoReturn:
        if new_num<1:
            raise Exception('Number of variants must be one or more')
        self._variants = self._variants[0:new_num]

    def _expand_num_variants(self, new_num: int, /, ) -> NoReturn:
        for i in range(self.num_variants, new_num):
            self._variants.append(copy.deepcopy(self._variants[-1]))

    def _collect_all_tokens(self, /, ) -> set[Token]:
        return set(generate_all_tokens_from_equations(self._equations))


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


    def _systemize_steady_linear(
        self,
        variant: Variant,
        /,
    ) -> systems.System:
        """
        """
        smf = self._steady_metaford
        num_columns = smf.system_differn_context.shape_data[1]
        logly_context = self.create_qid_to_logly()
        value_context = self._create_zero_array(variant, num_columns=num_columns)
        return systems.System.for_model(smf, logly_context, value_context)


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


    def _choose_steady_solver(
        self,
        **kwargs,
    ) -> Callable:
        _STEADY_SOLVER = {
            ModelFlags.DEFAULT: self._steady_nonlinear_nonflat,
            ModelFlags.LINEAR: self._steady_linear_nonflat,
            ModelFlags.FLAT: self._steady_nonlinear_flat,
            ModelFlags.LINEAR | ModelFlags.FLAT: self._steady_linear_flat,
        }
        model_flags = ModelFlags.update_from_kwargs(self._flags, **kwargs)
        return _STEADY_SOLVER[model_flags]


    def _steady_linear_flat(
        self, 
        variant: Variant,
        /,
    ) -> SteadySolverReturn:
        """
        """
        pinv = linalg.pinv
        smf = self._steady_metaford
        sys = self._systemize_steady_linear(variant)

        # A @ Xi + B @ Xi{-1} + C = 0
        # F @ Y + G @ Xi + H = 0
        Xi = -pinv(sys.A + sys.B) @ sys.C
        Y = -pinv(sys.F) @ (sys.G @ Xi + sys.H)

        levels = numpy.hstack((Xi.flat, Y.flat))
        tokens = list(itertools.chain(
            smf.system_vectors.transition_variables,
            smf.system_vectors.measurement_variables,
        ))

        # Extract only tokens with zero shift
        zero_shift_index = [ not t.shift for t in tokens ]
        zero_shift_levels = levels[zero_shift_index]
        qids_levels = [ t.qid for t in itertools.compress(tokens, zero_shift_index) ]

        return zero_shift_levels, qids_levels, None, None


    def _steady_linear_nonflat(
        self,
        variant: Variant,
        /,
    ) -> SteadySolverReturn:
        return []


    def _steady_nonlinear_flat(
        self,
        variant: Variant,
        /,
    ) -> SteadySolverReturn:
        return []


    def _steady_nonlinear_nonflat(
        self,
        variant: Variant,
        /,
    ) -> SteadySolverReturn:
        return []


    @classmethod
    @_resolve_model_flags
    def from_source(
        cls: type,
        model_source: sources.ModelSource,
        /,
        **kwargs,
    ) -> Self:
        """
        """
        self = cls()
        self._quantities = model_source.quantities[:]
        self._dynamic_equations = model_source.dynamic_equations[:]
        self._steady_equations = model_source.steady_equations[:]

        name_to_qid = create_name_to_qid(self._quantities)
        finalize_equations_from_humans(self._dynamic_equations, name_to_qid)
        finalize_equations_from_humans(self._steady_equations, name_to_qid)

        self._variants = [ variants.Variant(self._quantities) ]
        self._assign_auto_values()
        self._dynamic_metaford = metaford.Metaford(self._dynamic_equations, self._quantities)
        self._steady_metaford = metaford.Metaford(self._steady_equations, self._quantities)
        return self


    @classmethod
    def from_string(
        cls,
        source_string: str,
        /,
        context: dict | None = None,
        save_preparsed: str = "",
        **kwargs,
    ) -> tuple[Self, dict]:
        """
        """
        model_source, info = sources.ModelSource.from_string(
            source_string, context=context, save_preparsed=save_preparsed,
        )
        return Model.from_source(model_source, **kwargs)
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

