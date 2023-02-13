"""
"""

#[
from __future__ import annotations

from IPython import embed
from scipy import linalg
import time
from typing import Self, NoReturn
from numbers import Number
from collections.abc import Iterable, Callable

import enum
import copy
import numpy

from .. import sources

from ..incidence import (
    Token,
    get_max_shift, get_min_shift,
)
from ..equations import (
    EquationKind, Equation,
    finalize_equations_from_humans,
    generate_all_tokens_from_equations
)
from ..quantities import (
    QuantityKind, Quantity,
    create_name_to_qid, create_qid_to_name, create_qid_to_logly,
    generate_all_qids, generate_all_quantity_names, 
    generate_qids_by_kind,
    get_max_qid, change_logly
)
from ..audi import (
    Context, DiffernAtom,
)
from ..metaford import (
    SystemVectors, SolutionVectors,
    SystemMap,
)
from ..systems import (
    System
)
from ..exceptions import (
    UnknownName
)
#]


class ModelFlags(enum.IntFlag, ):
    """
    """
    #[
    LINEAR = enum.auto()
    GROWTH = enum.auto()
    DEFAULT = 0

    @property
    def is_linear(self, /, ) -> bool:
        return ModelFlags.LINEAR in self

    @property
    def is_growth(self, /, ) -> bool:
        return ModelFlags.GROWTH in self

    @classmethod
    def update(cls, self, /, **kwargs) -> Self:
        linear = kwargs.get("linear") if kwargs.get("linear") is not None else self.is_linear
        growth = kwargs.get("growth") if kwargs.get("growth") is not None else self.is_growth
        return cls.from_kwargs(linear=linear, growth=growth)

    @classmethod
    def from_kwargs(cls: type, **kwargs, ) -> Self:
        self = cls.DEFAULT
        if kwargs.get("linear"):
            self |= cls.LINEAR
        if kwargs.get("growth"):
            self |= cls.GROWTH
        return self
    #]


def _resolve_model_flags(func: Callable, /) -> Callable:
    """
    Decorator for resolving model flags
    """
    def wrapper(*args, **kwargs, ) -> Callable:
        model = func(*args, **kwargs, )
        model._flags = ModelFlags.from_kwargs(**kwargs, )
        return model
    return wrapper


class Variant:
    """
    """
    _missing_value: Any | None = None
    _values: dict[int, Any] | None = None
    _system: System | None = None
    #[
    def __init__(self, quantities:Quantities) -> NoReturn:
        self._initilize_values(quantities)
        self._system = System()

    def _initilize_values(self, quantities:Quantities) -> NoReturn:
        self._values = { qty.id: self._missing_value for qty in quantities }

    def update_values(self, update: dict) -> NoReturn:
        for qid in self._values.keys() & update.keys():
            self._values[qid] = update[qid]

    def create_steady_array(
        self,
        /,
        override_values: dict | None = None,
        num_columns: int=1,
    ) -> numpy.ndarray:
        """
        """
        steady_column = numpy.array(
            self.prepare_values_for_steady_array(override_values=override_values, shift=0),
            ndmin=2, dtype=float,
        ).transpose()
        steady_array = numpy.tile(steady_column, (1, num_columns))
        return steady_array

    def prepare_values_for_steady_array(self, override_values=None, shift:int=0) -> Iterable:
        _max_qid = max(self._values.keys())
        values = self._values | override_values if override_values else self._values
        return [ 
            values.get(qid, self._missing_value) 
            for qid in range(_max_qid+1)
        ]
    #]


class Model:
    """
    """
    #[
    def __init__(self):
        self._quantities: list[Quantity] = []
        self._qid_to_logly: dict[int, bool] = {}
        self._dynamic_equations: list[Equation] = []
        self._steady_equations: list[Equation] = []
        self._variants: list[Variant] = []


    def assign(
        self: Self,
        /,
        _variant: int | ... = ...,
        **kwargs, 
    ) -> Self:
        """
        """
        try:
            qid_to_value = _rekey_dict(kwargs, create_name_to_qid(self._quantities))
        except KeyError as _KeyError:
            raise UnknownName(_KeyError.args[0])
        _variant = resolve_variant(self, _variant)
        for v in _variant:
            self._variants[v].update_values(qid_to_value)
        return self


    def assign_from_databank(
        self, 
        databank: Databank,
        /,
        _variant = ...,
    ) -> Self:
        return self.assign(_variant=_variant, **databank.__dict__)


    def change_num_variants(self, new_num: int) -> NoReturn:
        """
        """
        if new_num<self.num_variants:
            self._shrink_num_variants(new_num)
        elif new_num>self.num_variants:
            self._expand_num_variants(new_num)


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
    def num_variants(self, /) -> int:
        return len(self._variants)

    @property
    def is_linear(self, /) -> bool:
        return self._flags.is_linear

    @property
    def is_growth(self, /) -> bool:
        return self._flags.is_growth


    def _get_max_shift(self: Self, /) -> int:
        return get_max_shift(self._collect_all_tokens)


    def _get_min_shift(self: Self, /) -> int:
        return get_min_shift(self._collect_all_tokens)


    def create_steady_evaluator(self, /) -> SteadyEvaluator:
        return SteadyEvaluator(self)


    def create_name_to_qid(self, /) -> dict[str, int]:
        return create_name_to_qid(self._quantities)


    def create_qid_to_name(self, /) -> dict[int, str]:
        return create_qid_to_name(self._quantities)


    def create_qid_to_logly(self, /) -> dict[int, bool]:
        return create_qid_to_logly(self._quantities)


    def create_steady_array(self, /, _variant: int=0, **kwargs, ) -> numpy.ndarray:
        return self._variants[_variant].create_steady_array(**kwargs)


    def create_zero_array(self, /, _variant: int=0, **kwargs, ) -> numpy.ndarray:
        qid_to_logly = self.create_qid_to_logly()
        override_values = {qid: int(qid_to_logly[qid]) for qid in qid_to_logly.keys() if qid_to_logly[qid] is not None }
        return self._variants[_variant].create_steady_array(override_values=override_values, **kwargs)


    def _assign_auto_values(self: Self, /) -> NoReturn:
        assign_shocks = { qid: 0 for qid in  generate_qids_by_kind(self._quantities, QuantityKind.SHOCK) }
        self._variants[0].update_values(assign_shocks)


    def _shrink_num_variants(self, new_num: int, /) -> NoReturn:
        if new_num<1:
            raise Exception('Number of variants must be one or more')
        self._variants = self._variants[0:new_num]


    def _expand_num_variants(self, new_num: int, /) -> NoReturn:
        for i in range(self.num_variants, new_num):
            self._variants.append(copy.deepcopy(self._variants[-1]))


    def _collect_all_tokens(self, /) -> set[Token]:
        return set(generate_all_tokens_from_equations(self._equations))


    def _prepare_ford(self, /) -> NoReturn:
        """
        """
        self._system_vectors = SystemVectors(self._dynamic_equations, self._quantities)
        self._solution_vectors = SolutionVectors(self._system_vectors)
        self._system_map = SystemMap(self._system_vectors)
        system_equations = self._system_vectors.generate_system_equations_from_equations(self._dynamic_equations)
        self._system_differn_context = Context.for_equations(
           DiffernAtom, 
           system_equations,
           self._system_vectors.eid_to_wrt_tokens,
        )


    def systemize(
        self,
        /,
        _variant: int | ... = ...,
        linear: bool | None = None,
        growth: bool | None = None,
    ) -> Self:
        """
        """
        model_flags = ModelFlags.update(self._flags, linear=linear, growth=growth)
        for v in resolve_variant(self, _variant):
            self._systemize(v, model_flags)
        return self


    def _systemize(
        self, 
        _variant: int,
        model_flags: ModelFlags,
        /,
    ) -> NoReturn:
        """
        """
        num_columns = self._system_differn_context.shape_data[1]
        logly_context = self.create_qid_to_logly()
        create_array = self.create_steady_array if not model_flags.is_linear else self.create_zero_array
        value_context = create_array(_variant=_variant, num_columns=num_columns)

        # Differentiate and evaluate constant
        tt = self._system_differn_context.eval(value_context, logly_context)
        td = numpy.vstack([x.diff for x in tt])
        tc = numpy.vstack([x.value for x in tt])

        map = self._system_map
        vec = self._system_vectors

        system = System()

        system.A = numpy.zeros(vec.shape_AB_excl_dynid, dtype=float)
        system.A[map.A.lhs] = td[map.A.rhs]
        system.A = numpy.vstack((system.A, map.dynid_A))

        system.B = numpy.zeros(vec.shape_AB_excl_dynid, dtype=float)
        system.B[map.B.lhs] = td[map.B.rhs]
        system.B = numpy.vstack((system.B, map.dynid_B))

        system.C = numpy.zeros(vec.shape_C_excl_dynid, dtype=float)
        system.C[map.C.lhs] = tc[map.C.rhs]
        system.C = numpy.vstack((system.C, map.dynid_C))

        system.D = numpy.zeros(vec.shape_D_excl_dynid, dtype=float)
        system.D[map.D.lhs] = td[map.D.rhs]
        system.D = numpy.vstack((system.D, map.dynid_D))

        system.F = numpy.zeros(vec.shape_F, dtype=float)
        system.F[map.F.lhs] = td[map.F.rhs]

        system.G = numpy.zeros(vec.shape_G, dtype=float)
        system.G[map.G.lhs] = td[map.G.rhs]

        system.H = numpy.zeros(vec.shape_H, dtype=float)
        system.H[map.H.lhs] = tc[map.H.rhs]

        system.J = numpy.zeros(vec.shape_J, dtype=float)
        system.J[map.J.lhs] = td[map.J.rhs]

        self._variants[_variant]._system = system


    def _steady_linear_no_growth(
        self, 
        _variant: int,
    ) -> NoReturn:
        """
        """
        sys = self._variants[_variant]._system

        transition_steady_level = -linalg.pinv(sys.A+sys.B) @ sys.C
        self._variants[_variant].update_values({
            t.qid:float(x) 
            for t, x in zip(self._system_vectors.transition_variables, transition_steady_level) 
            if not t.shift
        })

        measurement_steady_level = -linalg.pinv(sys.F) @ (sys.G @ transition_steady_level + sys.H)
        self._variants[_variant].update_values({
            t.qid:float(x) 
            for t, x in zip(self._system_vectors.measurement_variables, measurement_steady_level) 
            if not t.shift
        })


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

        self._variants = [ Variant(self._quantities) ]
        self._assign_auto_values()
        self._prepare_ford()
        return self


    @classmethod
    def from_string(
        cls,
        source_string: str,
        /,
        context: dict | None = None,
        **kwargs,
    ) -> tuple[Self, dict]:
        """
        """
        model_source, info = sources.ModelSource.from_string(source_string, context=context)
        return Model.from_source(model_source, **kwargs)
    #]


def _rekey_dict(dict_to_rekey: dict, old_key_to_new_key: dict, /) -> dict:
    return { 
        old_key_to_new_key[key]: value 
        for key, value in dict_to_rekey.items()
    }


def resolve_variant(self, _variant) -> Iterable[int]:
    #[
    if isinstance(_variant, Number):
        resolved_variant = [_variant,]
    elif _variant is Ellipsis:
        return range(self.num_variants)
    else:
        return _variant
    #]

