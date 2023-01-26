"""
"""

#[
from __future__ import annotations

from typing import Self, NoReturn
from numbers import Number
from collections.abc import Iterable, Callable

import enum
import copy
import numpy

from .sources import (
    Source,
)
from .incidence import (
    Token,
    get_max_shift, get_min_shift,
)
from .equations import (
    EquationKind, Equation,
    finalize_equations_from_humans,
    generate_all_tokens_from_equations
)
from .quantities import (
    QuantityKind, Quantity,
    create_name_to_qid, create_qid_to_name, create_qid_to_logly,
    generate_all_qids, generate_all_quantity_names, 
    generate_qids_by_kind,
    get_max_qid, change_logly
)
from .audi import (
    Context, DiffernAtom,
)
from .metaford import (
    SystemVectors, SolutionVectors,
    SystemMap,
)
from .ford import (
    System
)
from .exceptions import (
    UnknownName
)
#]


class ModelFlags(enum.IntFlag, ):
    """
    """
    #[
    LINEAR = enum.auto()
    GROWTH = enum.auto()

    def is_linear(self, /) -> bool:
        return self in ModelFlags.LINEAR

    def is_growth(self, /) -> bool:
        return self in ModelFlags.GROWTH

    @classmethod
    def from_kwargs(cls: type, **kwargs) -> Self:
        self = cls(0)
        if kwargs.get("linear", False):
            self |= ModelFlags.LINEAR
        if kwargs.get("growth", False):
            self |= ModelFlags.GROWTH
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

    def update_values(self, update:dict) -> NoReturn:
        for qid in self._values.keys() & update.keys():
            self._values[qid] = update[qid]

    def create_steady_array(self, num_columns: int=1) -> numpy.ndarray:
        steady_column = numpy.array(
            self.prepare_values_for_steady_array(shift=0),
            ndmin=2, dtype=float,
        ).transpose()
        steady_array = numpy.tile(steady_column, (1, num_columns))
        return steady_array

    def prepare_values_for_steady_array(self, shift:int=0) -> Iterable:
        _max_qid = max(self._values.keys())
        return [ self._values.get(qid, self._missing_value) for qid in range(_max_qid+1) ]
    #]


class Model:
    """
    """
    #[
    def __init__(self):
        self._quantities: list[Quantity] = []
        self._qid_to_logly: dict[int, bool] = {}
        self._equations: list[Equation] = []
        self._variants: list[Variant] = []


    def assign(self: Self, variant: int=0, **kwargs, ) -> Self:
        """
        """
        try:
            update = _rekey_dict(kwargs, create_name_to_qid(self._quantities))
        except KeyError as _KeyError:
            raise UnknownName(_KeyError.args[0])

        self._variants[variant].update_values(update)
        return self


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
        return self._flags.is_linear()


    @property
    def is_growth(self, /) -> bool:
        return self._flags.is_growth()


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


    def create_steady_array(self, /, variant: int=0, **kwargs, ) -> numpy.ndarray:
        return self._variants[variant].create_steady_array(**kwargs)


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
        self._system_vectors = SystemVectors(self._equations, self._quantities)
        self._solution_vectors = SolutionVectors(self._system_vectors)
        self._system_map = SystemMap(self._system_vectors)

        system_equations = self._system_vectors.generate_system_equations_from_equations(self._equations)
        self._system_differn_context = Context.for_equations(
           DiffernAtom, 
           system_equations,
           self._system_vectors.eid_to_wrt_tokens,
        )


    def systemize(
        self, 
        variant: int = 0,
        /
    ) -> Self:
        """
        """
        num_columns = self._system_differn_context.shape_data[1]
        value_context = self.create_steady_array(variant=variant, num_columns=num_columns)
        logly_context = self.create_qid_to_logly()

        tt = self._system_differn_context.eval(value_context, logly_context)

        td = numpy.vstack([x.diff for x in tt])
        tc = numpy.vstack([x.value for x in tt])

        ma = self._system_map
        sy = self._system_vectors

        system = System()

        system.A = numpy.zeros(sy.shape_AB_excl_dynid, dtype=float)
        system.A[ma.A.lhs] = td[ma.A.rhs]
        system.A = numpy.vstack((system.A, ma.dynid_A))

        system.B = numpy.zeros(sy.shape_AB_excl_dynid, dtype=float)
        system.B[ma.B.lhs] = td[ma.B.rhs]
        system.B = numpy.vstack((system.B, ma.dynid_B))

        system.C = numpy.zeros(sy.shape_C_excl_dynid, dtype=float)
        system.C = numpy.vstack((system.C, ma.dynid_C))

        system.D = numpy.zeros(sy.shape_D_excl_dynid, dtype=float)
        system.D[ma.D.lhs] = td[ma.D.rhs]
        system.D = numpy.vstack((system.D, ma.dynid_D))

        system.F = numpy.zeros(sy.shape_F, dtype=float)
        system.F[ma.F.lhs] = td[ma.F.rhs]

        system.G = numpy.zeros(sy.shape_G, dtype=float)
        system.G[ma.G.lhs] = td[ma.G.rhs]

        system.H = numpy.zeros(sy.shape_H, dtype=float)

        system.J = numpy.zeros(sy.shape_J, dtype=float)
        system.J[ma.J.lhs] = td[ma.J.rhs]

        self._variants[variant]._system = system


    @classmethod
    @_resolve_model_flags
    def from_source(
        cls: type,
        source: Source,
        **kwargs,
    ) -> Self:
        self = cls()
        self._quantities = copy.deepcopy(source.quantities)
        self._equations = copy.deepcopy(source.equations)
        name_to_qid = create_name_to_qid(self._quantities)
        finalize_equations_from_humans(self._equations, name_to_qid)
        self._variants = [ Variant(self._quantities) ]
        self._assign_auto_values()
        self._prepare_ford()
        return self
    #]


def _rekey_dict(dict_to_rekey: dict, old_key_to_new_key: dict, /) -> dict:
    return { 
        old_key_to_new_key[key]: value 
        for key, value in dict_to_rekey.items()
    }

