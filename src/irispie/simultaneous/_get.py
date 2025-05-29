"""
"""


#[

from __future__ import annotations

from typing import Literal, Callable
from collections.abc import Iterable
from numbers import Real
import functools as _ft
import json as _js
import numpy as _np

from .. import equations as _equations
from .. import quantities as _quantities
from ..quantities import QuantityKind
from .. import has_variants as _has_variants
from ..has_variants import unpack_singleton_decorator as _unpack_singleton
from ..series import main as _series
from ..incidences import main as _incidence
from ..databoxes.main import Databox
from ..fords.solutions import EigenvalueKind, Solution
from ..fords import descriptors as _descriptors


from . import _flags

#]


def _cast_as_output_type(func: Callable, ):
    """
    """
    #[
    @_ft.wraps(func)
    def _wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        output_type = kwargs.get("output_type", Databox, )
        return output_type(output, ) if output_type else output
    return _wrapper
    #]


def _unpack_singleton_in_dict(func: Callable, ):
    #[
    @_ft.wraps(func)
    def _wrapper(self, *args, **kwargs, ):
        unpack_singleton = kwargs.pop("unpack_singleton", True)
        output = func(self, *args, **kwargs)
        return self.unpack_singleton_in_dict(output, unpack_singleton=unpack_singleton, )
    return _wrapper
    #]


class Inlay(
    _quantities.Mixin,
):
    """
    Frontend getter methods for Simultaneous objects
    """
    #[

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_steady_levels(
        self,
        unpack_singleton: bool = True,
        kind: _quantities.QuantityKind | None = None,
        **kwargs,
    ) -> dict[str, list[Real] | Real]:
        """
················································································

==Get steady-state levels of variables==

```
steady_levels = self.get_steady_levels(
    *,
    round: int = None,
    unpack_singleton: bool = True,
    output_typ: type = Databox,
)
```

### Input arguments ###


???+ input "self"
    `Simultaneous` model object with a valid first-order solution.

???+ input "round"
    Number of decimal places to round the output values.

???+ input "unpack_singleton"
    If `True`, unpack singleton lists and return the single element. If
    `False`, return the list as is.

???+ input "output_type"
    Cast output as this type; the constructor for this type must accept a
    dictionary.


### Returns ###


???+ returns "steady_levels"
    Databox with the steady-state levels for each variable.

......................................................................
        """
        kind = kind if kind is not None else _resolve_steady_kind(**kwargs, )
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, kind, )
        return self._get_values_as_dict("levels", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_steady_changes(
        self,
        unpack_singleton: bool = True,
        kind: _quantities.QuantityKind | None = None,
        **kwargs,
    ) -> dict[str, list[Real] | Real]:
        """
················································································

==Get steady-state changes of variables==

```
steady_changes = self.get_steady_changes(
    *,
    round: int = None,
    unpack_singleton: bool = True,
    output_typ: type = Databox,
)
```


### Input arguments ###


???+ input "self"
    `Simultaneous` model object with a valid first-order solution.

???+ input "round"
    Number of decimal places to round the output values.

???+ input "unpack_singleton"
    If `True`, unpack singleton lists and return the single element. If
    `False`, return the list as is.

???+ input "output_type"
    Cast output as this type; the constructor for this type must accept a
    dictionary.


### Returns ###


???+ returns "steady_changes"
    Databox with the steady-state changes for each variable, i.e. first
    differences or gross rates of change depending on the log status of
    each variable.

......................................................................
        """
        kind = kind if kind is not None else _resolve_steady_kind(**kwargs, )
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, kind, )
        return self._get_values_as_dict("changes", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_steady(
        self,
        unpack_singleton: bool = True,
        kind: _quantities.QuantityKind | None = None,
        **kwargs,
    ) -> dict[str, list[tuple[Real, Real]] | tuple[Real, Real]]:
        """
        """
        kind = kind if kind is not None else _resolve_steady_kind(**kwargs, )
        qids = tuple(_quantities.generate_qids_by_kind(self._invariant.quantities, kind, ))
        levels = self._get_values_as_dict("levels", qids, **kwargs, )
        changes = self._get_values_as_dict("changes", qids, **kwargs, )
        return {
            k: list(zip(levels[k], changes[k]))
            for k in levels.keys()
        }

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_parameters(
        self,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        """
        """
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.PARAMETER, )
        return self._get_values_as_dict("levels", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_stds(
        self,
        kind: _quantities.QuantityKind | None = None,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        """
        """
        std_qids = self._get_std_qids(kind=kind, )
        return self._get_values_as_dict("levels", std_qids, **kwargs, )

    def _get_std_qids(
        self,
        kind: _quantities.QuantityKind | None = None,
    ) -> tuple[int, ...]:
        """
        """
        default_kind = _quantities.QuantityKind.ANY_STD
        kind = default_kind & kind if kind is not None else default_kind
        return _quantities.generate_qids_by_kind(self._invariant.quantities, kind, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_parameters_stds(
        self,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        """
        """
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.PARAMETER_OR_STD, )
        return self._get_values_as_dict("levels", qids, **kwargs, )

    def get_initials(
        self,
    ) -> _incidence.Tokens:
        """
        Get required list of initial conditions
        """
        # Get the tokens of state vector with actual incidence in the
        # model, and lag them by one period to get initial conditions
        initial_tokens = [
            _incidence.Token(t.qid, t.shift-1)
            for t in self._invariant.dynamic_descriptor.solution_vectors.get_initials()
        ]
        qid_to_name = self.create_qid_to_name()
        qid_to_logly = {} # [^1]
        return _incidence.print_tokens(initial_tokens, qid_to_name, qid_to_logly, )
        # [^1] Do not wrap initial conditions in log(...)

    def _get_dynamic_solution_vectors(self, ) -> _descriptors.SolutionVectors:
        r"""
        FordSimulatableProtocol
        """
        return self._invariant.dynamic_descriptor.solution_vectors

    def get_solution_vectors(self, ) -> _descriptors.HumanSolutionVectors:
        """
        Get the solution vectors of the model
        """
        qid_to_name = self.create_qid_to_name()
        qid_to_logly = self.create_qid_to_logly()
        return _descriptors.HumanSolutionVectors(
            self._dynamic_solution_vectors,
            qid_to_name,
            qid_to_logly,
        )

    def get_solution(
        self,
        #
        unpack_singleton: bool = True,
    ) -> Solution | list[Solution]:
        """
        """
        solution_matrices = [ i for i in self.iter_solution() ]
        return self.unpack_singleton(solution_matrices, unpack_singleton=unpack_singleton, )

    def iter_solution(self, ) -> Iterable[Solution]:
        """
        """
        for v in self._variants:
            yield v.solution

    def _gets_solution(
        self,
        deviation: bool = False,
        vid: int = 0,
    ) -> Solution:
        r"""
        """
        return (
            self._variants[vid].solution if not deviation
            else self._variants[vid].deviation_solution
        )

    def iter_std_name_to_value(self, ) -> Iterable[dict[str, Real]]:
        """
        """
        qid_to_name = self.create_qid_to_name()
        std_qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.ANY_STD, )
        for v in self._variants:
            std_qid_to_value = v.retrieve_levels_as_dict(std_qids, )
            yield {
                qid_to_name[qid]: value
                for qid, value in std_qid_to_value.items()
            }

    def get_dynamic_equations(
        self,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[_equations.Equation]:
        return tuple(
            _equations.generate_equations_of_kind(self._invariant.dynamic_equations, kind)
            if kind else self._invariant.dynamic_equations
        )

    def get_human_equations(
        self,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[str]:
        def _concatenate(dynamic: str, steady: str) -> str:
            return f"{dynamic} !! {steady}" if steady != dynamic else dynamic
        zipper = zip(
            self.get_dynamic_equations(kind=kind, ),
            self.get_steady_equations(kind=kind, ),
        )
        return tuple(
            _concatenate(dynamic.human, steady.human, )
            for dynamic, steady in zipper
        )

    def get_steady_equations(
        self,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[_equations.Equation]:
        return tuple(
            _equations.generate_equations_of_kind(self._invariant.steady_equations, kind)
            if kind else self._invariant.steady_equations
        )

    def get_flags(
        self,
    ) -> _flags.Flags:
        return self._invariant._flags

    def get_context(
        self,
    ) -> dict[str, Callable]:
        return self._invariant._context

    def _get_values_as_dict(
        self,
        variant_attr: Literal["levels", "changes"],
        qids: Iterable[int],
        **kwargs,
    ) -> dict[str, Any]:
        """
        """
        custom_round = kwargs.pop("round", None, )
        def apply(value: Real) -> Real:
            return (
                round(value, custom_round)
                if custom_round is not None and value is not None
                else value
            )
        #
        qid_to_name = self.create_qid_to_name()
        out_dict = {}
        for qid in qids:
            out_dict[qid_to_name[qid]] = [
                apply(getattr(v, variant_attr, )[qid], )
                for v in self._variants
            ]
        return out_dict

    def get_equation_descriptions(self, ) -> dict[str, str]:
        """
        """
        return _equations.create_human_to_description(self._invariant.dynamic_equations, )

    @_unpack_singleton
    def get_eigenvalues(
        self,
        kind: EigenvalueKind = EigenvalueKind.ALL,
        unpack_singleton: bool = True,
    ) -> tuple[Real, ...] | list[tuple[Real, ...]]:
        """
        Eigenvalues
        """
        return [
            tuple(e for e, s in zip(v.solution.eigenvalues, v.solution.eigenvalues_stability, ) if s in kind)
            for v in self._variants
        ]

    @_unpack_singleton
    def get_eigenvalues_stability(
        self,
        kind: EigenvalueKind = EigenvalueKind.ALL,
        unpack_singleton: bool = True,
    ):
        return [ 
            tuple(i for i in v.solution.eigenvalues_stability if i in kind)
            for v in self._variants
        ]

    def _get_variable_stability_for_variant(self, variant, ) -> dict[str, bool]:
        """
        """
        vec = self._invariant.dynamic_descriptor.solution_vectors
        qid_to_name = self.create_qid_to_name()
        return {
            qid_to_name[token.qid]: \
                variant.solution.transition_vector_stability[index] == EigenvalueKind.STABLE
            for index, token in enumerate(vec.transition_variables)
            if token.shift == 0
        } | {
            qid_to_name[token.qid]: \
                variant.solution.measurement_vector_stability[index] == EigenvalueKind.STABLE
            for index, token in enumerate(vec.measurement_variables)
            if token.shift == 0
        }

    @_unpack_singleton
    def get_variable_stability(
        self,
        unpack_singleton: bool = True,
    ) -> dict[str, bool] | list[dict[str, bool]]:
        """
        """
        return [
            self._get_variable_stability_for_variant(v, )
            for v in self._variants
        ]

    def generate_minus_control_quantities(self, ) -> tuple[int]:
        """
        """

    def map_name_to_minus_control_func(self, ) -> tuple[str]:
        """
        """
        minus_control_func = {
            True: lambda x, y: x / y,
            False: lambda x, y: x - y,
            None: lambda x, y: x - y,
        }
        kind = _quantities.ANY_VARIABLE | _quantities.ANY_SHOCK
        return {
            q.human: minus_control_func[q.logly]
            for q in self._invariant.quantities
            if q.kind in kind
        }

    #]


# Decorate methods from quantities.Mixin
Inlay.get_log_status = _cast_as_output_type(Inlay.get_log_status, )


def _resolve_steady_kind(
    include_shocks: bool = False,
    **kwargs,
) -> _quantities.QuantityKind:
    """
    """
    #[
    return (
        QuantityKind.LOGGABLE_VARIABLE if not include_shocks
        else QuantityKind.LOGGABLE_VARIABLE_OR_ANY_SHOCK
    )
    #]

