"""
"""


#[
from __future__ import annotations

from typing import (Literal, Callable, )
from collections.abc import (Iterable, )
from numbers import (Real, )
import functools as _ft
import json as _js
import numpy as _np

from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from .. import dates as _dates
from .. import has_variants as _has_variants
from ..series import main as _series
from ..incidences import main as _incidence
from ..databoxes import main as _databoxes
from ..fords import solutions as _solutions
from ..fords import descriptors as _descriptors

from . import _flags
#]


"""
Quantities that are time series in model databoxes
"""
_TIME_SERIES_QUANTITY = _quantities.QuantityKind.ANY_VARIABLE | _quantities.QuantityKind.ANY_SHOCK


def _cast_as_output_type(func: Callable, ):
    """
    """
    #[
    @_ft.wraps(func)
    def _wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        output_type = kwargs.get("output_type", _databoxes.Databox, )
        return output_type(output, ) if output_type else output
    return _wrapper
    #]


def _unpack_singleton(func: Callable, ):
    #[
    @_ft.wraps(func)
    def _wrapper(self, *args, **kwargs, ):
        unpack_singleton = kwargs.pop("unpack_singleton", True)
        output = func(self, *args, **kwargs)
        return self.unpack_singleton(output, unpack_singleton=unpack_singleton, )
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


class Inlay:
    """
    Frontend getter methods for Simultaneous objects
    """
    #[

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_steady_levels(
        self,
        /,
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
        return self._get_values("levels", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_steady_changes(
        self,
        /,
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
        return self._get_values("changes", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_steady(
        self,
        /,
        unpack_singleton: bool = True,
        kind: _quantities.QuantityKind | None = None,
        **kwargs,
    ) -> dict[str, list[tuple[Real, Real]] | tuple[Real, Real]]:
        """
        """
        kind = kind if kind is not None else _resolve_steady_kind(**kwargs, )
        qids = tuple(_quantities.generate_qids_by_kind(self._invariant.quantities, kind, ))
        levels = self._get_values("levels", qids, **kwargs, )
        changes = self._get_values("changes", qids, **kwargs, )
        return {
            k: list(zip(levels[k], changes[k]))
            for k in levels.keys()
        }

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_parameters(
        self,
        /,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        """
        """
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.PARAMETER, )
        return self._get_values("levels", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_stds(
        self,
        /,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        """
        """
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.ANY_STD, )
        return self._get_values("levels", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_parameters_stds(
        self,
        /,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        """
        """
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.PARAMETER_OR_STD, )
        return self._get_values("levels", qids, **kwargs, )

    @_cast_as_output_type
    def get_log_status(
        self,
        /,
    ) -> dict[str, bool]:
        return {
            qty.human: qty.logly
            for qty in self._invariant.quantities
            if qty.kind in _sources.LOGLY_VARIABLE
        }

    def get_initials(
        self,
        /,
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

    def generate_steady_items(
        self,
        start_date: _dates.Dater,
        end_date: _dates.Dater,
        /,
        deviation: bool = False,
    ) -> _databoxes.Databox:
        """
        """
        num_columns = int(end_date - start_date + 1)
        shift_in_first_column = start_date.get_distance_from_origin()
        qid_to_name = self.create_qid_to_name()
        qid_to_description = self.create_qid_to_description()
        #
        array = self.create_some_array(
            deviation=deviation,
            num_columns=num_columns,
            shift_in_first_column=shift_in_first_column,
        )
        num_rows = array.shape[0]
        #
        qid_to_kind = self.create_qid_to_kind()
        remove_qids = tuple(
            qid
            for qid, kind in qid_to_kind.items()
            if kind not in _TIME_SERIES_QUANTITY
        )
        names = (
            qid_to_name[qid]
            for qid in range(num_rows, )
            if qid in remove_qids
        )
        descriptions = (
            qid_to_description[qid]
            for qid in range(num_rows, )
            if qid in remove_qids
        )
        for qid in qid_to_name.keys():
            if qid_to_kind[qid] in _TIME_SERIES_QUANTITY:
                yield qid_to_name[qid], _series.Series(
                    start_date=start_date,
                    values=array[qid, :].reshape(-1, 1),
                    description=qid_to_description[qid],
                )
            else:
                yield qid_to_name[qid], array[qid, 0]

    def get_solution_vectors(self, /, ) -> _descriptors.HumanSolutionVectors:
        """
        Get the solution vectors of the model
        """
        qid_to_name = self.create_qid_to_name()
        qid_to_logly = self.create_qid_to_logly()
        return _descriptors.HumanSolutionVectors(
            self._solution_vectors,
            qid_to_name,
            qid_to_logly,
        )

    def get_solution_matrices(
        self,
        /,
        unpack_singleton: bool = True,
    ):
        """
        """
        solution_matrices = [ v.solution for v in self._variants ]
        return self.unpack_singleton(solution_matrices, unpack_singleton=unpack_singleton, )

    def get_dynamic_equations(
        self,
        /,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[_equations.Equation]:
        return tuple(
            _equations.generate_equations_of_kind(self._invariant.dynamic_equations, kind)
            if kind else self._invariant.dynamic_equations
        )

    def get_std_qids_for_shock_qids(
        self: Self,
        shock_qids: Iterable[int],
        /,
    ) -> tuple[int]:
        """
        """
        return tuple(
            self._invariant.shock_qid_to_std_qid[qid]
            for qid in shock_qids
        )

    def get_steady_equations(
        self,
        /,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[_equations.Equation]:
        return tuple(
            _equations.generate_equations_of_kind(self._invariant.steady_equations, kind)
            if kind else self._invariant.steady_equations
        )

    def get_quantities(
        self,
        /,
        *,
        kind: _quantities.QuantityKind | None = None,
    ) -> _quantities.Quantities:
        return tuple(
            _quantities.generate_quantities_of_kind(self._invariant.quantities, kind, )
            if kind else self._invariant.quantities
        )

    def get_names(
        self,
        /,
        *,
        kind: _quantities.QuantityKind | None = None,
    ) -> _quantities.Quantities:
        return tuple(q.human for q in self.get_quantities(kind=kind, ))

    def get_flags(
        self,
        /,
    ) -> _flags.Flags:
        return self._invariant._flags

    def get_context(
        self,
        /,
    ) -> dict[str, Callable]:
        return self._invariant._context

    def _get_values(
        self,
        variant_attr: Literal["levels", "changes"],
        qids: Iterable[int],
        /,
        **kwargs,
    ) -> dict[str, Any]:
        """
        """
        custom_round = kwargs.pop("round", None, )
        def apply(value: Real) -> Real:
            value = float(value)
            return round(value, custom_round) if custom_round is not None else value
        #
        qids = tuple(qids)
        qid_to_name = self.create_qid_to_name()
        out_dict = {}
        #
        for v in self._variants:
            values = v.retrieve_values(variant_attr, qids)
            for qid, value in zip(qids, values):
                out_dict.setdefault(qid_to_name[qid], []).append(apply(value))
        return out_dict

    def get_descriptions(self, ) -> dict[str, str]:
        """
        """
        return (
            _quantities.create_name_to_description(self._invariant.quantities, )
            | _equations.create_human_to_description(self._invariant.dynamic_equations, )
        )

    @_unpack_singleton
    def get_eigenvalues(
        self,
        /,
        *,
        unpack_singleton: bool = True,
    ) -> list[_solutions.Eigenvalues] | _solutions.Eigenvalues:
        """Eigenvalues
        """
        return [ v.solution.eigenvalues for v in self._variants ]

    @_unpack_singleton
    def get_eigenvalues_stability(
        self,
        /,
        *,
        unpack_singleton: bool = True,
    ):
        return [ v.solution.eigenvalues_stability for v in self._variants ]

    #]


def _resolve_steady_kind(
    *,
    include_shocks: bool = False,
    **kwargs,
) -> _quantities.QuantityKind:
    """
    """
    #[
    return (
        _sources.LOGLY_VARIABLE if not include_shocks
        else _sources.LOGLY_VARIABLE_OR_ANY_SHOCK
    )
    #]

