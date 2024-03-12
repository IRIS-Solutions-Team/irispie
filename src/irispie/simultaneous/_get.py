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
from ..series import main as _series
from ..incidences import main as _incidence
from ..databoxes import main as _databoxes
from ..fords import descriptors as _descriptors

from . import _flags
#]


"""
Quantities that are time series in model databoxes
"""
_TIME_SERIES_QUANTITY = _quantities.QuantityKind.ANY_VARIABLE | _quantities.QuantityKind.ANY_SHOCK


def _decorate_output_format(func: Callable, ):
    """
    """
    #[
    def _wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        output_format = kwargs.get("output", "Databox")
        if output_format == "Databox":
            output = _databoxes.Databox.from_dict(output, )
        return output
    return _wrapper
    #]


class GetMixin:
    """
    Frontend getter methods for Simultaneous objects
    """
    #[

    @_decorate_output_format
    def get_steady_levels(
        self,
        /,
        **kwargs,
    ) -> dict[str, Real]:
        kind = _resolve_steady_kind(**kwargs, )
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, kind, )
        return self._get_values("levels", qids, **kwargs, )

    @_decorate_output_format
    def get_steady_changes(
        self,
        /,
        **kwargs,
    ) -> dict[str, Real]:
        kind = _resolve_steady_kind(**kwargs, )
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, kind, )
        return self._get_values("changes", qids, **kwargs, )

    @_decorate_output_format
    def get_steady(
        self,
        /,
        **kwargs,
    ) -> dict[str, Real]:
        """
        """
        kind = _resolve_steady_kind(**kwargs, )
        qids = tuple(_quantities.generate_qids_by_kind(self._invariant.quantities, kind, ))
        levels = self._get_values("levels", qids, **kwargs, )
        changes = self._get_values("changes", qids, **kwargs, )
        if self.is_singleton:
            out = {
                k: (levels[k], changes[k], )
                for k in levels.keys()
            }
        else:
            out = { k: None for k in levels.keys() }
            for k in levels.keys():
                out[k] = list(zip(levels[k], changes[k]))
        return out

    @_decorate_output_format
    def get_parameters(
        self,
        /,
        **kwargs,
    ) -> dict[str, Real]:
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.PARAMETER, )
        return self._get_values("levels", qids, **kwargs, )

    @_decorate_output_format
    def get_stds(
        self,
        /,
        **kwargs,
    ) -> dict[str, Real]:
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.ANY_STD, )
        return self._get_values("levels", qids, **kwargs, )

    @_decorate_output_format
    def get_parameters_stds(
        self,
        /,
        **kwargs,
    ) -> dict[str, Real]:
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.PARAMETER_OR_STD, )
        return self._get_values("levels", qids, **kwargs, )

    @_decorate_output_format
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
        if self.is_singleton:
            def insert_value(out_dict, name, value):
                out_dict[name] = apply(value)
        else:
            def insert_value(out_dict, name, value):
                out_dict.setdefault(name, []).append(apply(value))
        #
        qids = tuple(qids)
        qid_to_name = self.create_qid_to_name()
        out_dict = {}
        #
        for v in self._variants:
            values = v.retrieve_values(variant_attr, qids)
            for qid, value in zip(qids, values):
                insert_value(out_dict, qid_to_name[qid], value)
        return out_dict

    def get_descriptions(self, ) -> dict[str, str]:
        """
        """
        return (
            _quantities.create_name_to_description(self._invariant.quantities, )
            | _equations.create_human_to_description(self._invariant.dynamic_equations, )
        )

    def get_eigenvalues(
        self,
        /,
        unpack_singleton: bool = True,
    ):
        eigenvalues = [ v.solution.eigenvalues for v in self._variants ]
        return self.unpack_singleton(eigenvalues, unpack_singleton=unpack_singleton, )

    def get_eigenvalues_stability(
        self,
        /,
        unpack_singleton: bool = True,
    ):
        stability = [ v.solution.eigenvalues_stability for v in self._variants ]
        return self.unpack_singleton(stability, unpack_singleton=unpack_singleton, )

    #]


def _resolve_steady_kind(
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

