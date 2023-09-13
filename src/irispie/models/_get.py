"""
"""


#[
from __future__ import annotations

from typing import (Literal, Callable, )
from collections.abc import (Iterable, )
from numbers import (Number, )
import functools as _ft
import json as _js

from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from .. import dates as _dates
from ..incidences import main as _incidence
from ..databanks import main as _databanks
from ..fords import descriptors as _descriptors

from . import _flags
#]


"""
Quantities that are time series in model databanks
"""
_TIME_SERIES_QUANTITY = _quantities.QuantityKind.VARIABLE | _quantities.QuantityKind.SHOCK


_DATABANK_OUTPUT_FORMAT_RESOLUTION = {
    "dict": lambda x: x,
    "Databank": lambda x: _databanks.Databank._from_dict(x),
    "databank": lambda x: _databanks.Databank._from_dict(x),
    "json": lambda x: js_.dumps(x),
    "json4": lambda x: js_.dumps(x, indent=4),
}


def _decorate_output_format(func: Callable, ):
    """
    """
    #[
    def _wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        output_format = kwargs.get("output", "Databank")
        return _DATABANK_OUTPUT_FORMAT_RESOLUTION[output_format](output)
    return _wrapper
    #]


class GetMixin:
    """
    Frontend getter methods for Model objects
    """
    #[
    def _get_values_from_primary_variant(
        self,
        /,
        variant_attr: Literal["levels"] | Literal["changes"],
        kind: _quantities.QuantityKind,
        **kwargs,
    ) -> dict[str, Number]:
        """
        """
        qid_to_name = self.create_qid_to_name()
        qids = list(_quantities.generate_qids_by_kind(self._invariant._quantities, kind))
        x = self._variants[0].retrieve_values(variant_attr, qids)
        return {
            qid_to_name[q]: float(x[i, 0])
            for i, q in enumerate(qids)
        }

    @_decorate_output_format
    def get_steady_levels(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        return self._get_values_from_primary_variant(variant_attr="levels", kind=_sources.LOGLY_VARIABLE, **kwargs, )

    @_decorate_output_format
    def get_steady_changes(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        return self._get_values_from_primary_variant(variant_attr="changes", kind=_sources.LOGLY_VARIABLE, **kwargs, )

    @_decorate_output_format
    def get_steady(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        levels = self._get_values_from_primary_variant(variant_attr="levels", kind=_sources.LOGLY_VARIABLE, **kwargs, )
        changes = self._get_values_from_primary_variant(variant_attr="changes", kind=_sources.LOGLY_VARIABLE, **kwargs, )
        return { k: (levels[k], changes[k]) for k in levels.keys() }

    @_decorate_output_format
    def get_parameters(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        return self._get_values_from_primary_variant(variant_attr="levels", kind=_quantities.QuantityKind.PARAMETER)

    @_decorate_output_format
    def get_stds(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        return self._get_values_from_primary_variant(variant_attr="levels", kind=_quantities.QuantityKind.STD)

    @_decorate_output_format
    def get_parameters_stds(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        return self._get_values_from_primary_variant(variant_attr="levels", kind=_quantities.QuantityKind.PARAMETER_OR_STD)

    @_decorate_output_format
    def get_log_status(
        self,
        /,
        **kwargs,
    ) -> dict[str, bool]:
        return {
            qty.human: qty.logly
            for qty in self._invariant._quantities
            if qty.kind in _sources.LOGLY_VARIABLE
        }

    def get_initials(
        self,
        /,
        kind: Literal["required"] | Literal["discarded"] = "required",
    ) -> _incidence.Tokens:
        """
        Get required list of initial conditions
        """
        # Get the tokens of state vector with actual incidence in the
        # model, and lag them by one period to get initial conditions
        initial_tokens = [
            _incidence.Token(t.qid, t.shift-1)
            for t in self._invariant._dynamic_descriptor.solution_vectors.get_initials(kind, )
        ]
        return _incidence.print_tokens(initial_tokens, self.create_qid_to_name(), )

    def _get_steady_databank(
        self,
        start_date: _dates.Dater,
        end_date: _dates.Dater,
        /,
        deviation: bool = False,
    ) -> _databanks.Databank:
        """
        """
        num_columns = int(end_date - start_date + 1)
        shift_in_first_column = start_date.get_distance_from_origin()
        #
        array = self.create_some_array(
            deviation=deviation,
            num_columns=num_columns,
            shift_in_first_column=shift_in_first_column,
        )
        #
        qid_to_kind = self.create_qid_to_kind()
        qid_to_name = {
            qid: (name if qid_to_kind[qid] in _TIME_SERIES_QUANTITY else "")
            for qid, name in self.create_qid_to_name().items()
        }
        qid_to_descriptor = self.create_qid_to_descriptor()
        #
        return _databanks.Databank._from_array(
            array, qid_to_name, start_date, 
            array_orientation="horizontal",
            interpret_dates="start_date",
            qid_to_descriptor=qid_to_descriptor,
        )

    def get_solution_vectors(self, /, ) -> _descriptors.SolutionVectors:
        """
        Get the solution vectors of the model
        """
        return self._invariant._dynamic_descriptor.solution_vectors

    def get_all_solution_matrices(self, /, ):
        return [ v.solution for v in self._variants ]

    def get_solution_matrices(self, /, ):
        return self._variants[0].solution

    def get_dynamic_equations(
        self,
        /,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[_equations.Equation]:
        return tuple(
            _equations.generate_equations_of_kind(self._invariant._dynamic_equations, kind)
            if kind else self._invariant._dynamic_equations
        )

    def get_steady_equations(
        self,
        /,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[_equations.Equation]:
        return tuple(
            _equations.generate_equations_of_kind(self._invariant._steady_equations, kind)
            if kind else self._invariant._steady_equations
        )

    def get_quantities(
        self,
        /,
        *,
        kind: _quantities.QuantityKind | None = None,
    ) -> _quantities.Quantities:
        return tuple(
            _quantities.generate_quantities_of_kind(self._invariant._quantities, kind)
            if kind else self._invariant._quantities
        )

    def get_flags(
        self,
        /,
    ) -> _flags.Flags:
        return self._invariant._flags

    def get_custom_functions(
        self,
        /,
    ) -> dict[str, Callable]:
        return self._invariant._function_context
    #]

