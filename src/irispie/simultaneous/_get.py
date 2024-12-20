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
from .. import sources as _sources
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
    r"""
    ................................................................................
    
    ==Decorator to cast output to a specified type==
    
    This decorator wraps a function, allowing its output to be cast to a type
    specified via the `output_type` keyword argument. If `output_type` is not
    provided, the output is returned as-is.

    ### Input arguments ###

    ???+ input "func"
        A callable to be wrapped. The wrapped function must return a result that
        can be cast to the desired type.

    ### Returns ###

    ???+ returns "Callable"
        A wrapped function that returns the original output or the cast output
        type.
    
    ................................................................................
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
    r"""
    ................................................................................
    
    ==Decorator to handle single-element unpacking in dictionaries==
    
    This decorator modifies a function's return dictionary, optionally unpacking
    singleton lists into their single elements. This behavior can be controlled
    using the `unpack_singleton` keyword argument.

    ### Input arguments ###

    ???+ input "func"
        A callable to be wrapped. The wrapped function must return a dictionary.

    ### Returns ###

    ???+ returns "Callable"
        A wrapped function that applies unpacking logic to its dictionary output.
    
    ................................................................................
    """
    #[
    @_ft.wraps(func)
    def _wrapper(self, *args, **kwargs, ):
        unpack_singleton = kwargs.pop("unpack_singleton", True)
        output = func(self, *args, **kwargs)
        return self.unpack_singleton_in_dict(output, unpack_singleton=unpack_singleton, )
    return _wrapper
    #]


class Inlay:
    r"""
    ................................................................................
    
    ==Frontend getter methods for Simultaneous objects==
    
    The `Inlay` class provides a suite of getter methods for extracting steady-state
    levels, changes, parameters, and other simulation details from a `Simultaneous`
    model object. This class interacts with internal model data, applying flexible
    formatting, filtering, and aggregation to provide a user-friendly interface.
    
    Attributes are inherited and enhanced by specialized decorators to support
    single-element unpacking, type casting, and contextual lookup for enhanced
    accessibility.
    
    ................................................................................
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
        r"""
        ............................................................................
        
        ==Get steady-state levels of variables==
        
        This method retrieves steady-state levels of all quantities in the model,
        optionally filtered by type. The output can be formatted as lists or scalars,
        and cast into a specific type for downstream use.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with a valid first-order solution.

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists and return the single element. If
            `False`, return the list as is.

        ???+ input "kind"
            Specifies the type of quantities to retrieve. If `None`, all quantities
            are included.

        ???+ input "**kwargs"
            Additional arguments controlling filtering and formatting.

        ### Returns ###

        ???+ returns "steady_levels"
            A dictionary with quantity names as keys and their steady-state levels
            as values. Values are either lists or single elements depending on
            `unpack_singleton`.

        ### Example ###

        ```python
            steady_levels = model.get_steady_levels(unpack_singleton=True)
            print(steady_levels)
        ```
        
        ............................................................................
        """
        kind = kind if kind is not None else _resolve_steady_kind(**kwargs, )
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, kind, )
        return self._get_values_as_dict("levels", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_steady_changes(
        self,
        /,
        unpack_singleton: bool = True,
        kind: _quantities.QuantityKind | None = None,
        **kwargs,
    ) -> dict[str, list[Real] | Real]:
        r"""
        ............................................................................
        
        ==Get steady-state changes of variables==
        
        This method calculates the steady-state changes for all quantities in the
        model, such as first differences or growth rates. The output can be
        formatted as lists or scalars, depending on the `unpack_singleton` option.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with a valid first-order solution.

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists and return the single element. If
            `False`, return the list as is.

        ???+ input "kind"
            Specifies the type of quantities to retrieve. If `None`, all quantities
            are included.

        ???+ input "**kwargs"
            Additional arguments controlling filtering and formatting.

        ### Returns ###

        ???+ returns "steady_changes"
            A dictionary with quantity names as keys and their steady-state changes
            as values. Values are either lists or single elements depending on
            `unpack_singleton`.

        ### Example ###

        ```python
            steady_changes = model.get_steady_changes(unpack_singleton=True)
            print(steady_changes)
        ```
        
        ............................................................................
        """
        kind = kind if kind is not None else _resolve_steady_kind(**kwargs, )
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, kind, )
        return self._get_values_as_dict("changes", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_steady(
        self,
        /,
        unpack_singleton: bool = True,
        kind: _quantities.QuantityKind | None = None,
        **kwargs,
    ) -> dict[str, list[tuple[Real, Real]] | tuple[Real, Real]]:
        r"""
        ............................................................................
        
        ==Get steady-state levels and changes==
        
        This method retrieves both steady-state levels and changes for all
        quantities in the model, organizing them into paired tuples. This allows
        simultaneous analysis of levels and rates of change.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with a valid first-order solution.

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists and return the single element. If
            `False`, return the list as is.

        ???+ input "kind"
            Specifies the type of quantities to retrieve. If `None`, all quantities
            are included.

        ???+ input "**kwargs"
            Additional arguments controlling filtering and formatting.

        ### Returns ###

        ???+ returns "steady"
            A dictionary with quantity names as keys and paired tuples of
            steady-state levels and changes as values. Values are either lists or
            single elements depending on `unpack_singleton`.

        ### Example ###

        ```python
            steady_data = model.get_steady(unpack_singleton=True)
            print(steady_data)
        ```
        
        ............................................................................
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
        /,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        r"""
        ............................................................................
        
        ==Retrieve model parameters==
        
        This method extracts all parameter values from the model, allowing their
        direct access as a dictionary. The `unpack_singleton` option enables
        simplifying lists of parameters to scalars when applicable.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with defined parameters.

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists and return the single element. If
            `False`, return the list as is.

        ???+ input "**kwargs"
            Additional arguments controlling formatting.

        ### Returns ###

        ???+ returns "parameters"
            A dictionary with parameter names as keys and their values as values.

        ### Example ###

        ```python
            parameters = model.get_parameters(unpack_singleton=True)
            print(parameters)
        ```
        
        ............................................................................
        """
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.PARAMETER, )
        return self._get_values_as_dict("levels", qids, **kwargs, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_stds(
        self,
        /,
        kind: _quantities.QuantityKind | None = None,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        r"""
        ............................................................................
        
        ==Retrieve standard deviations of quantities==
        
        This method retrieves standard deviations for the specified types of
        quantities in the model. The `unpack_singleton` option allows single-value
        lists to be reduced to scalars for convenience.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing quantity definitions.

        ???+ input "kind"
            Specifies the type of quantities for which standard deviations should
            be retrieved. If `None`, includes all applicable quantities.

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists and return the single element. If
            `False`, return the list as is.

        ???+ input "**kwargs"
            Additional arguments for formatting or filtering.

        ### Returns ###

        ???+ returns "stds"
            A dictionary with quantity names as keys and their standard deviations
            as values.

        ### Example ###

        ```python
            stds = model.get_stds(unpack_singleton=True)
            print(stds)
        ```
        
        ............................................................................
        """
        std_qids = self._get_std_qids(kind=kind, )
        return self._get_values_as_dict("levels", std_qids, **kwargs, )

    def _get_std_qids(
        self,
        /,
        kind: _quantities.QuantityKind | None = None,
    ) -> tuple[int, ...]:
        r"""
        ............................................................................
        
        ==Get QIDs for standard deviations==
        
        This method determines the unique quantity IDs (QIDs) associated with
        standard deviations in the model. These QIDs can be used to retrieve or
        manipulate specific standard deviation values.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing quantity metadata.

        ???+ input "kind"
            Specifies the type of quantities for which QIDs should be generated. If
            `None`, includes all applicable quantities.

        ### Returns ###

        ???+ returns "std_qids"
            A tuple of integer QIDs for the relevant standard deviations.

        ### Example ###

        ```python
            std_qids = model._get_std_qids(kind=_quantities.QuantityKind.PARAMETER)
            print(std_qids)
        ```
        
        ............................................................................
        """
        default_kind = _quantities.QuantityKind.ANY_STD
        kind = default_kind & kind if kind is not None else default_kind
        return _quantities.generate_qids_by_kind(self._invariant.quantities, kind, )

    @_cast_as_output_type
    @_unpack_singleton_in_dict
    def get_parameters_stds(
        self,
        /,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Real]:
        r"""
        ............................................................................
        
        ==Retrieve parameters and their standard deviations==
        
        This method retrieves both parameter values and their associated standard
        deviations from the model. The results are organized into a dictionary for
        easy access.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with parameter definitions.

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists and return the single element. If
            `False`, return the list as is.

        ???+ input "**kwargs"
            Additional arguments for filtering or formatting.

        ### Returns ###

        ???+ returns "parameters_stds"
            A dictionary where keys are parameter names and values are their
            standard deviations.

        ### Example ###

        ```python
            parameters_stds = model.get_parameters_stds(unpack_singleton=True)
            print(parameters_stds)
        ```
        
        ............................................................................
        """
        qids = _quantities.generate_qids_by_kind(self._invariant.quantities, _quantities.QuantityKind.PARAMETER_OR_STD, )
        return self._get_values_as_dict("levels", qids, **kwargs, )

    @_cast_as_output_type
    def get_log_status(self, **kwargs, ) -> dict[str, bool]:
        r"""
        ............................................................................
        
        ==Retrieve log status of variables==
        
        This method identifies whether each quantity in the model is specified in
        logarithmic terms. The log status is derived from the model's internal
        metadata.

        ### Input arguments ###

        ???+ input "**kwargs"
            Reserved for future filtering or formatting options.

        ### Returns ###

        ???+ returns "log_status"
            A dictionary where keys are quantity names and values are booleans,
            indicating whether the quantity is specified in logarithmic form.

        ### Example ###

        ```python
            log_status = model.get_log_status()
            print(log_status)
        ```
        
        ............................................................................
        """
        return {
            qty.human: qty.logly
            for qty in self._invariant.quantities
            if qty.kind in _sources.LOGGABLE_VARIABLE
        }

    def get_initials(
        self,
        /,
    ) -> _incidence.Tokens:
        r"""
        ............................................................................
        
        ==Get required initial conditions==
        
        This method retrieves the initial conditions required for the simulation.
        It identifies the state variables, shifts them by one period, and
        returns the tokens representing these initial conditions.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with valid state vector definitions.

        ### Returns ###

        ???+ returns "initials"
            Tokens representing the initial conditions for the state variables.

        ### Example ###

        ```python
            initials = model.get_initials()
            print(initials)
        ```
        
        ............................................................................
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

    def get_solution_vectors(self, /, ) -> _descriptors.HumanSolutionVectors:
        r"""
        ............................................................................
        
        ==Retrieve solution vectors of the model==
        
        This method fetches the solution vectors, including transition and
        measurement vectors, with human-readable names and log statuses.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with valid solution vectors.

        ### Returns ###

        ???+ returns "solution_vectors"
            Human-readable solution vectors with quantity names and log statuses.

        ### Example ###

        ```python
            solution_vectors = model.get_solution_vectors()
            print(solution_vectors)
        ```
        
        ............................................................................
        """
        qid_to_name = self.create_qid_to_name()
        qid_to_logly = self.create_qid_to_logly()
        return _descriptors.HumanSolutionVectors(
            self.solution_vectors,
            qid_to_name,
            qid_to_logly,
        )

    def get_solution(
        self,
        /,
        unpack_singleton: bool = True,
    ) -> Solution | list[Solution]:
        r"""
        ............................................................................
        
        ==Retrieve model solutions==
        
        This method retrieves the first-order solutions for all model variants.
        Solutions are represented by their transition matrices and eigenvalues.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with valid solutions.

        ???+ input "unpack_singleton"
            If `True`, unpack singleton solutions and return a single solution
            object. If `False`, return a list of solutions.

        ### Returns ###

        ???+ returns "solution"
            A single `Solution` object or a list of `Solution` objects, depending
            on the `unpack_singleton` option.

        ### Example ###

        ```python
            solution = model.get_solution(unpack_singleton=True)
            print(solution)
        ```
        
        ............................................................................
        """
        solution_matrices = [ v.solution for v in self._variants ]
        return self.unpack_singleton(solution_matrices, unpack_singleton=unpack_singleton, )

    def get_singleton_solution(
        self,
        deviation: bool = False,
        *,
        vid: int = 0,
    ) -> Solution:
        r"""
        ............................................................................
        
        ==Retrieve solution for a specific variant==
        
        This method retrieves the first-order solution for a specified model
        variant. Optionally, the solution can be returned in deviation form.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with valid solution data.

        ???+ input "deviation"
            If `True`, return the solution in deviation form. Default is `False`.

        ???+ input "vid"
            Variant ID for which the solution is retrieved. Default is `0`.

        ### Returns ###

        ???+ returns "singleton_solution"
            A `Solution` object for the specified variant, optionally in deviation
            form.

        ### Example ###

        ```python
            singleton_solution = model.get_singleton_solution(deviation=False, vid=0)
            print(singleton_solution)
        ```
        
        ............................................................................
        """
        solution = self._variants[vid].solution
        return (
            solution if not deviation
            else Solution.deviation_solution(solution, )
        )

    def get_dynamic_equations(
        self,
        /,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[_equations.Equation]:
        r"""
        ............................................................................
        
        ==Retrieve dynamic equations==
        
        This method fetches the dynamic equations of the model. Equations can be
        filtered by their kind, such as transition or measurement equations.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing dynamic equations.

        ???+ input "kind"
            Specifies the type of equations to retrieve. If `None`, includes all
            equations.

        ### Returns ###

        ???+ returns "dynamic_equations"
            A tuple of `Equation` objects representing the dynamic equations.

        ### Example ###

        ```python
            dynamic_eqs = model.get_dynamic_equations(kind=_equations.EquationKind.TRANSITION)
            print(dynamic_eqs)
        ```
        
        ............................................................................
        """
        return tuple(
            _equations.generate_equations_of_kind(self._invariant.dynamic_equations, kind)
            if kind else self._invariant.dynamic_equations
        )

    def get_human_equations(
        self,
        /,
        kind: _equations.EquationKind | None = None,
    ) -> tuple[str]:
        r"""
        ............................................................................
        
        ==Retrieve human-readable dynamic and steady equations==
        
        This method retrieves dynamic and steady equations in a human-readable
        format. It combines the equations into strings where dynamic and steady
        equations are separated by `!!`, unless they are identical.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing equations.

        ???+ input "kind"
            Specifies the type of equations to retrieve. If `None`, includes all
            equations.

        ### Returns ###

        ???+ returns "human_equations"
            A tuple of strings representing human-readable equations, formatted
            dynamically and with steady forms.

        ### Example ###

        ```python
            human_eqs = model.get_human_equations(kind=_equations.EquationKind.TRANSITION)
            print(human_eqs)
        ```
        
        ............................................................................
        """
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

    def get_std_qids_for_shock_qids(
        self: Self,
        shock_qids: Iterable[int],
        /,
    ) -> tuple[int]:
        r"""
        ............................................................................
        
        ==Map shock QIDs to their standard deviation QIDs==
        
        This method maps the unique quantity IDs (QIDs) of shocks to their
        corresponding standard deviation QIDs.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing shock metadata.

        ???+ input "shock_qids"
            An iterable of shock QIDs to map to their standard deviation QIDs.

        ### Returns ###

        ???+ returns "std_qids"
            A tuple of integers representing the standard deviation QIDs for the
            specified shocks.

        ### Example ###

        ```python
            shock_qids = [1, 2, 3]
            std_qids = model.get_std_qids_for_shock_qids(shock_qids)
            print(std_qids)
        ```
        
        ............................................................................
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
        r"""
        ............................................................................
        
        ==Retrieve steady-state equations==
        
        This method fetches the steady-state equations for the model. Equations
        can be filtered by their kind, such as transition or measurement equations.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing steady-state equations.

        ???+ input "kind"
            Specifies the type of equations to retrieve. If `None`, includes all
            equations.

        ### Returns ###

        ???+ returns "steady_equations"
            A tuple of `Equation` objects representing the steady-state equations.

        ### Example ###

        ```python
            steady_eqs = model.get_steady_equations(kind=_equations.EquationKind.TRANSITION)
            print(steady_eqs)
        ```
        
        ............................................................................
        """
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
        r"""
        ............................................................................
        
        ==Retrieve quantities by kind==
        
        This method fetches all quantities in the model that match a specific kind,
        such as parameters or shocks.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing quantity metadata.

        ???+ input "kind"
            Specifies the type of quantities to retrieve. If `None`, includes all
            quantities.

        ### Returns ###

        ???+ returns "quantities"
            A tuple of quantities matching the specified kind.

        ### Example ###

        ```python
            quantities = model.get_quantities(kind=_quantities.QuantityKind.PARAMETER)
            print(quantities)
        ```
        
        ............................................................................
        """
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
        r"""
        ............................................................................
        
        ==Retrieve human-readable names of quantities==
        
        This method retrieves the human-readable names of quantities that match a
        specified kind, such as parameters or shocks.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing quantity metadata.

        ???+ input "kind"
            Specifies the type of quantities for which names should be retrieved. If
            `None`, includes all quantities.

        ### Returns ###

        ???+ returns "names"
            A tuple of human-readable names for quantities matching the specified
            kind.

        ### Example ###

        ```python
            names = model.get_names(kind=_quantities.QuantityKind.PARAMETER)
            print(names)
        ```
        
        ............................................................................
        """
        return tuple(q.human for q in self.get_quantities(kind=kind, ))

    def get_flags(
        self,
        /,
    ) -> _flags.Flags:
        r"""
        ............................................................................
        
        ==Retrieve model flags==
        
        This method retrieves the flags associated with the model. Flags are used
        internally to denote specific attributes or behaviors of the model.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing flag metadata.

        ### Returns ###

        ???+ returns "flags"
            An instance of `_flags.Flags` containing the flags for the model.

        ### Example ###

        ```python
            flags = model.get_flags()
            print(flags)
        ```
        
        ............................................................................
        """
        return self._invariant._flags

    def get_context(
        self,
        /,
    ) -> dict[str, Callable]:
        r"""
        ............................................................................
        
        ==Retrieve model context==
        
        This method retrieves the context associated with the model, providing a
        mapping of context names to callable objects.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing contextual metadata.

        ### Returns ###

        ???+ returns "context"
            A dictionary mapping context names (strings) to callable objects.

        ### Example ###

        ```python
            context = model.get_context()
            print(context)
        ```
        
        ............................................................................
        """
        return self._invariant._context

    def _get_values_as_dict(
        self,
        variant_attr: Literal["levels", "changes"],
        qids: Iterable[int],
        /,
        **kwargs,
    ) -> dict[str, Any]:
        r"""
        ............................................................................
        
        ==Retrieve values as a dictionary==
        
        This internal method retrieves values (e.g., levels or changes) for
        specified quantities, organizes them into a dictionary, and applies
        optional rounding.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object used for extracting values.

        ???+ input "variant_attr"
            A string indicating the attribute to retrieve: `"levels"` or `"changes"`.

        ???+ input "qids"
            An iterable of quantity IDs to retrieve values for.

        ???+ input "**kwargs"
            Additional arguments, including `round`, to control output formatting.

        ### Returns ###

        ???+ returns "values_dict"
            A dictionary where keys are quantity names and values are lists of
            values corresponding to the specified attribute.

        ### Example ###

        ```python
            values = model._get_values_as_dict("levels", qids=[1, 2], round=2)
            print(values)
        ```
        
        ............................................................................
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

    def get_quantity_descriptions(self, ) -> dict[str, str]:
        r"""
        ............................................................................
        
        ==Retrieve quantity descriptions==
        
        This method retrieves the descriptions for all quantities in the model,
        mapping quantity names to their descriptions.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing quantity metadata.

        ### Returns ###

        ???+ returns "descriptions"
            A dictionary mapping quantity names (strings) to their descriptions
            (strings).

        ### Example ###

        ```python
            descriptions = model.get_quantity_descriptions()
            print(descriptions)
        ```
        
        ............................................................................
        """
        return _quantities.create_name_to_description(self._invariant.quantities, )

    def get_equation_descriptions(self, ) -> dict[str, str]:
        r"""
        ............................................................................
        
        ==Retrieve equation descriptions==
        
        This method retrieves the descriptions for all dynamic equations in the
        model, mapping human-readable equation strings to their descriptions.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing equation metadata.

        ### Returns ###

        ???+ returns "equation_descriptions"
            A dictionary mapping human-readable equations (strings) to their
            descriptions (strings).

        ### Example ###

        ```python
            eq_descriptions = model.get_equation_descriptions()
            print(eq_descriptions)
        ```
        
        ............................................................................
        """
        return _equations.create_human_to_description(self._invariant.dynamic_equations, )

    @_unpack_singleton
    def get_eigenvalues(
        self,
        /,
        *,
        kind: EigenvalueKind = EigenvalueKind.ALL,
        unpack_singleton: bool = True,
    ) -> tuple[Real, ...] | list[tuple[Real, ...]]:
        r"""
        ............................................................................
        
        ==Retrieve eigenvalues==
        
        This method retrieves the eigenvalues associated with the solution for
        all variants of the model. Eigenvalues are filtered based on stability
        criteria specified by the `kind` parameter.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with valid eigenvalue data.

        ???+ input "kind"
            Specifies the type of eigenvalues to retrieve (e.g., stable, unstable).

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists into tuples. Default is `True`.

        ### Returns ###

        ???+ returns "eigenvalues"
            A tuple of eigenvalues for each variant, or a list of tuples if there
            are multiple variants.

        ### Example ###

        ```python
            eigenvalues = model.get_eigenvalues(kind=EigenvalueKind.STABLE)
            print(eigenvalues)
        ```
        
        ............................................................................
        """
        return [
            tuple(e for e, s in zip(v.solution.eigenvalues, v.solution.eigenvalues_stability, ) if s in kind)
            for v in self._variants
        ]

    @_unpack_singleton
    def get_eigenvalues_stability(
        self,
        *,
        kind: EigenvalueKind = EigenvalueKind.ALL,
        unpack_singleton: bool = True,
    ):
        r"""
        ............................................................................
        
        ==Retrieve eigenvalue stability==
        
        This method retrieves the stability status of eigenvalues for all variants
        of the model. Stability is determined based on the `kind` parameter.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing eigenvalue stability data.

        ???+ input "kind"
            Specifies the type of stability to retrieve (e.g., stable, unstable).

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists into tuples. Default is `True`.

        ### Returns ###

        ???+ returns "eigenvalue_stability"
            A tuple or list of tuples representing eigenvalue stability states.

        ### Example ###

        ```python
            eigen_stability = model.get_eigenvalues_stability(kind=EigenvalueKind.STABLE)
            print(eigen_stability)
        ```
        
        ............................................................................
        """
        return [ 
            tuple(i for i in v.solution.eigenvalues_stability if i in kind)
            for v in self._variants
        ]

    def _get_variable_stability_for_variant(self, variant, /, ) -> dict[str, bool]:
        r"""
        ............................................................................
        
        ==Retrieve variable stability for a variant==
        
        This internal method determines the stability of transition and measurement
        variables for a given model variant.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object containing variant data.

        ???+ input "variant"
            A model variant object for which stability is calculated.

        ### Returns ###

        ???+ returns "variable_stability"
            A dictionary where keys are variable names and values are booleans,
            indicating stability (`True` for stable, `False` otherwise).

        ### Example ###

        ```python
            stability = model._get_variable_stability_for_variant(variant)
            print(stability)
        ```
        
        ............................................................................
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
        *,
        unpack_singleton: bool = True,
    ) -> dict[str, bool] | list[dict[str, bool]]:
        r"""
        ............................................................................
        
        ==Retrieve variable stability==
        
        This method retrieves the stability status of all variables in the model,
        organized by variants.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with variable stability data.

        ???+ input "unpack_singleton"
            If `True`, unpack singleton lists into dictionaries. Default is `True`.

        ### Returns ###

        ???+ returns "variable_stability"
            A dictionary or list of dictionaries where keys are variable names
            and values are booleans, indicating stability.

        ### Example ###

        ```python
            stability = model.get_variable_stability(unpack_singleton=True)
            print(stability)
        ```
        
        ............................................................................
        """
        return [
            self._get_variable_stability_for_variant(v, )
            for v in self._variants
        ]

    def generate_minus_control_quantities(self, /, ) -> tuple[int]:
         r"""
        ............................................................................
        
        ==Generate minus-control quantities==
        
        This method generates identifiers for quantities requiring minus-control
        operations. Minus-control quantities are typically derived variables or
        shocks adjusted by control mechanisms.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with quantity definitions.

        ### Returns ###

        ???+ returns "minus_control_quantities"
            A tuple of identifiers for quantities requiring minus-control.

        ### Example ###

        ```python
            minus_quantities = model.generate_minus_control_quantities()
            print(minus_quantities)
        ```
        
        ............................................................................
        """

    def map_name_to_minus_control_func(self, /, ) -> tuple[str]:
        r"""
        ............................................................................
        
        ==Map names to minus-control functions==
        
        This method creates a mapping from quantity names to their respective
        minus-control functions. Functions vary depending on whether quantities
        are logged or linear.

        ### Input arguments ###

        ???+ input "self"
            `Simultaneous` model object with quantity metadata.

        ### Returns ###

        ???+ returns "name_to_minus_control_func"
            A mapping from quantity names to minus-control functions.

        ### Example ###

        ```python
            minus_control_map = model.map_name_to_minus_control_func()
            print(minus_control_map)
        ```
        
        ............................................................................
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


def _resolve_steady_kind(
    *,
    include_shocks: bool = False,
    **kwargs,
) -> _quantities.QuantityKind:
    r"""
    ............................................................................
    
    ==Resolve the steady-state kind for quantities==
    
    This function determines the appropriate kind of quantities to include in
    steady-state calculations. By default, it excludes shocks unless explicitly
    instructed via the `include_shocks` parameter.

    ### Input arguments ###

    ???+ input "include_shocks"
        A boolean indicating whether to include shock variables in the kind
        resolution. Defaults to `False`.

    ???+ input "**kwargs"
        Additional arguments that may influence the kind resolution.

    ### Returns ###

    ???+ returns "quantity_kind"
        A `QuantityKind` object that represents the resolved kind of quantities
        to include.

    ### Example ###

    ```python
        steady_kind = _resolve_steady_kind(include_shocks=True)
        print(steady_kind)
    ```
    
    ............................................................................
    """
    #[
    return (
        _sources.LOGGABLE_VARIABLE if not include_shocks
        else _sources.LOGGABLE_VARIABLE_OR_ANY_SHOCK
    )
    #]

