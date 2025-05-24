"""
"""


#[

from __future__ import annotations

from typing import (Self, Any, )
from numbers import (Real, )
import numpy as _np
import itertools as _it
import functools as _ft
import warnings as _wa
import documark as _dm

from .. import wrongdoings as _wrongdoings
from .. import has_variants as _has_variants
from ..databoxes.main import Databox
from ..plans.simulation_plans import SimulationPlan
from ..plans.transforms import PlanTransform
from ..explanatories import main as _explanatories
from ..dataslates.main import Dataslate

#]


class Inlay:
    """
    """
    #[

    @_dm.reference(category="simulation", )
    def simulate(
        self,
        input_db: Databox,
        span: Iterable[Dater],
        *,
        plan: SimulationPlan | None = None,
        prepend_input: bool = True,
        target_db: Databox | None = None,
        when_nonfinite = None,
        when_simulates_nan: Literal["error", "warning", "silent", ] = "warning",
        execution_order: Literal["dates_equations", "equations_dates", ] = "dates_equations",
        num_variants: int | None = None,
        remove_initial: bool = True,
        remove_terminal: bool = True,
        shocks_from_data: bool = True,
        parameters_from_data: bool = False,
        catch_warnings: bool = False,
        unpack_singleton: bool = True,
        return_info: bool = False,
        method: Literal["sequential", ] = "sequential",
    ) -> tuple[Databox, dict[str, Any]]:
        """
················································································

==Simulate sequential model==

Simulate a `Sequential` model, `self`, on a time `span`, period by period,
equation by equation. The `simulate` function does not reorder the
equations; if needed, this must be done by running `reorder` before
simulating the model.


```
out_db = self.simulate(
    input_db,
    simulation_span,
    *,
    plan=None,
    execution_order="dates_equations",
    prepend_input=True,
    target_db=None,
    when_simulates_nan="warning",
    num_variants=None,
    remove_initial=True,
    remove_terminal=True,
    return_info=False,
)
```

```
out_db, info = self.simulate(
    ...,
    return_info=True,
    ...,
)
```


### Input arguments ###


???+ input "self"
    `Sequential` model that will be simulated.

???+ input "input_db"
    Input databox (a `Databox` object) with all the necessary initial
    conditions (initial lags) for the LHS variables, and all the values for
    the RHS variables needed to simulate `self` on the time `span`.

???+ input "plan"
    `PlanSimulate` object with a simulation plan, i.e. information about
    which LHS variables to exogenize at which dates. If `plan=None`, no
    simulation plan is imposed on the simulation.

???+ input "simulation_span"
    [Time span](../data_management/spans.md) for the simulation; the time
    span needs to go forward and have a one-period step.

???+ input "plan"
    [Simulation plan](plans.md) for the simulation specifying the
    exogenized data points. If `None`, no simulation plan is imposed.

???+ input "prepend_input"
    If `True`, the input time series observations are prepended to the results.

???+ input "target_db"
    Custom databox to which the simulated time series will be added. If
    `None`, a new databox is created.

???+ input "when_simulates_nan"
    Action to take when a simulated data point is non-finite (`nan` or `inf` or `-inf`). The options are

    * `"error"`: raise an error,
    * `"warning"`: log a warning,
    * `"silent"`: do nothing.

???+ input "execution_order"
    Order in which the model equations and simulation periods are executed. The options are

    * `"dates_equations"`: all equations for the first period, all equations for the second period, …
    * `"equations_dates"`: all periods for the first equation, all periods for the second equation, …

???+ input "num_variants"
    Number of variants to simulate. If `None`, the number of variants is
    determined by the number of variants in the `self` model.

???+ input "remove_initial"
    If `True`, remove the initial condition data, i.e. all lags before the
    start of the simulation span.

???+ input "remove_terminal"
    If `True`, remove the terminal condition data, i.e. all leads after the
    end of the simulation span.


### Returns ###


???+ returns "out_db"
    Output databox with the simulated time series for the LHS variables.

???+ returns "info"
    (Only returned if `return_info=True which is not the default behavior)
    Dictionary with information about the simulation; `info` contains the
    following items:

    | Key | Description
    |-----|-------------
    | `"method"` | Simulation method used
    | `"execution_order"` | Execution order of the equations and periods


················································································
        """

        # Legacy
        if when_nonfinite is not None:
            when_simulates_nan = when_nonfinite

        num_variants \
            = self.resolve_num_variants_in_context(num_variants, )

        base_dates = tuple(span, )

        extra_databox_names = None
        if plan is not None:
            plan.check_consistency(self, base_dates, )
            extra_databox_names = plan.get_databox_names()
        #
        slatable = self.slatable_for_simulate(
            shocks_from_data=shocks_from_data,
            parameters_from_data=parameters_from_data,
        )
        dataslate = Dataslate.from_databox_for_slatable(
            slatable, input_db, base_dates,
            num_variants=self.num_variants,
            extra_databox_names=extra_databox_names,
        )
        #
        zipped = zip(
            range(num_variants, ),
            self.iter_variants(),
            dataslate.iter_variants(),
        )
        #
        #=======================================================================
        # Main loop over variants
        out_info = []
        simulate_method = _SIMULATION_METHOD_DISPATCH[method]
        for vid, model_v, dataslate_v in zipped:
            info_v = simulate_method(
                model_v, dataslate_v, plan, vid,
                when_simulates_nan=when_simulates_nan,
                execution_order=execution_order,
                catch_warnings=catch_warnings,
            )
            out_info.append(info_v, )
        #=======================================================================
        #
        # Remove initial and terminal condition data (all lags and leads
        # before and after the simulation span)
        if remove_terminal:
            dataslate.remove_terminal()
        if remove_initial:
            dataslate.remove_initial()
        #
        # Convert all variants of the dataslate to a databox
        out_db = dataslate.to_databox()
        if prepend_input:
            out_db.prepend(input_db, base_dates[0]-1, )
        #
        # Add to custom databox
        if target_db is not None:
            out_db = target_db | out_db
        #
        if return_info:
            out_info = _has_variants.unpack_singleton(
                out_info, self.is_singleton,
                unpack_singleton=unpack_singleton,
            )
            return out_db, out_info
        else:
            return out_db

    #]


def _simulate_v(
    self,
    ds: Dataslate,
    plan: SimulationPlan | None,
    vid: int,
    *,
    when_simulates_nan,
    execution_order,
    catch_warnings,
) -> dict[str, Any]:
    """
    """
    when_nonfinite_stream = \
        _wrongdoings.STREAM_FACTORY[when_simulates_nan] \
        ("These simulated data point(s) are nan or inf:", )
    #
    name_to_row = ds.create_name_to_row()
    base_columns = ds.base_columns
    first_base_column = base_columns[0] if base_columns else None
    working_data = ds.get_data_variant(0, )
    columns_dates = ( (c, ds.periods[c]) for c in base_columns )
    iterator_creator = _CREATE_EXECUTION_ITERATOR[execution_order]
    #
    columns_dates_equations = iterator_creator(
        columns_dates,
        self.iter_equations(),
    )
    #
    detect_exogenized = _ft.partial(
        _detect_exogenized,
        name_to_row=name_to_row,
        data=working_data,
    )
    #
    info = {
        "method": "sequential",
        "execution_order": execution_order,
    }
    #
    if catch_warnings:
        _wa.simplefilter("error", )
    for (column, date), equation in columns_dates_equations:
        lhs_date_str = f"{equation.lhs_name}[{date}]"
        residual_date_str = f"{equation.residual_name}[{date}]"
        transform = _get_transform(plan, equation, date)
        try:
            implied_value = detect_exogenized(equation.lhs_name, transform, column, )
            simulation_func = (
                equation.simulate
                if implied_value is None
                else equation.exogenize
            )
            info_eq = simulation_func(working_data, column, implied_value, )
        except Exception as exc:
            message = (
                f"Error when simulating {lhs_date_str}"
                f"\nDirect cause: {str(exc)}"
            )
            raise _wrongdoings.IrisPieCritical(message, ) from exc
        #
        _catch_nonfinite(
            when_nonfinite_stream,
            info_eq["is_finite"],
            info_eq["simulated_name"],
            info_eq["simulated_value"],
            date,
        )
    when_nonfinite_stream._raise()
    _wa.simplefilter("default", )
    #
    return info


def _get_transform(
    plan: SimulationPlan | None,
    equation: _explanatories.Explanatory,
    date: _dates.Dater,
) -> PlanTransform | None:
    """
    """
    return (
        plan.get_exogenized_point(equation.lhs_name, date, )
        if plan and not equation.is_identity
        else None
    )


def _detect_exogenized(
    lhs_name: str,
    transform: PlanTransform | None,
    data_column: int,
    *,
    data: _np.ndarray,
    name_to_row: dict[str, int],
) -> Real | None:
    """
    """
    #[
    if transform is None:
        return None
    #
    lhs_name_row = name_to_row[lhs_name]
    values_before = data[lhs_name_row, :data_column]
    values_after_inclusive = data[lhs_name_row, data_column:]
    #
    transform_name = transform.resolve_databox_name(lhs_name, )
    transform_row = name_to_row.get(transform_name, None, )
    transform_values_after = (
        data[transform_row, data_column:]
        if transform_row is not None else None
    )
    implied_value = transform.eval_exogenized(transform_values_after, values_before, values_after_inclusive, )
    if transform.when_data and _np.isnan(implied_value):
        return None
    return implied_value
    #]


def _catch_nonfinite(
    stream: _wrongdoings.Stream,
    is_finite: _np.ndarray,
    simulated_name: str,
    simulated_value: Real | _np.ndarray,
    date: Dater,
) -> None:
    """
    """
    #[
    if is_finite.all():
        return
    simulate_value = (
        simulated_value.tolist()
        if isinstance(simulated_value, _np.ndarray)
        else simulated_value
    )
    message = f"{simulated_name}[{date}]={simulated_value}"
    stream.add(message, )
    #]


def _iter_dates_equations(columns_dates, equations, ) -> Iterator[tuple[int, _dates.Dater], _explanatories.Explanatory]:
    return _it.product(columns_dates, equations, )


def _iter_equations_dates(columns_dates, equations, ) -> Iterator[tuple[int, _dates.Dater], _explanatories.Explanatory]:
    return _swap_product(_it.product(equations, columns_dates, ), )


def _swap_product(iterator: Iterable[tuple[Any, Any]], ) -> Iterable[tuple[Any, Any]]:
    return ((b, a) for a, b in iterator)


_SIMULATION_METHOD_DISPATCH = {
    "sequential": _simulate_v,
}


_CREATE_EXECUTION_ITERATOR = {
    "dates_equations": _iter_dates_equations,
    "equations_dates": _iter_equations_dates,
}

