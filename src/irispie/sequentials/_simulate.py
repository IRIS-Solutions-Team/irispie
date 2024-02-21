"""
"""


#[
from __future__ import annotations

from typing import Any
from numbers import Real
import numpy as _np
import itertools as _it
import wlogging as _wl

from .. import pages as _pages
from .. import wrongdoings as _wrongdoings
from ..databoxes import main as _databoxes
from ..plans import main as _plans
from ..explanatories import main as _explanatories
from ..dataslates import main as _dataslates
#]


_LOGGER_NAME = "Sequential.simulate"
_dataslate_constructor = _dataslates.Dataslate.from_databox_for_slatable


class Inlay:
    """
    """
    #[

    @_pages.reference(category="simulation", )
    def simulate(
        self,
        in_databox: _databoxes.Databox,
        span: Iterable[Dater],
        /,
        plan: _plans.Plan | None = None,
        prepend_input: bool = True,
        target_databox: _databoxes.Databox | None = None,
        when_nonfinite: Literal["error", "warning", "silent", ] = "warning",
        execution_order: Literal["dates_equations", "equations_dates", ] = "dates_equations",
        num_variants: int | None = None,
        remove_initial: bool = True,
        remove_terminal: bool = True,
        logging_level: int = _wl.INFO,
    ) -> tuple[_databoxes.Databox, dict[str, Any]]:
        """
......................................................................

==Simulate sequential model==

```
output_db, info = self.simulate(
    input_db, span,
    /,
    prepend_input=True,
    plan=None,
    target_databox=None,
    when_nonfinite="warning",
    num_variants=None,
    remove_initial=True,
    remove_terminal=True,
)
```

Simulate a `Sequential` model, `self`, on a time `span`, period by period,
equation by equation. The `simulate` function does not reorder the
equations; if needed, this must be done by running `reorder` before
simulating the model.


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


### Returns ###

???+ returns "output_db"

    Output databox with the simulated time series for the LHS variables.

???+ returns "info"

    Information about the simulation; `info` is a dict with the following
    items.

......................................................................
        """
        logger = _wl.get_colored_logger(_LOGGER_NAME, level=logging_level, )
        num_variants = self.num_variants if num_variants is None else num_variants
        base_dates = tuple(span, )
        extra_databox_names = None
        if plan is not None:
            plan.check_consistency(self, base_dates, )
            extra_databox_names = plan.get_databox_names()
        #
        dataslate = _dataslate_constructor(
            self, in_databox, base_dates,
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
        for vid, model_v, dataslate_v in zipped:
            model_v._simulate_v(
                dataslate_v,
                logger,
                plan=plan,
                when_nonfinite=when_nonfinite,
                execution_order=execution_order,
            )
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
            out_db.prepend(in_databox, base_dates[0]-1, )
        #
        # Add to custom databox
        if target_databox is not None:
            out_db = target_databox | out_db
        #
        info = {}
        #
        return out_db, info

    def _simulate_v(
        self,
        ds: _dataslates.Dataslate,
        logger: _wl.Logger,
        /,
        *,
        plan,
        when_nonfinite,
        execution_order,
    ) -> None:
        """
        """
        when_nonfinite_stream = \
            _wrongdoings.STREAM_FACTORY[when_nonfinite] \
            ("These simulated data point(s) are non-finite values:", )
        #
        name_to_row = ds.create_name_to_row()
        base_columns = ds.base_columns
        first_base_column = base_columns[0] if base_columns else None
        working_data = ds.get_data_variant(0, )
        columns_dates = ( (c, ds.dates[c]) for c in base_columns )
        execution_iterator_creator = _CREATE_EXECUTION_ITERATOR[execution_order]
        columns_dates_equations = execution_iterator_creator(columns_dates, self.iter_equations(), )
        for (column, date), equation in columns_dates_equations:
            transform = _get_transform(plan, equation, date)
            is_exogenized, implied_value = _is_exogenized(
                equation.lhs_name,
                transform,
                working_data,
                column,
                name_to_row,
            )
            #
            simulation_function = equation.simulate if not is_exogenized else equation.exogenize
            info = simulation_function(working_data, column, implied_value, )
            logger.debug(
                f"{simulation_function.__name__}"
                f" {equation.lhs_name}[{date}]={info['lhs_value']}"
                f" (equation.residual_name[{date}]={info['residual_value']})"
            )
            #
            _catch_nonfinite(
                when_nonfinite_stream,
                info["is_finite"],
                info["simulated_name"],
                date,
            )
        when_nonfinite_stream._raise()

    #]


def _get_transform(
    plan: _plans.Plan | None,
    equation: _explanatories.Explanatory,
    date: _dates.Dater,
) -> _plans.Transform | None:
    """
    """
    return (
        plan.get_exogenized_point(equation.lhs_name, date, )
        if plan and not equation.is_identity
        else None
    )


def _is_exogenized(
    lhs_name: str,
    transform: _plans.Transform | None,
    data: _np.ndarray,
    data_column: int,
    name_to_row: dict[str, int],
) -> tuple[bool, Real | None]:
    """
    """
    #[
    if transform is None:
        return False, None
    #
    lhs_name_row = name_to_row[lhs_name]
    values_before = data[lhs_name_row, :data_column]
    #
    transform_name = transform.resolve_databox_name(lhs_name, )
    transform_row = name_to_row.get(transform_name, None, )
    transform_values_after = data[transform_row, data_column:] if transform_row is not None else None
    implied_value = transform.eval_exogenized(transform_values_after, values_before, )
    if transform.when_data and _np.isnan(implied_value):
        return False, None
    return True, implied_value
    #]


def _catch_nonfinite(
    stream: _wrongdoings.Stream,
    is_finite: bool | _np.ndarray,
    simulated_name: str,
    date: Dater,
) -> None:
    """
    """
    #[
    if is_finite.all():
        return
    message = f"{simulated_name}[{date}]"
    stream.add(message, )
    #]


def _iter_dates_equations(columns_dates, equations, ) -> Iterator[tuple[int, _dates.Dater], _explanatories.Explanatory]:
    return _it.product(columns_dates, equations, )


def _iter_equations_dates(columns_dates, equations, ) -> Iterator[tuple[int, _dates.Dater], _explanatories.Explanatory]:
    return _swap_product(_it.product(equations, columns_dates, ), )


def _swap_product(iterator: Iterable[tuple[Any, Any]], ) -> Iterable[tuple[Any, Any]]:
    return ((b, a) for a, b in iterator)


_CREATE_EXECUTION_ITERATOR = {
    "dates_equations": _iter_dates_equations,
    "equations_dates": _iter_equations_dates,
}


