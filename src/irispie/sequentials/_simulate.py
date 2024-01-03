"""
"""


#[
from __future__ import annotations

from typing import (Any, )
from numbers import (Real, )
import numpy as _np

from .. import pages as _pages
from .. import wrongdoings as _wrongdoings
from ..databoxes import main as _databoxes
from ..plans import main as _plans
from ..explanatories import main as _explanatories
from ..dataslates import main as _dataslates
#]


_dataslate_constructor = _dataslates.Dataslate.from_databox_for_slatable


class SimulateInlay:
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
        num_variants: int | None = None,
        remove_initial: bool = True,
        remove_terminal: bool = True,
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
    which LHS variables to exogenize in which periods. If `plan=None`, no
    simulation plan is imposed on the simulation.


### Returns ###

???+ returns "output_db"

    Output databox with the simulated time series for the LHS variables.

???+ returns "info"

    Information about the simulation; `info` is a dict with the following
    items.

......................................................................
        """
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
            model_v._simulate_periods_and_explanatories(
                dataslate_v,
                plan=plan,
                when_nonfinite=when_nonfinite,
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

    def _simulate_periods_and_explanatories(
        self,
        ds: _dataslates.Dataslate,
        /,
        plan: _plans.Plan | None = None,
        when_nonfinite: Literal["error", "warning", "silent", ] = "warning",
    ) -> None:
        """
        """
        when_nonfinite_stream = \
            _wrongdoings.STREAM_FACTORY[when_nonfinite] \
            ("Simulating the following data point(s) resulted in non-finite values:", )
        #
        name_to_row = ds.create_name_to_row()
        base_columns = ds.base_columns
        first_base_column = (
            base_columns[0]
            if base_columns
            else None
        )
        #
        working_data = ds.get_data_variant(0, )
        for data_column in base_columns:
            plan_column = data_column - first_base_column
            date = ds.dates[data_column]
            #
            for x in self.iter_explanatories():
                #
                transform = (
                    _get_transform(plan, x.lhs_name, plan_column, )
                    if not x.is_identity
                    else None
                )
                #
                is_exogenized, implied_value = _is_exogenized(x.lhs_name, transform, working_data, data_column, name_to_row, )
                if is_exogenized:
                    info = x.exogenize(working_data, data_column, implied_value)
                else:
                    info = x.simulate(working_data, data_column, )
                _catch_nonfinite(when_nonfinite_stream, info["is_finite"], x.lhs_name, date, )
                #
        when_nonfinite_stream._raise()

    #]


def _get_transform(
    plan: _plans.Plan | None,
    lhs_name: str,
    plan_column: int,
) -> _plans.Transform | None:
    """
    """
    return (
        plan.get_exogenized_point(lhs_name, plan_column, )
        if plan is not None else None
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
    name: str,
    date: Dater,
) -> None:
    """
    """
    #[
    if _np.all(is_finite, ):
        return
    message = f"{name}[{date}]"
    stream.add(message, )
    #]

