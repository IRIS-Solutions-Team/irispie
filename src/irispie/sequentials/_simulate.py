"""
"""


#[
from __future__ import annotations

from typing import (Any, )
from numbers import (Number, )
import numpy as _np

from .. import wrongdoings as _wrongdoings
from ..databoxes import main as _databoxes
from ..plans import main as _plans
from ..explanatories import main as _explanatories
from .. import dataslates as _dataslates
#]


class SimulateMixin:
    """
    """
    #[

    def simulate(
        self,
        in_databox: _databoxes.Databox,
        base_range: Iterable[Dater],
        /,
        plan: _plans.Plan | None = None,
        prepend_input: bool = True,
        add_to_databox: _databoxes.Databox | None = None,
        when_nonfinite: Literal["error", "warning", "silent", ] = "warning",
    ) -> tuple[_databoxes.Databox, dict[str, Any]]:
        """
        """
        #
        # Arrange input data into a dataslate
        #
        ds = _dataslates.Dataslate(
            self, in_databox, base_range,
            slate=0, plan=plan,
        )
        #
        # Fill missing residuals with zeros
        #
        ds.fill_missing_in_base_columns(self.res_names, fill=0, )
        #
        # Run simulation period by period, equation by equation
        #
        columns_to_simulate = ds.base_columns
        self._simulate_periods_and_explanatories(
            ds,
            columns_to_simulate,
            plan,
            when_nonfinite=when_nonfinite,
        )
        #
        # Build output databox
        #
        ds.remove_terminal()
        out_db = ds.to_databox()
        if prepend_input:
            out_db.prepend(in_databox, ds.column_dates[0]-1, )
        #
        # Add to custom databox
        #
        if add_to_databox is not None:
            out_db = add_to_databox | out_db
        #
        info = {"dataslate": ds, }
        return out_db, info

    def _simulate_periods_and_explanatories(
        self,
        ds: _dataslates.Dataslate,
        columns: Iterable[int],
        plan: _plans.Plan | None,
        /,
        when_nonfinite: Literal["error", "warning", "silent", ] = "warning",
    ) -> None:
        """
        """
        when_nonfinite_stream = \
            _wrongdoings.STREAM_FACTORY[when_nonfinite] \
            ("Simulating the following data point(s) resulted in non-finite values:", )
        #
        name_to_row = ds.create_name_to_row()
        first_base_column = ds.base_columns[0] if ds.base_columns else None
        #
        for data_column in columns:
            plan_column = data_column - first_base_column
            date = ds.column_dates[data_column]
            #
            for x in self.explanatories:
                is_exogenized, implied_value = _is_exogenized(x, ds.data, plan, data_column, plan_column, name_to_row, )
                if is_exogenized:
                    info = x.exogenize(ds.data, data_column, implied_value)
                else:
                    info = x.simulate(ds.data, data_column, )
                _catch_nonfinite(when_nonfinite_stream, info["is_finite"], x.lhs_name, date, )
                #
        when_nonfinite_stream.throw()

    #]


def _is_exogenized(
    explanatory: _explanatories.Explanatory,
    data: _np.ndarray,
    plan: _plans.Plan | None,
    data_column: int,
    plan_column: int,
    name_to_row: dict[str, int],
) -> tuple[bool, Number | None]:
    """
    """
    #[
    if plan is None or explanatory.is_identity:
        return False, None
    #
    lhs_name = explanatory.lhs_name
    transform = plan.get_exogenized_point(lhs_name, plan_column)
    if transform is None:
        return False, None
    #
    transform_name = transform.resolve_databox_name(lhs_name, )
    lhs_name_row = name_to_row[lhs_name]
    transform_row = name_to_row[transform_name]
    transform_value = data[transform_row, data_column]
    lagged_value = data[lhs_name_row, data_column-1]
    implied_value = transform.eval_exogenized(transform_value, lagged_value, )
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

