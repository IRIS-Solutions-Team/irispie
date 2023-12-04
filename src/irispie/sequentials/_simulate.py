"""
"""


#[
from __future__ import annotations

from typing import (Any, )
from numbers import (Real, )
import numpy as _np

from .. import wrongdoings as _wrongdoings
from ..databoxes import main as _databoxes
from ..plans import main as _plans
from ..explanatories import main as _explanatories
from .. import dataslates as _dataslates
#]


_DATASLATE_VARIANT_ITERATOR = \
    _dataslates.HorizontalDataslate.iter_variants_from_databox_for_slatable


class SimulateMixin:
    """
    """
    #[

    def simulate(
        self,
        in_databox: _databoxes.Databox,
        span: Iterable[Dater],
        /,
        plan: _plans.Plan | None = None,
        prepend_input: bool = True,
        target_databox: _databoxes.Databox | None = None,
        when_nonfinite: Literal["error", "warning", "silent", ] = "warning",
    ) -> tuple[_databoxes.Databox, dict[str, Any]]:
        """
        """
        #
        # Check consistency of the plan and the model/simulation
        # Add extra custom databox names to the model
        base_dates = tuple(span, )
        if plan is not None:
            plan.check_consistency(self, base_dates, )
            self.set_extra_databox_names(plan.get_databox_names(), )
        #
        out_dataslates = []
        num_variants = 1
        zipped = zip(
            range(num_variants, ),
            self.iter_variants(),
            _DATASLATE_VARIANT_ITERATOR(self, in_databox, base_dates, ),
        )
        #
        #=======================================================================
        # Main loop over variants
        for vid, mdi, dsi in zipped:
            mdi._simulate_periods_and_explanatories(
                dsi,
                plan=plan,
                when_nonfinite=when_nonfinite,
            )
            dsi.remove_terminal()
            out_dataslates.append(dsi, )
        #=======================================================================
        #
        # Build output databox
        self.set_extra_databox_names(None, )
        out_db = _dataslates.multiple_to_databox(out_dataslates, )
        #
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
        ds: _dataslates.HorizontalDataslate,
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
        first_base_column = (
            ds.base_columns[0]
            if ds.base_columns
            else None
        )
        #
        for data_column in ds.base_columns:
            plan_column = data_column - first_base_column
            date = ds.column_dates[data_column]
            #
            for x in self.iter_explanatories():
                #
                transform = (
                    _get_transform(plan, x.lhs_name, plan_column, )
                    if not x.is_identity
                    else None
                )
                #
                is_exogenized, implied_value = _is_exogenized(x.lhs_name, transform, ds.data, data_column, name_to_row, )
                if is_exogenized:
                    info = x.exogenize(ds.data, data_column, implied_value)
                else:
                    info = x.simulate(ds.data, data_column, )
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

