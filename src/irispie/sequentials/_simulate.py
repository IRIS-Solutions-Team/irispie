"""
"""

#[
from __future__ import annotations

from typing import (Any, )
from numbers import (Number, )
import numpy as _np

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
        # Run simulation
        #
        columns_to_simulate = ds.base_columns
        name_to_row = ds.create_name_to_row()
        self._simulate(ds.data, columns_to_simulate, name_to_row, plan, )
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

    def _simulate(
        self,
        data: _np.ndarray,
        columns: Iterable[int],
        name_to_row: dict[str, int],
        plan: _plans.Plan | None,
        /,
    ) -> None:
        """
        """
        for plan_column, data_column in enumerate(columns):
            for x in self.explanatories:
                status, implied_value = _is_exogenized(x, data, plan, data_column, plan_column, name_to_row, )
                if status:
                    x.exogenize(data, data_column, implied_value)
                else:
                    x.simulate(data, data_column, )

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

