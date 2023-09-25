"""
"""

#[
from __future__ import annotations

from typing import (Any, )
import numpy as _np

from ..databanks import main as _databanks
from ..plans import main as _plans
from ..explanatories import main as _explanatories
from .. import dataslabs as _dataslabs
#]


class SimulateMixin:
    """
    """
    #[
    def simulate(
        self,
        in_databank: _databanks.Databank,
        base_range: Iterable[Dater],
        /,
        plan: _plans.Plan | None = None,
        prepend_input: bool = True,
        add_to_databank: _databanks.Databank | None = None,
    ) -> tuple[_databanks.Databank, dict[str, Any]]:
        """
        """
        #
        # Arrange input data into a dataslab
        #
        ds = _dataslabs.Dataslab.from_databank_for_simulation(
            self, in_databank, base_range, column=0, plan=plan,
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
        # Build output databank
        #
        ds.remove_terminal()
        out_db = ds.to_databank()
        if prepend_input:
            out_db.prepend(in_databank, ds.column_dates[0]-1, )
        #
        # Add to custom databank
        #
        if add_to_databank is not None:
            out_db = add_to_databank | out_db
        #
        info = {"dataslab": ds, }
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
                status, value = _is_exogenized(x, data, plan, data_column, plan_column, name_to_row, )
                if status:
                    x.exogenize(data, data_column, value)
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
) -> bool:
    if plan is None or explanatory.is_identity:
        return False, None
    #
    lhs_name = explanatory.lhs_name
    transform = plan.exogenized[lhs_name][plan_column]
    if transform is None:
        return False, None
    #
    transform_name = transform.resolve_databank_name(lhs_name, )
    lhs_name_row = name_to_row[lhs_name]
    transform_row = name_to_row[transform_name]
    transform_value = data[transform_row, data_column]
    lagged_value = data[lhs_name_row, data_column-1]
    implied_value = transform.eval_exogenized(transform_value, lagged_value, )
    if transform.when_data and _np.isnan(implied_value):
        return False, None
    return True, implied_value

