"""
"""

#[
from __future__ import annotations

from ..databanks import main as _databanks
from ..plans import main as _plans
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
    ) -> _databanks.Databank:
        """
        """
        ds = _dataslabs.Dataslab.from_databank_for_simulation(
            self, in_databank, base_range, column=0,
        )
        #
        ds.fill_missing_in_base_columns(self.res_names, )
        #
        columns_to_simulate = ds.base_columns
        new_data = ds.copy_data()
        new_data = self._simulate(new_data, columns_to_simulate, )
        ds.data = new_data
        #
        ds.remove_terminal()
        out_db = ds.to_databank()
        if prepend_input:
            out_db.prepend(in_databank, ds.column_dates[0]-1, )
        return out_db

    def _simulate(
        self,
        data: _np.ndarray,
        columns: Iterable[int],
        /,
    ) -> _dataslabs.Dataslab:
        """
        """
        for t in columns:
            for x in self.explanatories:
                x.assign_level(data, t, )
        return data

    #]

