
#[
from __future__ import annotations
from IPython import embed

import numpy as np_
from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )

from . import databanks as db_
from . import dates as da_
from . import series as se_
#]


class Dataslab:
    data: np_.ndarray | None = None
    row_names: Iterable[str] | None = None,
    missing_names: Iterable[str] | None = None,
    column_dates: Iterable[Dater] | None = None

    def to_databank(self, /, ) -> da_.Databank:
        """
        """
        out_databank = db_.Databank()
        for row, n in enumerate(self.row_names):
            x = se_.Series()
            x.set_data(self.column_dates, self.data[(row,), :].T)
            setattr(out_databank, n, x)
        return out_databank

    @classmethod
    def from_databank(
        cls,
        in_databank: db_.Databank,
        names: Iterable[str],
        ext_range: Ranger,
        /,
        column: int = 0,
    ) -> Self:
        """
        """
        self = cls()
        #
        missing_names = [
            n for n in names
            if not hasattr(in_databank, n)
        ]
        #
        num_periods = len(ext_range)
        nan_row = np_.full((1, num_periods), np_.nan, dtype=float)
        generate_data = tuple(
            _extract_data_from_record(getattr(in_databank, n), ext_range, column)
            if n not in missing_names else nan_row
            for n in names
        )
        #
        self.data = np_.vstack(generate_data)
        self.row_names = tuple(names)
        self.column_dates = tuple(ext_range)
        self.missing_names = missing_names
        return self

    def remove_columns(
        self,
        remove: int,
        /,
    ) -> NoReturn:
        if remove > 0:
            self.data = self.data[:, remove:]
            self.column_dates = self.column_dates[remove:]
        elif remove < 0:
            self.data = self.data[:, :remove]
            self.column_dates = self.column_dates[:remove]



def _extract_data_from_record(record, ext_range, column, /, ):
    """
    """
    return (
        record.get_data_column(ext_range, column).reshape(1, -1) 
        if hasattr(record, "get_data")
        else np_.full((1, len(ext_range)), float(record), dtype=float)
    )

