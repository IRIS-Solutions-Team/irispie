
#[
from __future__ import annotations
# from IPython import embed

import numpy as np_
from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )

from . import databanks as db_
from . import dates as da_
from . import series as se_
#]


class Dataslab:
    """
    """
    #[
    data: np_.ndarray | None = None
    row_names: Iterable[str] | None = None,
    missing_names: Iterable[str] | None = None,
    column_dates: Iterable[Dater] | None = None

    def to_databank(self, *args, **kwargs, ) -> da_.Databank:
        """
        """
        return multiple_to_databank((self,), *args, **kwargs, )

    def add_from_databank(
        self,
        in_databank: db_.Databank,
        names: Iterable[str],
        ext_range: Ranger,
        /,
        column: int = 0,
    ) -> Self:
        """
        Add data from a databank to this dataslab
        """
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
        """
        """
        if remove > 0:
            self.data = self.data[:, remove:]
            self.column_dates = self.column_dates[remove:]
        elif remove < 0:
            self.data = self.data[:, :remove]
            self.column_dates = self.column_dates[:remove]

    @classmethod
    def from_databank(
        cls,
        *args,
        **kwargs,
    ) -> Self:
        """
        Create a new dataslab from a databank
        """
        self = cls();
        self.add_from_databank(*args, **kwargs)
        return self
    #]


def _extract_data_from_record(record, ext_range, column, /, ):
    """
    """
    return (
        record.get_data_column(ext_range, column).reshape(1, -1) 
        if hasattr(record, "get_data")
        else np_.full((1, len(ext_range)), float(record), dtype=float)
    )


def multiple_to_databank(
    selves,
    /,
    target_databank: db_.Databank | None = None,
) -> db_.Databank:
    """
    Add data from a dataslab to a new or existing databank
    """
    out_databank = target_databank if target_databank is not None else db_.Databank()
    self = selves[0]
    num_columns = len(selves)
    for row, n in enumerate(self.row_names):
        data = np_.hstack(tuple(ds.data[(row,), :].T for ds in selves))
        x = se_.Series(num_columns=num_columns)
        x.set_data(self.column_dates, data)
        setattr(out_databank, n, x)
    return out_databank

