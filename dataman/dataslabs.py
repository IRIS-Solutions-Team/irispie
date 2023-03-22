
#[
from __future__ import annotations

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

    def to_datbank(
        self,
        /,
    ) -> Self:
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
            getattr(in_databank, n).get_data(ext_range).reshape(1, -1)
            if n not in missing_names else nan_row
            for n in  names
        )
        #
        self.data = np_.vstack(generate_data)
        self.row_names = tuple(names)
        self.column_dates = tuple(ext_range)
        self.missing_names = missing_names
        return self


