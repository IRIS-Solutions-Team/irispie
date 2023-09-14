"""
Exporting data to CSV sheets
"""


#[
from __future__ import annotations

import pickle as _pickle

from collections.abc import (Iterable, )
import csv as _cs
import numpy as _np
import dataclasses as _dc
import itertools as _it

from .. import dates as _dates
#]


class NotImplementedYet(Exception):
    pass


@_dc.dataclass
class _ExportBlockDescriptor:
    """
    """
    #[
    frequency: _dates.Frequency | None = None
    names: Iterable[str] | None = None
    descriptors: Iterable[str] | None = None
    descriptor_row: bool | None = None
    num_series_columns: int | None = None
    dates: Iterable[_dates.Dater] | None = None
    data_array: _np.ndarray | None = None
    delimiter: str | None = None
    numeric_format: str | None = None
    nan_str: str | None = None

    def row_iterator(self):
        """
        """
        names = list(_it.chain.from_iterable( 
            [n] + ["*"]*(self.num_series_columns[i] - 1)
            for i, n in enumerate(self.names)
        ))
        yield [_get_frequency_mark(self.frequency)] + names
        if self.descriptor_row:
            descriptors = list(_it.chain.from_iterable( 
                [n] + ["*"]*(self.num_series_columns[i] - 1)
                for i, n in enumerate(self.descriptors)
            ))
            yield [""] + descriptors
        for date, data_row in zip(self.dates, self.data_array):
            yield [date] + [ x if not _np.isnan(x) else self.nan_str for x in data_row.tolist() ]
    #]


class DatabankExportMixin:
    """
    Databank mixin for exporting data to CSV sheets
    """
    #[
    def to_sheet(
        self,
        file_name: str,
        *,
        descriptor_row: bool = False,
        range: Iterable[_dates.Dater] | None = None,
        frequency: _dates.Frequency | None = None,
        delimiter: str = ",",
        numeric_format: str = "g",
        nan_str: str = "",
        csv_writer_settings: dict = {},
    ) -> Iterable[str]:
        """
        """
        if not frequency:
            raise NotImplementedYet("frequency=None")
        frequency = _dates.Frequency(frequency)
        range = range if range else self._get_range_by_frequency(frequency)
        range = [ t for t in range ]
        num_data_rows = len(range)
        #
        names = self._get_series_names_by_frequency(frequency)
        descriptors = [ self[n]._description for n in names ]
        #
        num_series_columns = [ self[n].shape[1] for n in names ]
        data_array = _np.hstack([ self[n].get_data(range) for n in names ])
        export_block = _ExportBlockDescriptor(
            frequency, names, descriptors, descriptor_row,
            num_series_columns, range, data_array, 
            delimiter, numeric_format, nan_str,
        )
        with open(file_name, "w+") as fid:
            writer = _cs.writer(fid, delimiter=delimiter, **csv_writer_settings)
            for row in export_block.row_iterator():
                writer.writerow(row)
        #
        return names

    def to_pickle(
        self,
        file_name: str,
        /,
        **kwargs,
    ) -> None:
        """
        """
        with open(file_name, "wb+") as fid:
            _pickle.dump(self, fid, **kwargs, )
    #]


def _get_frequency_mark(frequency, ):
    return "__" + frequency.name.lower() + "__"


