"""
Exporting data to CSV sheets
"""


#[
from __future__ import annotations
# from IPython import embed

import csv as cs_
import numpy as np_
import dataclasses as dc_
import itertools as it_

from ..dataman import dates as da_
from ..dataman import series as se_
#]


class NotImplementedYet(Exception):
    pass


@dc_.dataclass
class _ExportBlockDescriptor:
    """
    """
    #[
    frequency: da_.Frequency | None = None
    names: Iterable[str] | None = None
    descriptors: Iterable[str] | None = None
    descriptor_row: bool | None = None
    num_series_columns: int | None = None
    dates: Iterable[Dater] | None = None
    data_array: np_.ndarray | None = None
    delimiter: str | None = None
    numeric_format: str | None = None
    nan_str: str | None = None

    def row_iterator(self):
        """
        """
        names = list(it_.chain.from_iterable( 
            [n] + ["*"]*(self.num_series_columns[i] - 1)
            for i, n in enumerate(self.names)
        ))
        yield [_get_frequency_mark(self.frequency)] + names
        if self.descriptor_row:
            descriptors = list(it_.chain.from_iterable( 
                [n] + ["*"]*(self.num_series_columns[i] - 1)
                for i, n in enumerate(self.descriptors)
            ))
            yield [""] + descriptors
        for date, data_row in zip(self.dates, self.data_array):
            yield [date] + [ x if not np_.isnan(x) else self.nan_str for x in data_row.tolist() ]
    #]


class DatabankExportMixin:
    """
    Databank mixin for exporting data to CSV sheets
    """
    #[
    def _to_sheet(
        self,
        file_name: str,
        /,
        descriptor_row: bool = False,
        range: Iterable[Dater] | None = None,
        frequency: da_.Frequency | None = None,
        delimiter: str = ",",
        numeric_format: str = "g",
        nan_str: str = "",
        csv_writer_settings: dict = {},
    ) -> Iterable[str]:
        """
        """
        if not frequency:
            raise NotImplementedYet("frequency=None")
        frequency = da_.Frequency(frequency)
        range = range if range else self._get_range_by_frequency(frequency)
        range = [ t for t in range ]
        num_data_rows = len(range)
        #
        names = self._get_series_names_by_frequency(frequency)
        descriptors = [ getattr(self, n)._description for n in names ]
        #
        num_series_columns = [ getattr(self, n).shape[1] for n in names ]
        data_array = np_.hstack([ getattr(self, n).get_data(range) for n in names ])
        export_block = _ExportBlockDescriptor(
            frequency, names, descriptors, descriptor_row,
            num_series_columns, range, data_array, 
            delimiter, numeric_format, nan_str,
        )
        with open(file_name, "w+") as fid:
            writer = cs_.writer(fid, delimiter=delimiter, **csv_writer_settings)
            for row in export_block.row_iterator():
                writer.writerow(row)
        #
        return names
    #]


def _get_frequency_mark(frequency, ):
    return "__" + frequency.name.lower() + "__"


