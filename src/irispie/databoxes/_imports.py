"""
Databox imports
"""


#[
from __future__ import annotations

from typing import (Self, )
from collections.abc import (Iterable, Callable, )
import csv as _cs
import numpy as _np
import dataclasses as _dc
import pickle as _pickle

from .. import dates as _dates
from .. import pages as _pages
from ..series import main as _series
#]


_DEFAULT_DATE_CREATOR = _dates.Dater.from_sdmx_string


@_dc.dataclass
class _ImportBlock:
    """
    """
    #[
    row_index: Iterable[int] | None = None,
    dates: Iterable[_dates.Dater] | None = None,
    column_start: int | None = None,
    num_columns: int | None = None,
    names: Iterable[str] | None = None,
    descriptions: Iterable[str] | None = None,

    def column_iterator(self, /, ):
        """
        """
        status = False
        names = self.names + [""]
        descriptions = self.descriptions + [""]
        current_columns = None
        current_name = None
        current_description = None
        for i, (n, d) in enumerate(zip(names, descriptions, )):
            if status and n!="*":
                status = False
                yield current_columns, current_name, current_description
            if status and n=="*":
                current_columns.append(i, )
            if not status and n and n!="*":
                status = True
                current_columns = [i]
                current_name = n
                current_description = d
    #]


class Inlay:
    """
    """
    #[
    @classmethod
    @_pages.reference(category="import_export", )
    def from_sheet(
        klass,
        file_name: str,
        /,
        date_creator: Callable | None = None,
        start_date_only: bool = False,
        description_row: bool = False,
        delimiter: str = ",",
        csv_reader_settings: dict = {},
        numpy_reader_settings: dict = {},
        name_row_transform: Callable | None = None,
        **kwargs,
    ) -> Self:
        """
················································································

==Import `Databox` from a CSV or spreadsheet file==

················································································
        """
        self = klass(**kwargs)

        num_header_rows = 1 + int(description_row)
        csv_rows = _read_csv(file_name, num_header_rows, **csv_reader_settings, )
        if not csv_rows:
            return self

        header_rows = csv_rows[0:num_header_rows]
        data_rows = csv_rows[num_header_rows:]
        name_row = header_rows[0]

        if name_row_transform:
            name_row = _apply_name_row_transform(name_row, name_row_transform, )

        description_row = header_rows[1] if description_row else [""] * len(name_row)
        date_creator = date_creator or _DEFAULT_DATE_CREATOR
        #
        for b in _block_iterator(name_row, description_row, data_rows, date_creator, start_date_only, ):
            array = _read_array_for_block(file_name, b, num_header_rows, delimiter=delimiter, **numpy_reader_settings, )
            _add_series_for_block(self, b, array, )
        #
        return self

    @classmethod
    def from_pickle(
        klass,
        file_name: str,
        /,
        **kwargs,
    ) -> Self:
        """
        """
        with open(file_name, "rb") as fid:
            return _pickle.load(fid, **kwargs, )
    #]


def _read_csv(file_name, num_header_rows, /, delimiter=",", **kwargs, ):
    """
    Read CSV cells into a list of lists
    """
    #[
    with open(file_name, "rt", encoding="utf-8-sig", ) as fid:
        all_rows = [ line for line in _cs.reader(fid, **kwargs, ) ]
    return all_rows
    #]


def _remove_nonascii_from_start(string, /, ):
    """
    Remove non-ascii characters from the start of a string
    """
    #[
    while string and not string[0].isascii():
        string = string[1:]
    return string
    #]


def _block_iterator(name_row, description_row, data_rows, date_creator, start_date_only, /, ):
    """
    """
    #[
    def _is_end(cell, /, ) -> bool:
        return cell.startswith("__")
    #
    def _is_start(cell, /, ) -> bool:
        if not cell.startswith("__"):
            return False
        try:
            letter = cell.removeprefix("__")[0]
            _dates.Frequency.from_letter(letter)
            return True
        except:
            return False
    #
    name_row += ["__"]
    status = False
    blocks = []
    current_date_column = None
    current_start = None
    current_frequency = None
    num_columns = len(name_row)
    for column, cell in enumerate(name_row):
        if status and _is_end(cell):
            status = False
            num_columns = column - current_start
            names = name_row[current_start:column]
            descriptions = description_row[current_start:column]
            row_index, dates = _extract_dates_from_data_rows(data_rows, current_frequency, current_date_column, date_creator, start_date_only, )
            yield _ImportBlock(row_index, dates, current_start, num_columns, names, descriptions)
        if not status and _is_start(cell):
            status = True
            current_date_column = column
            current_start = column + 1
            current_frequency = _dates.Frequency.from_letter(cell)
    #]


def _extract_dates_from_data_rows(
    data_rows,
    frequency: _dates.Frequency,
    column: int,
    date_creator: Callable,
    start_date_only: bool,
    /,
) -> tuple[tuple[int], tuple[_dates.Dater]]:
    """
    """
    #[
    start_date = date_creator(frequency, data_rows[0][column])
    date_extractor = {
        True: lambda i, line: start_date + i,
        False: lambda i, line: date_creator(frequency, line[column]),
    }[start_date_only]
    row_indices_and_dates = ( 
        (i, date_extractor(i, line))
        for i, line in enumerate(data_rows)
        if start_date_only or line[column]
    )
    return tuple(zip(*row_indices_and_dates)) or ((), ())
    #]


def _read_array_for_block(file_name, block, num_header_rows, /, delimiter=",", **kwargs, ):
    #[
    skip_header = num_header_rows
    usecols = [ c for c in range(block.column_start, block.column_start+block.num_columns) ]
    return _np.genfromtxt(file_name, skip_header=skip_header, usecols=usecols, delimiter=delimiter, ndmin=2, **kwargs)
    #]


def _add_series_for_block(self, block, array, /, ):
    """
    """
    #[
    array = array[block.row_index, :]
    for columns, name, description in block.column_iterator():
        series = _series.Series(num_variants=len(columns), description=description)
        series.set_data(block.dates, array[:, columns])
        self[name] = series
    #]

def _apply_name_row_transform(
    name_row: list[str],
    name_row_transform: Callable,
    /,
) -> list[str]:
    """
    """
    return [ name_row_transform(s) for s in name_row ]

