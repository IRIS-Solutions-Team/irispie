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

from ..series import main as _series
from .. import dates as _dates
#]


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


class DataboxImportMixin:
    """
    """
    #[
    @classmethod
    def from_sheet(
        klass,
        file_name: str,
        /,
        start_date_only: bool = False,
        description_row: bool = False,
        delimiter: str = ",",
        csv_reader_settings: dict = {},
        numpy_reader_settings: dict = {},
        **kwargs,
    ) -> Self:
        """
        """
        self = klass(**kwargs)

        num_header_lines = 1 + int(description_row)
        csv_lines = _read_csv(file_name, num_header_lines, **csv_reader_settings, )
        if not csv_lines:
            return self

        header_lines = csv_lines[0:num_header_lines]
        data_lines = csv_lines[num_header_lines:]
        name_line = header_lines[0]
        description_row = header_lines[1] if description_row else [""] * len(name_line)
        #
        for b in _block_iterator(name_line, description_row, data_lines, start_date_only, ):
            array = _read_array_for_block(file_name, b, num_header_lines, delimiter=delimiter, **numpy_reader_settings, )
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


def _read_csv(file_name, num_header_lines, /, delimiter=",", **kwargs, ):
    """
    Read CSV cells into a list of lists
    """
    #[
    with open(file_name, "rt", encoding="utf-8-sig", ) as fid:
        all_lines = [ line for line in _cs.reader(fid, **kwargs, ) ]
    return all_lines
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


def _block_iterator(name_line, description_row, data_lines, start_date_only, /, ):
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
    name_line += ["__"]
    status = False
    blocks = []
    current_date_column = None
    current_start = None
    current_frequency = None
    num_columns = len(name_line)
    for column, cell in enumerate(name_line):
        if status and _is_end(cell):
            status = False
            num_columns = column - current_start
            names = name_line[current_start:column]
            descriptions = description_row[current_start:column]
            row_index, dates = _extract_dates_from_data_lines(data_lines, current_frequency, current_date_column, start_date_only, )
            yield _ImportBlock(row_index, dates, current_start, num_columns, names, descriptions)
        if not status and _is_start(cell):
            status = True
            current_date_column = column
            current_start = column + 1
            current_frequency = _dates.Frequency.from_letter(cell)
    #]


def _extract_dates_from_data_lines(
    data_lines,
    frequency,
    column,
    start_date_only,
    /,
) -> tuple[tuple[int], tuple[_dates.Dater]]:
    """
    """
    #[
    start_date = _dates.Dater.from_sdmx_string(frequency, data_lines[0][column])
    date_extractor = {
        True: lambda i, line: start_date + i,
        False: lambda i, line: _dates.Dater.from_sdmx_string(frequency, line[column]),
    }[start_date_only]
    row_indices_and_dates = ( 
        (i, date_extractor(i, line))
        for i, line in enumerate(data_lines)
        if start_date_only or line[column]
    )
    return tuple(zip(*row_indices_and_dates)) or ((), ())
    #]


def _read_array_for_block(file_name, block, num_header_lines, /, delimiter=",", **kwargs, ):
    #[
    skip_header = num_header_lines
    usecols = [ c for c in range(block.column_start, block.column_start+block.num_columns) ]
    return _np.genfromtxt(file_name, skip_header=skip_header, usecols=usecols, delimiter=delimiter, ndmin=2, **kwargs)
    #]


def _add_series_for_block(self, block, array, /, ):
    """
    """
    #[
    array = array[block.row_index, :]
    for columns, name, description in block.column_iterator():
        series = _series.Series(num_columns=len(columns), description=description)
        series.set_data(block.dates, array[:, columns])
        self[name] = series
    #]

