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
import warnings as _wa
import documark as _dm

from ..dates import (Period, Frequency, Span, )
from ..series.main import (Series, )

#]


_DEFAULT_PERIOD_FROM_STRING = Period.from_sdmx_string


@_dc.dataclass
class _ImportBlock:
    """
    """
    #[
    row_index: Iterable[int] | None = None,
    periods: Iterable[Period] | None = None,
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
    @_dm.reference(
        category="constructor",
        call_name="Databox.from_csv_file",
    )
    def from_csv_file(
        klass,
        file_name: str,
        *,
        period_from_string: Callable | None = None,
            date_creator: Callable | None = None,
        start_period_only: bool = False,
            start_date_only: bool | None = None,
        description_row: bool = False,
        delimiter: str = ",",
        name_row_transform: Callable | None = None,
        csv_reader_settings: dict = {},
        numpy_reader_settings: dict = {},
        databox_settings: dict = {},
    ) -> Self:
        r"""
················································································


==Create a new Databox by reading time series from a CSV file==


    self = Databox.from_csv_file(
        file_name,
        *,
        period_from_string=None,
        start_period_only=False,
        description_row=False,
        delimiter=",",
        csv_reader_settings={},
        numpy_reader_settings={},
        name_row_transform=None,
    )


### Input arguments ###


???+ input "file_name"
    Path to the CSV file to be read.

???+ input "period_from_string"
    A callable for creating date objects from string representations. If `None`,
    a default method expecting the SDMX string format is used.

???+ input "start_period_only"
    If `True`, only the start date of each time series is parsed from the CSV;
    subsequent periods are inferred based on frequency.

???+ input "description_row"
    Indicates if the CSV contains a row for descriptions of the time series.
    Defaults to `False`.

???+ input "delimiter"
    Character used to separate values in the CSV file.

???+ input "name_row_transform"
    A function to transform names in the name row of the CSV.

???+ input "csv_reader_settings"
    Additional settings for the CSV reader.

???+ input "numpy_reader_settings"
    Settings for reading data into numpy arrays.

???+ input "databox_settings"
    Settings for the Databox constructor.


### Returns ###


???+ returns "self"
    An `Databox` populated with time series from the CSV file.

················································································
        """
        self = klass(**databox_settings, )
        #
        period_from_string = _resolve_legacy_option(
            period_from_string,
            date_creator,
            "The 'date_creator' option is deprecated; use 'period_from_string' instead",
            )
        start_period_only = _resolve_legacy_option(
            start_period_only,
            start_date_only,
            "The 'start_date_only' option is deprecated; use 'start_period_only' instead",
        )
        #
        num_header_rows = 1 + int(description_row)
        csv_rows = _read_csv(file_name, num_header_rows, **csv_reader_settings, )
        if not csv_rows:
            return self
        #
        header_rows = csv_rows[0:num_header_rows]
        data_rows = csv_rows[num_header_rows:]
        name_row = header_rows[0]
        #
        if name_row_transform:
            name_row = _apply_name_row_transform(name_row, name_row_transform, )
        #
        description_row = header_rows[1] if description_row else [""] * len(name_row)
        period_from_string = period_from_string or _DEFAULT_PERIOD_FROM_STRING
        #
        for b in _block_iterator(name_row, description_row, data_rows, period_from_string, start_period_only, ):
            array = _read_array_for_block(file_name, b, num_header_rows, delimiter=delimiter, **numpy_reader_settings, )
            _add_series_for_block(self, b, array, )
        #
        return self

    @classmethod
    @_dm.no_reference
    def from_sheet(klass, *args, **kwargs, ):
        r"""
        """
        return klass.from_csv_file(*args, **kwargs, )

    # Legacy alias
    from_csv = from_csv_file

    @classmethod
    @_dm.reference(category="import_export", )
    def from_pickle_file(
        klass,
        file_name: str,
        /,
        **kwargs,
    ) -> Self:
        r"""
................................................................................

==Read a Databox from a pickled file==

    self = Databox.from_pickle(
        file_name,
        **kwargs,
    )

### Input arguments ###

???+ input "file_name"
    Path to the pickled file to be read.

???+ input "kwargs"
    Additional keyword arguments to pass to the `pickle.load` function.

### Returns ###

???+ returns "self"
    A `Databox` object read from the pickled file.

................................................................................
        """
        with open(file_name, "rb") as fid:
            return _pickle.load(fid, **kwargs, )

    from_pickle = from_pickle_file

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


def _block_iterator(
    name_row,
    description_row,
    data_rows,
    period_from_string,
    start_period_only,
    /,
):
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
            Frequency.from_letter(letter)
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
            row_index, periods = _extract_periods_from_data_rows(
                data_rows,
                current_frequency,
                current_date_column,
                period_from_string,
                start_period_only,
            )
            yield _ImportBlock(row_index, periods, current_start, num_columns, names, descriptions)
        if not status and _is_start(cell):
            status = True
            current_date_column = column
            current_start = column + 1
            current_frequency = Frequency.from_letter(cell)
    #]


def _extract_periods_from_data_rows(
    data_rows,
    frequency: Frequency | None,
    column: int,
    period_from_string: Callable,
    start_period_only: bool,
    /,
) -> tuple[tuple[int], tuple[Period]]:
    """
    """
    #[
    start_date = period_from_string(data_rows[0][column], frequency=frequency, )
    date_extractor = {
        True: lambda i, line: start_date + i,
        False: lambda i, line: period_from_string(line[column], frequency=frequency, ),
    }[start_period_only]
    row_indices_and_dates = ( 
        (i, date_extractor(i, line))
        for i, line in enumerate(data_rows)
        if start_period_only or line[column]
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
        series = Series(num_variants=len(columns), description=description)
        series.set_data(block.periods, array[:, columns])
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


def _resolve_legacy_option(
    option,
    legacy,
    warning_message: str,
):
    """
    """
    if (option is None) and (legacy is not None):
        _wa.warn(warning_message, FutureWarning, )
        option = legacy
    return option

