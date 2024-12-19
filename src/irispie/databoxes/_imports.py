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
    r"""
    ................................................................................
    ==Import Block for Parsing CSV Data==

    Represents a logical block of data within a CSV file, consisting of 
    periods, series names, and descriptions. This class helps in organizing 
    and iterating through columns in a structured manner.

    ### Attributes ###
    - `row_index`: Row indices for the data rows.
    - `periods`: Period objects representing the time span.
    - `column_start`: Starting column index for the block.
    - `num_columns`: Number of columns in the block.
    - `names`: Names of the series within the block.
    - `descriptions`: Descriptions of the series within the block.

    ### Example ###
    ```python
        block = _ImportBlock(
            row_index=[0, 1, 2],
            periods=[Period(...), ...],
            column_start=0,
            num_columns=3,
            names=["Series1", "Series2"],
            descriptions=["Description1", "Description2"]
        )
    ```
    ................................................................................
    """
    #[
    row_index: Iterable[int] | None = None,
    periods: Iterable[Period] | None = None,
    column_start: int | None = None,
    num_columns: int | None = None,
    names: Iterable[str] | None = None,
    descriptions: Iterable[str] | None = None,

    def column_iterator(self, /, ):
        r"""
        ................................................................................
        ==Iterate Through Columns in the Block==

        Yields the columns, names, and descriptions of series in this block. 
        Columns marked with `*` are considered part of the same series.

        ### Input arguments ###
        ???+ input
            None

        ### Returns ###
        ???+ returns
            `Iterable[tuple[list[int], str, str]]`: A generator yielding a tuple 
            with columns, series name, and description.

        ### Example ###
        ```python
            for columns, name, description in block.column_iterator():
                print(columns, name, description)
        ```
        ................................................................................
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
    r"""
    ................................................................................
    ==Databox Import Inlay==

    Provides methods for importing `Databox` objects from various file formats, 
    such as CSV and pickle. Includes support for handling time series data, 
    descriptions, and transformations.

    ### Example ###
    ```python
        databox = Inlay.from_csv("data.csv", description_row=True)
    ```
    ................................................................................
    """
    #[
    @classmethod
    @_dm.reference(
        category="import_export",
        call_name="Databox.from_csv",
    )
    def from_csv(
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
        ................................................................................
        ==Create a Databox from a CSV File==

        Reads time series data from a CSV file and constructs a `Databox` 
        object. Supports parsing metadata and transforming name rows.

        ### Input arguments ###
        ???+ input "file_name"
            Path to the CSV file to be read.

        ???+ input "period_from_string"
            A callable for creating date objects from string representations. 
            Defaults to a method based on the SDMX string format.

        ???+ input "start_period_only"
            If `True`, only the start date of each time series is parsed from the 
            CSV; subsequent periods are inferred based on frequency.

        ???+ input "description_row"
            Indicates if the CSV contains a row for descriptions of the time series.

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
            A `Databox` populated with time series from the CSV file.

        ### Example ###
        ```python
            databox = Inlay.from_csv("data.csv", description_row=True)
            print(databox)
        ```
        ................................................................................
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
    def from_sheet(klass, *args, **kwargs, ):
        r"""
        ................................................................................
        ==Alias for `from_csv`==

        Provides an alias for the `from_csv` method, enabling consistent naming 
        for importing from different file formats.

        ### Input arguments ###
        ???+ input
            Accepts all parameters of the `from_csv` method.

        ### Returns ###
        ???+ returns
            Same return values as the `from_csv` method.

        ### Example ###
        ```python
            databox = Inlay.from_sheet("data.csv", description_row=True)
        ```
        ................................................................................
        """
        return klass.from_csv(*args, **kwargs, )

    @classmethod
    def from_pickle(
        klass,
        file_name: str,
        /,
        **kwargs,
    ) -> Self:
        r"""
        ................................................................................
        ==Load a Databox from a Pickle File==

        Deserialize a `Databox` object from a pickle file, restoring its state.

        ### Input arguments ###
        ???+ input "file_name"
            Path to the pickle file to load the `Databox` from.

        ???+ input "kwargs"
            Additional arguments passed to the `pickle.load` function.

        ### Returns ###
        ???+ returns
            A `Databox` object restored from the pickle file.

        ### Example ###
        ```python
            databox = Inlay.from_pickle("data.pkl")
            print(databox)
        ```
        ................................................................................
        """
        with open(file_name, "rb") as fid:
            return _pickle.load(fid, **kwargs, )

    #]


def _read_csv(file_name, num_header_rows, /, delimiter=",", **kwargs, ):
    r"""
    ................................................................................
    ==Read CSV File into List of Rows==

    Reads a CSV file and returns its contents as a list of rows, excluding 
    non-ASCII characters from the file header.

    ### Input arguments ###
    ???+ input "file_name"
        Path to the CSV file.

    ???+ input "num_header_rows"
        Number of header rows to include in the result.

    ???+ input "delimiter"
        Character used to separate fields in the CSV file.

    ???+ input "kwargs"
        Additional arguments passed to the CSV reader.

    ### Returns ###
    ???+ returns
        `list[list[str]]`: A list of rows, where each row is a list of strings.

    ### Example ###
    ```python
        rows = _read_csv("data.csv", 2, delimiter=",")
        print(rows)
    ```
    ................................................................................
    """
    #[
    with open(file_name, "rt", encoding="utf-8-sig", ) as fid:
        all_rows = [ line for line in _cs.reader(fid, **kwargs, ) ]
    return all_rows
    #]


def _remove_nonascii_from_start(string, /, ):
    r"""
    ................................................................................
    ==Remove Non-ASCII Characters from String Start==

    Removes non-ASCII characters from the beginning of a string.

    ### Input arguments ###
    ???+ input "string"
        The string to process.

    ### Returns ###
    ???+ returns
        `str`: The cleaned string with non-ASCII characters removed from the start.

    ### Example ###
    ```python
        clean_string = _remove_nonascii_from_start("©Data")
        print(clean_string)  # Output: "Data"
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Iterate Through Data Blocks in a CSV==

    Identify and yield blocks of data within a CSV file based on series 
    names, descriptions, and date columns.

    ### Input arguments ###
    ???+ input "name_row"
        The row containing series names.

    ???+ input "description_row"
        The row containing series descriptions.

    ???+ input "data_rows"
        The rows containing data values.

    ???+ input "period_from_string"
        Callable to parse period strings into `Period` objects.

    ???+ input "start_period_only"
        If `True`, only the start date of the series is processed.

    ### Returns ###
    ???+ returns
        `Iterable[_ImportBlock]`: A generator yielding `_ImportBlock` objects 
        for each identified data block.

    ### Example ###
    ```python
        for block in _block_iterator(name_row, desc_row, data_rows, parser, False):
            print(block)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Extract Periods from Data Rows==

    Parses the data rows to extract indices and periods for a specific frequency 
    and column. Handles both complete and inferred period sequences.

    ### Input arguments ###
    ???+ input "data_rows"
        Rows containing the data values.

    ???+ input "frequency"
        Frequency of the periods (e.g., yearly, monthly).

    ???+ input "column"
        Column index in the rows that contains date values.

    ???+ input "period_from_string"
        Callable to parse date strings into `Period` objects.

    ???+ input "start_period_only"
        If `True`, only the first date is parsed; subsequent periods are inferred.

    ### Returns ###
    ???+ returns
        `tuple[tuple[int], tuple[Period]]`: A tuple containing:
        - Row indices of valid periods.
        - Corresponding `Period` objects.

    ### Example ###
    ```python
        indices, periods = _extract_periods_from_data_rows(data_rows, Frequency.MONTHLY, 0, parser, False)
        print(indices, periods)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Read Data Array for a Block==

    Reads a block of data from the CSV file into a NumPy array. Handles column 
    selection and header skipping.

    ### Input arguments ###
    ???+ input "file_name"
        Path to the CSV file.

    ???+ input "block"
        The `_ImportBlock` object defining the data block.

    ???+ input "num_header_rows"
        Number of header rows to skip.

    ???+ input "delimiter"
        Character used to separate fields in the CSV file.

    ???+ input "kwargs"
        Additional arguments passed to `numpy.genfromtxt`.

    ### Returns ###
    ???+ returns
        `numpy.ndarray`: The data block read into a NumPy array.

    ### Example ###
    ```python
        array = _read_array_for_block("data.csv", block, 2, delimiter=",")
        print(array)
    ```
    ................................................................................
    """
    #[
    skip_header = num_header_rows
    usecols = [ c for c in range(block.column_start, block.column_start+block.num_columns) ]
    return _np.genfromtxt(file_name, skip_header=skip_header, usecols=usecols, delimiter=delimiter, ndmin=2, **kwargs)
    #]


def _add_series_for_block(self, block, array, /, ):
    r"""
    ................................................................................
    ==Add Series for a Data Block==

    Converts data from an array into `Series` objects and adds them to the `Databox`.

    ### Input arguments ###
    ???+ input "self"
        The `Databox` object to which the series will be added.

    ???+ input "block"
        The `_ImportBlock` object defining the block of data.

    ???+ input "array"
        The NumPy array containing the data for the block.

    ### Returns ###
    ???+ returns
        None: This function updates the `Databox` in-place.

    ### Example ###
    ```python
        _add_series_for_block(databox, block, array)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Transform Name Row==

    Applies a transformation function to each name in the name row of a CSV file.

    ### Input arguments ###
    ???+ input "name_row"
        List of names in the name row.

    ???+ input "name_row_transform"
        Function to apply to each name.

    ### Returns ###
    ???+ returns
        `list[str]`: A list of transformed names.

    ### Example ###
    ```python
        transformed = _apply_name_row_transform(["Name1", "Name2"], str.upper)
        print(transformed)  # Output: ["NAME1", "NAME2"]
    ```
    ................................................................................
    """
    return [ name_row_transform(s) for s in name_row ]


def _resolve_legacy_option(
    option,
    legacy,
    warning_message: str,
):
    r"""
    ................................................................................
    ==Resolve Legacy Option==

    Resolves and deprecates legacy options by warning users of the updated parameter.

    ### Input arguments ###
    ???+ input "option"
        The new option value.

    ???+ input "legacy"
        The legacy option value.

    ???+ input "warning_message"
        Warning message displayed if the legacy option is used.

    ### Returns ###
    ???+ returns
        The resolved option value.

    ### Example ###
    ```python
        resolved = _resolve_legacy_option(None, legacy_value, "Use the new option.")
        print(resolved)
    ```
    ................................................................................
    """
    if (option is None) and (legacy is not None):
        _wa.warn(warning_message, FutureWarning, )
        option = legacy
    return option

