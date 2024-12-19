"""
Import time series data from a CSV file
"""


#[
from __future__ import annotations

from typing import (Self, TypeAlias, Protocol, Callable, )
from collections.abc import (Iterator, Iterable, Generator, )
import csv as _cs
import numpy as _np
import itertools as _it
import functools as _ft

from ..series import main as _series
from .. import dates as _dates

from . import main as _databoxes
#]


_Header: TypeAlias = tuple[str, str]
_DataVector: TypeAlias = tuple[str, ...]
_HeaderIterator: TypeAlias = Iterator[_Header, ]
_DataIterator: TypeAlias = Iterator[_DataVector, ]
_DataArrayReader: TypeAlias = Callable[[Iterable[int], ], _np.ndarray]


_GENFROMTXT_SETTINGS = {
    "delimiter": ",",
    "ndmin": 2,
}


class _SheetFileFactory(Protocol):
    r"""
    ................................................................................
    ==Protocol for File Factory==

    This protocol defines the structure for file factories that manage the 
    creation of iterators and readers for headers, data, and arrays.

    ### Methods ###
    - `create_header_and_data_iterators`: Creates iterators for headers and data.
    - `create_data_array_reader`: Creates a reader for loading data arrays.

    ### Example ###
    ```python
        factory = SomeFileFactory()
        headers, data = factory.create_header_and_data_iterators()
        data_reader = factory.create_data_array_reader()
    ```
    ................................................................................
    """
    #[

    def create_header_and_data_iterators(self, ) -> tuple[_HeaderIterator, _DataIterator, ]:
        ...
    def create_data_array_reader(self, ) -> _DataArrayReader:
        ...

    #]


#
# FIXME: Make read_csv a class method
#
def read_csv(
    file_path: str,
    /,
    **kwargs,
) -> Iterable[Self]:
    r"""
    ................................................................................
    ==Read a CSV File==

    Reads a CSV file and parses its content into one or more `Databox` objects. 
    Data is grouped into logical blocks based on headers.

    ### Input arguments ###
    ???+ input "file_path"
        Path to the CSV file.

    ???+ input "kwargs"
        Additional arguments passed to the file factory and block generator.

    ### Returns ###
    ???+ returns
        `Iterable[Self]`: An iterable of `Databox` objects created from the CSV.

    ### Example ###
    ```python
        databoxes = read_csv("data.csv")
        for databox in databoxes:
            print(databox)
    ```
    ................................................................................
    """
    #[

    factory = _ColumnwiseFileFactory(file_path, **kwargs, )
    #
    header_iterator, data_iterator = factory.create_header_and_data_iterators()
    data_array_reader = factory.create_data_array_reader()
    all_databoxes = tuple(
        b.create_databox_from_block(data_array_reader, **kwargs, )
        for b in _generate_blocks(header_iterator, data_iterator, **kwargs, )
    )
    return all_databoxes

    #]


def _remove_nonascii_from_start(
    string: str,
    /,
) -> str:
    r"""
    ................................................................................
    ==Remove Non-ASCII Characters from String Start==

    Strips non-ASCII characters from the beginning of a string.

    ### Input arguments ###
    ???+ input "string"
        The input string to process.

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


def _generate_blocks(
    header_iterator: _HeaderIterator,
    data_iterator: _DataIterator,
    /,
    **kwargs,
) -> _Block | None:
    r"""
    ................................................................................
    ==Generate Data Blocks==

    Creates blocks of data grouped by the same date frequency. Each block is 
    represented as a `_Block` object.

    ### Input arguments ###
    ???+ input "header_iterator"
        Iterator over the headers in the file.

    ???+ input "data_iterator"
        Iterator over the data rows in the file.

    ???+ input "kwargs"
        Additional settings for block creation.

    ### Returns ###
    ???+ returns
        `Iterable[_Block]`: An iterable of `_Block` objects representing the data blocks.

    ### Example ###
    ```python
        blocks = _generate_blocks(headers, data, freq="MONTHLY")
        for block in blocks:
            print(block)
    ```
    ................................................................................
    """
    #[
    def _is_yieldable(block, /, ) -> bool:
        return block is not None and block.num_columns > 0
    #
    current_block = None
    for index, (header, data, ) in enumerate(zip(header_iterator, data_iterator, )):
        if _is_end_of_file(header, ):
            if _is_yieldable(current_block, ):
                yield current_block
            current_block = None
        elif _is_start_of_block(header, ):
            if _is_yieldable(current_block, ):
                yield current_block
            current_block = _Block(index, header, data, **kwargs, )
        elif current_block is not None:
            current_block.add_header(header, )
        else:
            current_block = None
    if _is_yieldable(current_block, ):
        yield current_block
    #]


class _Block:
    r"""
    ................................................................................
    ==Block of Data==

    Represents a logical block of data within a file. Each block includes metadata 
    such as headers, date frequencies, and associated data. Blocks are created 
    based on common properties like date frequency.

    ### Attributes ###
    - `_start_index`: The starting column index for the block.
    - `_headers`: Headers associated with the block.
    - `_dates`: Dates representing the time period of the block.
    - `_data_array`: The actual data values for the block.

    ### Example ###
    ```python
        block = _Block(0, ("__M", ""), ("2023-01", "2023-02"), freq="MONTHLY")
        print(block.num_columns)
    ```
    ................................................................................
    """
    #[

    __slots__ = (
        "_start_index",
        "_headers",
        "_dates",
        "_data_array",
    )

    def __init__(
        self,
        start_index: int | None,
        header: _Header | None,
        date_str_vector: _DataVector | None,
        /,
        **kwargs,
    ) -> None:
                r"""
        ................................................................................
        ==Initialize a Block==

        Constructs a new `_Block` object using provided metadata and data.

        ### Input arguments ###
        ???+ input "start_index"
            Starting column index for the block.

        ???+ input "header"
            The header information for the block.

        ???+ input "date_str_vector"
            A vector of date strings representing the time periods.

        ???+ input "kwargs"
            Additional arguments for block configuration.

        ### Returns ###
        ???+ returns
            None: Initializes the `_Block` instance.
        ................................................................................
        """
        self._start_index = start_index
        self._headers = []
        freq = _dates.Frequency.from_letter(header[0]) if header is not None else None
        self._dates = _create_dates_for_block(freq, date_str_vector, **kwargs, )
        self._data_array = None

    def add_header(
        self,
        header: _Header,
        /,
    ) -> None:
        r"""
        ................................................................................
        ==Add a Header to the Block==

        Appends a header to the block's metadata.

        ### Input arguments ###
        ???+ input "header"
            The header tuple (name, description) to be added.

        ### Returns ###
        ???+ returns
            None: Updates the block's headers in-place.

        ### Example ###
        ```python
            block.add_header(("Series1", "Description1"))
        ```
        ................................................................................
        """
        self._headers.append(header)

    def create_databox_from_block(
        self,
        data_array_reader: _DataArrayReader,
        /,
        **kwargs,
    ) -> _databoxes.Databox:
        r"""
        ................................................................................
        ==Create a Databox from the Block==

        Uses the block's data to create a `Databox` object.

        ### Input arguments ###
        ???+ input "data_array_reader"
            A callable to read the block's data array.

        ???+ input "kwargs"
            Additional arguments for creating the `Databox`.

        ### Returns ###
        ???+ returns
            `_databoxes.Databox`: A `Databox` object representing the block's data.

        ### Example ###
        ```python
            databox = block.create_databox_from_block(data_reader)
        ```
        ................................................................................
        """
        self._load_data_array(data_array_reader, )
        return self._data_array

    def _load_data_array(
        self,
        data_array_reader: _DataArrayReader,
        /,
    ) -> _np.ndarray:
        r"""
        ................................................................................
        ==Load Data Array for the Block==

        Loads the block's data as a NumPy array using the provided reader.

        ### Input arguments ###
        ???+ input "data_array_reader"
            A callable to read the block's data array.

        ### Returns ###
        ???+ returns
            `_np.ndarray`: The loaded data array.

        ### Example ###
        ```python
            array = block._load_data_array(data_reader)
            print(array)
        ```
        ................................................................................
        """
        self._data_array = data_array_reader(usecols=self.column_indices, )

    @property
    def num_columns(
        self,
        /,
    ) -> int:
        r"""
        ................................................................................
        ==Get the Number of Columns==

        Returns the number of columns in the block.

        ### Input arguments ###
        ???+ input
            None

        ### Returns ###
        ???+ returns
            `int`: The number of columns in the block.

        ### Example ###
        ```python
            columns = block.num_columns
            print(columns)
        ```
        ................................................................................
        """
        return len(self._headers)

    @property
    def column_indices(
        self,
        /,
    ) -> range:
        r"""
        ................................................................................
        ==Get Column Indices==

        Returns the range of column indices for the block.

        ### Input arguments ###
        ???+ input
            None

        ### Returns ###
        ???+ returns
            `range`: The range of column indices.

        ### Example ###
        ```python
            indices = block.column_indices
            print(list(indices))
        ```
        ................................................................................
        """
        return range(
            self._start_index,
            self._start_index + self.num_columns,
        )

    #]


_END_OF_FILE = "__eof__"


def _is_start_of_block(
    header: _Header,
    /,
) -> bool:
    r"""
    ................................................................................
    ==Check if Header Marks the Start of a Block==

    Determines if a header represents the start of a new data block.

    ### Input arguments ###
    ???+ input "header"
        The header tuple to check.

    ### Returns ###
    ???+ returns
        `bool`: `True` if the header marks the start of a block, otherwise `False`.

    ### Example ###
    ```python
        is_start = _is_start_of_block(("__M", ""))
        print(is_start)  # Output: True
    ```
    ................................................................................
    """
    return header[0].startswith("__") and header[0] != _END_OF_FILE


def _is_end_of_file(
    header: _Header,
    /,
) -> bool:
    r"""
    ................................................................................
    ==Check if Header Marks End of File==

    Determines if a header represents the end of the file.

    ### Input arguments ###
    ???+ input "header"
        The header tuple to check.

    ### Returns ###
    ???+ returns
        `bool`: `True` if the header marks the end of the file, otherwise `False`.

    ### Example ###
    ```python
        is_eof = _is_end_of_file(("__eof__", ""))
        print(is_eof)  # Output: True
    ```
    ................................................................................
    """
    return header[0] == _END_OF_FILE


class _ColumnwiseFileFactory(_SheetFileFactory):
    r"""
    ................................................................................
    ==Columnwise File Factory==

    Provides iterators and readers for processing CSV files in a columnwise 
    manner. Handles parsing of headers, data rows, and data arrays.

    ### Attributes ###
    - `_file_path`: Path to the CSV file.
    - `_skip_rows`: Number of rows to skip before reading data.
    - `_has_description_row`: Indicates if the file includes a description row.

    ### Example ###
    ```python
        factory = _ColumnwiseFileFactory("data.csv", skip_rows=1, has_description_row=True)
        headers, data = factory.create_header_and_data_iterators()
        reader = factory.create_data_array_reader()
    ```
    ................................................................................
    """
    #[
    def __init__(
        self,
        file_path: str,
        /,
        **kwargs,
    ) -> None:
        r"""
        ................................................................................
        ==Initialize the File Factory==

        Sets up the factory with file path and configuration options.

        ### Input arguments ###
        ???+ input "file_path"
            Path to the CSV file.

        ???+ input "kwargs"
            Additional settings such as `skip_rows` and `has_description_row`.

        ### Returns ###
        ???+ returns
            None: Initializes the file factory.
        ................................................................................
        """        
        self._file_path = file_path
        self._skip_rows = kwargs.get("skip_rows", 0, )
        self._has_description_row = kwargs.get("has_description_row", False, )

    def create_header_and_data_iterators(
        self,
        /,
    ) -> tuple[Iterator[_Header], Iterator[_DataVector], ]:
        r"""
        ................................................................................
        ==Create Header and Data Iterators==

        Creates iterators for headers and data rows from the CSV file.

        ### Returns ###
        ???+ returns
            `tuple[Iterator[_Header], Iterator[_DataVector]]`: Iterators for headers 
            and data rows.

        ### Example ###
        ```python
            headers, data = factory.create_header_and_data_iterators()
            for header in headers:
                print(header)
        ```
        ................................................................................
        """        
        with open(self._file_path, encoding="utf-8-sig", ) as file:
            csv_reader = _cs.reader(file, )
            header_iterator = self._create_header_iterator(csv_reader, )
            data_iterator = self._create_data_iterator(csv_reader, )
        return header_iterator, data_iterator, 

    def create_data_array_reader(
        self,
        /,
    ) -> Callable[[Iterable[int], ], _np.ndarray]:
        r"""
        ................................................................................
        ==Create Data Array Reader==

        Creates a reader function for loading data arrays from the CSV file.

        ### Returns ###
        ???+ returns
            `Callable[[Iterable[int]], _np.ndarray]`: A function to read data arrays.

        ### Example ###
        ```python
            reader = factory.create_data_array_reader()
            array = reader(usecols=[1, 2, 3])
            print(array)
        ```
        ................................................................................
        """
        skip_rows = self._skip_rows + 1 + int(self._has_description_row)
        return _ft.partial(
            _np.genfromtxt,
            self._file_path,
            skip_header=skip_rows,
            **_GENFROMTXT_SETTINGS,
        )

    def _create_header_iterator(
        self,
        csv_reader,
        /,
    ) -> Iterator[_Header]:
        r"""
        ................................................................................
        ==Create Header Iterator==

        Parses the CSV file to create an iterator over headers.

        ### Input arguments ###
        ???+ input "csv_reader"
            The CSV reader object.

        ### Returns ###
        ???+ returns
            `Iterator[_Header]`: An iterator over header tuples.

        ### Example ###
        ```python
            headers = factory._create_header_iterator(csv_reader)
            for header in headers:
                print(header)
        ```
        ................................................................................
        """
        for _ in range(self._skip_rows):
            next(csv_reader, )
        name_row = next(csv_reader, )
        if name_row and name_row[0]:
            name_row[0] = _remove_nonascii_from_start(name_row[0], )
        description_row = next(csv_reader, ) if self._has_description_row else _it.repeat("", )
        return zip(name_row, description_row, )

    def _create_data_iterator(
        self,
        csv_reader,
        /,
    ) -> Iterator[_DataVector]:
        r"""
        ................................................................................
        ==Create Data Iterator==

        Parses the CSV file to create an iterator over data rows.

        ### Input arguments ###
        ???+ input "csv_reader"
            The CSV reader object.

        ### Returns ###
        ???+ returns
            `Iterator[_DataVector]`: An iterator over data rows.

        ### Example ###
        ```python
            data = factory._create_data_iterator(csv_reader)
            for row in data:
                print(row)
        ```
        ................................................................................
        """
        return _it.zip_longest(*csv_reader, fillvalue="", )


def _create_dates_for_block(
    freq: _dates.Frequency | None,
    date_str_vector: _DataVector | None,
    /,
    start_date_only: bool = False,
    **kwargs,
) -> tuple[Dater]:
    r"""
    ................................................................................
    ==Create Dates for a Block==

    Generates a sequence of dates for a block based on its frequency and 
    date string vector.

    ### Input arguments ###
    ???+ input "freq"
        The frequency of the dates (e.g., MONTHLY, DAILY).

    ???+ input "date_str_vector"
        A vector of date strings.

    ???+ input "start_date_only"
        If `True`, only the start date is used; subsequent dates are inferred.

    ???+ input "kwargs"
        Additional arguments for date creation.

    ### Returns ###
    ???+ returns
        `tuple[_dates.Dater]`: A tuple of `Dater` objects representing the block's dates.

    ### Example ###
    ```python
        dates = _create_dates_for_block(Frequency.MONTHLY, ("2023-01", "2023-02"))
        print(dates)
    ```
    ................................................................................
    """
    #[
    if start_date_only:
        dates_factory = _create_dates_from_start_date
    else:
        dates_factory = _create_dates_from_date_column
    return dates_factory(freq, date_str_vector, )
    #]


def _create_dates_from_start_date(
    freq: _dates.Frequency | None,
    date_str_vector: _DataVector | None,
) -> tuple[Dater]:
    r"""
    ................................................................................
    ==Create Dates from Start Date==

    Generates a sequence of dates starting from the first date in the vector.

    ### Input arguments ###
    ???+ input "freq"
        The frequency of the dates.

    ???+ input "date_str_vector"
        A vector of date strings.

    ### Returns ###
    ???+ returns
        `tuple[_dates.Dater]`: A tuple of `Dater` objects.

    ### Example ###
    ```python
        dates = _create_dates_from_start_date(Frequency.MONTHLY, ("2023-01",))
        print(dates)
    ```
    ................................................................................
    """
    #[
    num_dates = len(date_str_vector)
    start_date = _dates.Dater.from_sdmx_string(date_str_vector[0], frequency=freq, )
    return tuple(_dates.Ranger(start_date, num_dates, ))
    #]


def _create_dates_from_date_column(
    freq: _dates.Frequency | None,
    date_str_vector: _DataVector | None,
) -> tuple[Dater]:
    r"""
    ................................................................................
    ==Create Dates from Date Column==

    Generates a sequence of dates based on the provided date string vector.

    ### Input arguments ###
    ???+ input "freq"
        The frequency of the dates.

    ???+ input "date_str_vector"
        A vector of date strings.

    ### Returns ###
    ???+ returns
        `tuple[_dates.Dater]`: A tuple of `Dater` objects.

    ### Example ###
    ```python
        dates = _create_dates_from_date_column(Frequency.DAILY, ("2023-01-01", "2023-01-02"))
        print(dates)
    ```
    ................................................................................
    """
    #[
    return tuple(
        _dates.Dater.from_sdmx_string(s, frequency=freq, ) if s else None
        for s in date_str_vector
    )
    #]

