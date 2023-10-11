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
    """
    Protocol for file factory
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
    """
    Read a CSV file
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
    """
    Remove non-ASCII characters from start of string
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
    """
    Generate blocks of data of same date frequency
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
    """
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
        """
        Add a header to the block
        """
        self._headers.append(header)

    def create_databox_from_block(
        self,
        data_array_reader: _DataArrayReader,
        /,
        **kwargs,
    ) -> _databoxes.Databox:
        """
        Create a databox from the block
        """
        self._load_data_array(data_array_reader, )
        return self._data_array

    def _load_data_array(
        self,
        data_array_reader: _DataArrayReader,
        /,
    ) -> _np.ndarray:
        """
        Load block data as a numpy array
        """
        self._data_array = data_array_reader(usecols=self.column_indices, )

    @property
    def num_columns(
        self,
        /,
    ) -> int:
        """
        Return the number of columns
        """
        return len(self._headers)

    @property
    def column_indices(
        self,
        /,
    ) -> range:
        """
        Return the column indices
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
    """
    Return true if the name is the start of a block
    """
    return header[0].startswith("__") and header[0] != _END_OF_FILE


def _is_end_of_file(
    header: _Header,
    /,
) -> bool:
    """
    Return true if the name is the end of a file
    """
    return header[0] == _END_OF_FILE


class _ColumnwiseFileFactory(_SheetFileFactory):
    """
    Iterator factory for columnwise data
    """
    #[
    def __init__(
        self,
        file_path: str,
        /,
        **kwargs,
    ) -> None:
        self._file_path = file_path
        self._skip_rows = kwargs.get("skip_rows", 0, )
        self._has_description_row = kwargs.get("has_description_row", False, )

    def create_header_and_data_iterators(
        self,
        /,
    ) -> tuple[Iterator[_Header], Iterator[_DataVector], ]:
        with open(self._file_path, encoding="utf-8-sig", ) as file:
            csv_reader = _cs.reader(file, )
            header_iterator = self._create_header_iterator(csv_reader, )
            data_iterator = self._create_data_iterator(csv_reader, )
        return header_iterator, data_iterator, 

    def create_data_array_reader(
        self,
        /,
    ) -> Callable[[Iterable[int], ], _np.ndarray]:
        """
        Create a reader for the data array
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
        """
        Create an iterator over headers (name, description, )
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
        """
        Create an iterator over individual data series or dates
        """
        return _it.zip_longest(*csv_reader, fillvalue="", )


def _create_dates_for_block(
    freq: _dates.Frequency | None,
    date_str_vector: _DataVector | None,
    /,
    start_date_only: bool = False,
    **kwargs,
) -> tuple[Dater]:
    """
    Create dates for a block
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
    """
    Create dates from a start date
    """
    #[
    num_dates = len(date_str_vector)
    start_date = _dates.Dater.from_sdmx_string(freq, date_str_vector[0], )
    return tuple(_dates.Ranger(start_date, num_dates, ))
    #]


def _create_dates_from_date_column(
    freq: _dates.Frequency | None,
    date_str_vector: _DataVector | None,
) -> tuple[Dater]:
    """
    Create dates from a date column
    """
    #[
    return tuple(
        _dates.Dater.from_sdmx_string(freq, s, ) if s else None
        for s in date_str_vector
    )
    #]

