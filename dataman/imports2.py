"""
"""


#[
from __future__ import annotations

from typing import (TypeAlias, Self, Protocol, )
from collections.abc import (Iterable, Iterator, Generator, Callable, )
import csv as cs_
import numpy as np_
import itertools as it_

from . import (dates as dd_, series as ds_, )
#]


Reader: TypeAlias = Iterator[list[str]]


def read_csv(
    file_path: str,
    /,
    **kwargs,
) -> list[_Block]:
    """
    Read a CSV file
    """
    #[
    with open(file_path, newline="", ) as file:
        reader = cs_.reader(file, )
        header_iterator = _create_header_iterator(reader, **kwargs, )
        indexed_column_generator = _IndexedDataColumnGenerator(reader, )
    return [
        b for b in _generate_blocks(header_iterator, indexed_column_generator, **kwargs, )
        if len(b.headers) > 0
    ]
    #]


def _generate_blocks(
    header_iterator: Iterator[tuple[str, str]],
    indexed_column_generator: _IndexedDataColumnGenerator,
    /,
    **kwargs,
) -> Generator[_Block, None, None]:
    """
    Generate blocks of columns of same data frequency
    """
    #[
    current_block = _Block(None, None, None, )
    for column_index, header in enumerate(header_iterator, ):
        current_block, to_yield = current_block.handle_next_column(
            column_index, header, indexed_column_generator, **kwargs,
        )
        if to_yield is not None:
            yield to_yield
        if current_block is None:
            break
    else:
        yield current_block
    #]


class _Block:
    __slots__ = ("start_column_index", "headers", "dates", )
    #[
    def __init__(
        self,
        start_column_index: int | None,
        header: tuple[str, str] | None,
        date_str_column: tuple[str] | None,
        /,
        start_date_only: bool = False,
        **kwargs,
    ) -> None:
        self.start_column_index = start_column_index
        self.headers = []
        freq = dd_.frequency_from_string(header[0]) if header is not None else None
        self.dates = (freq, date_str_column, start_date_only, )

    def handle_next_column(
        self,
        column_index: int,
        header: tuple[str, str],
        indexed_column_generator: _IndexedDataColumnGenerator,
        /,
        **kwargs,
    ) -> tuple[_Block | None, _Block | None]:
        """
        Handle next column
        """
        if _is_end_of_file(header, ):
            return (None, self, )
        elif _is_start_of_block(header, ):
            new_block = _Block(column_index, header, indexed_column_generator[column_index], **kwargs, )
            return (new_block, self, )
        else:
            self._add_column(header, )
            return (self, None, )

    def _add_column(
        self,
        header: tuple[str, str],
        /,
    ) -> None:
        """
        Add a header to the block
        """
        self.headers.append(header)
    #]


_END_OF_FILE = "__eof__"


def _is_start_of_block(
    header: tuple[str, str],
    /,
) -> bool:
    """
    Return true if the name is the start of a block
    """
    return header[0].startswith("__") and header[0] != _END_OF_FILE


def _is_end_of_file(
    header: tuple[str, str],
    /,
) -> bool:
    """
    Return true if the name is the end of a file
    """
    return header[0] == _END_OF_FILE


def _create_header_iterator(
    reader: Reader,
    /,
    skip_rows: int = 0,
    has_comment_row: bool = False,
    **kwargs,
) -> Iterator[tuple[str, str]]:
    """
    Create an iterator over the headers
    """
    #[
    for _ in range(skip_rows):
        next(reader, )
    name_row = next(reader, )
    comment_row = next(reader, ) if has_comment_row else it_.repeat("")
    return zip(name_row, comment_row, )
    #]

class _IndexedDataColumnGenerator:
    """
    A class of objects that contain a column generator, keep track of the number of columns already generated, and returns a specific column
    """
    #[
    def __init__(
        self,
        reader: Reader,
        /,
    ) -> None:
        self._column_generator = it_.zip_longest(*reader, fillvalue="", )
        self._column_count = 0

    def __getitem__(
        self,
        index: int,
        /,
    ) -> tuple:
        """
        Return the column at the specified index
        """
        if index < self._column_count:
            raise ValueError("Index must be greater than the column count")
        for _ in range(index - self._column_count):
            next(self._column_generator, )
        self._column_count = index + 1
        return next(self._column_generator, )
    #]


