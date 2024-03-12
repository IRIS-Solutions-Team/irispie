"""
Exporting data to CSV sheets
"""


#[
from __future__ import annotations

import pickle as _pk

from collections.abc import (Iterable, )
from typing import (Any, Callable, )
import csv as _cs
import numpy as _np
import itertools as _it
import dataclasses as _dc
import functools as _ft

from .. import pages as _pages
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from ..databoxes import main as _databoxes
#]


_DEFAULT_ROUND = 12
_DEFAULT_DATE_FORMATTER = str


@_dc.dataclass
class _ExportBlock:
    """
    """
    #[

    databox: _databoxes.Databox | None = None
    frequency: _dates.Frequency | None = None
    dates: tuple[_dates.Dater] | None = None
    names: tuple[str] | None = None
    #
    # Options
    total_num_data_rows: int | None = None
    description_row: bool | None = None
    delimiter: str | None = None
    numeric_format: str | None = None
    nan_str: str | None = None
    round: int | None = None
    date_formatter: Callable | None = None

    def __iter__(self, ):
        """
        """
        def _round(x, /, ):
            if self.round is None:
                return x
            else:
                return _np.round(x, self.round)
        #
        descriptions = _get_descriptions_for_names(self.databox, self.names, )
        num_data_columns = _get_num_data_columns_for_names(self.databox, self.names, )
        data_array = _get_data_array_for_names(self.databox, self.names, self.dates, )
        #
        # Empty row
        empty_row = ("", ) + ("", )*sum(num_data_columns) + ("", )
        #
        # Name row
        name_row = tuple(_it.chain.from_iterable( 
            (n, ) + ("*", )*(num_data_columns[i] - 1)
            for i, n in enumerate(self.names, )
        ))
        yield (_get_frequency_mark(self.frequency), ) + name_row + ("", )
        #
        # Description row
        if self.description_row:
            description_row = tuple(_it.chain.from_iterable( 
                (n, ) + ("*", )*(num_data_columns[i] - 1)
                for i, n in enumerate(descriptions, )
            ))
            yield ("", ) + description_row + ("", )
        #
        # Dates and data, row by row
        date_formatter = self.date_formatter or _DEFAULT_DATE_FORMATTER
        for date, data_row in zip(self.dates, data_array, ):
            yield \
                (date_formatter(date, ), ) \
                + tuple(x if not _np.isnan(x) else self.nan_str for x in _round(data_row).tolist()) \
                + ("", )
        #
        # Empty rows afterwards
        for _ in range(self.total_num_data_rows - len(self.dates), ):
            yield empty_row

    #]


class ExportMixin:
    """
    Databox mixin for exporting data to disk files
    """
    #[
    @_pages.reference(category="import_export", )
    def to_sheet(
        self,
        file_name: str,
        *,
        frequency_span: dict | None = None,
        names: Iterable[str] | None = None,
        description_row: bool = False,
        frequency: _dates.Frequency | None = None,
        numeric_format: str = "g",
        nan_str: str = "",
        delimiter: str = ",",
        round: int = _DEFAULT_ROUND,
        date_formatter: Callable | None = None,
        csv_writer_settings: dict | None = {},
        when_empty: Literal["error", "warning", "silent"] = "warning",
    ) -> dict[str, Any]:
        """
················································································

==Export `Databox` to a CSV or spreadsheet file==

················································································
        """
        databox = self.shallow(source_names=names, )
        frequency_span = frequency_span if frequency_span is not None else _DEFAULT_FREQUENCY_SPAN
        frequency_span = _resolve_frequency_span(databox, frequency_span, )
        frequency_names = _resolve_frequency_names(databox, frequency_span, )
        _catch_empty(frequency_span, frequency_names, when_empty, file_name, )
        #
        total_num_data_rows = _get_total_num_data_rows(frequency_span, )
        export_block_constructor = _ft.partial(
            _ExportBlock,
            databox=databox,
            total_num_data_rows=total_num_data_rows,
            description_row=description_row,
            delimiter=delimiter,
            numeric_format=numeric_format,
            nan_str=nan_str,
            round=round,
            date_formatter=date_formatter,
        )
        export_blocks = (
            export_block_constructor(
                frequency=f,
                dates=frequency_span[f],
                names=frequency_names[f],
            )
            for f in frequency_span.keys()
            if frequency_names[f]
        )
        #
        csv_writer_settings = csv_writer_settings or {}
        with open(file_name, "w+") as fid:
            writer = _cs.writer(fid, delimiter=delimiter, lineterminator="\n", **csv_writer_settings, )
            for row in zip(*export_blocks, ):
                writer.writerow(_it.chain.from_iterable(row))
        #
        info = {
            "names_exported": tuple(_it.chain.from_iterable(frequency_names.values())),
        }
        #
        return info

    def to_pickle(
        self,
        file_name: str,
        /,
        **kwargs,
    ) -> None:
        """
        """
        with open(file_name, "wb+") as fid:
            _pk.dump(self, fid, **kwargs, )
    #]


def _get_frequency_mark(frequency, ):
    return "__" + frequency.name.lower() + "__"


def _get_names_to_export(databox, frequency, names, ):
    """
    """
    #[
    names_from_databox = self.get_series_names_by_frequency(frequency)
    names = names_from_databox if names is None else names
    return [n for n in names if n in names_from_databox]
    #]


_DEFAULT_FREQUENCY_SPAN = {
    _dates.Frequency.YEARLY: ...,
    _dates.Frequency.HALFYEARLY: ...,
    _dates.Frequency.QUARTERLY: ...,
    _dates.Frequency.MONTHLY: ...,
    _dates.Frequency.DAILY: ...,
    _dates.Frequency.INTEGER: ...,
    _dates.Frequency.UNKNOWN: ...,
}


def _resolve_frequency_span(
    databox: _databoxes.Databox,
    frequency_span: dict[_dates.Frequency | int: Iterable[_dates.Dater]],
    /,
) -> tuple[dict[_dates.Frequency: tuple[_dates.Dater]], int]:
    """
    """
    #[
    # Remove Nones, ensure Frequency objects
    frequency_span = {
        _dates.Frequency(k): v
        for k, v in frequency_span.items()
        if v is not None
    }
    #
    # Resolve date span for each ...
    frequency_span = {
        k: v if v is not ... else databox.get_span_by_frequency(k, )
        for k, v in frequency_span.items()
    }
    #
    # Expand date spans into tuples, remove empty spans
    frequency_span = {
        k: tuple(i for i in v)
        for k, v in frequency_span.items()
        if v is not _dates.EmptyRanger() or k is _dates.Frequency.UNKNOWN
    }
    #
    return frequency_span
    #]


def _resolve_frequency_names(
    databox: _databoxes.Databox,
    frequency_span: _dates.Frequency | None,
    /,
) -> dict[_dates.Frequency: tuple[str, ...]]:
    """
    """
    #[
    return {
        k: tuple(databox.get_series_names_by_frequency(k, ))
        for k in frequency_span.keys()
    }


def _get_total_num_data_rows(
    frequency_span: dict[_dates.Frequency: tuple[_dates.Dater]],
    /,
) -> int:
    """
    """
    #
    # Find maximum number of data rows
    return max((len(v) for v in frequency_span.values()), default=0, )


def _get_descriptions_for_names(
    self,
    names: Iterable[str],
    /,
) -> tuple[str, ...]:
    """
    """
    return tuple(self[n].get_description() for n in names)


def _get_num_data_columns_for_names(
    self,
    names: Iterable[str],
    /,
) -> tuple[int, ...]:
    """
    """
    return tuple(self[n].shape[1] for n in names)


def _get_data_array_for_names(
    self,
    names: Iterable[str],
    dates: Iterable[_dates.Dater],
    /,
) -> _np.ndarray:
    """
    """
    empty_lead = _np.empty((len(dates), 0, ), dtype=_np.float64, )
    return _np.hstack([empty_lead] + [self[n].get_data(dates, ) for n in names], )


def _catch_empty(
    frequency_span: dict[_dates.Frequency: tuple[_dates.Dater]],
    frequency_names: dict[_dates.Frequency: tuple[str, ...]],
    when_empty: Literal["error", "warning", "silent"],
    file_name: str,
    /,
) -> None:
    """
    """
    #[
    if not frequency_span or all(len(v) == 0 for v in frequency_names.values()):
        _wrongdoings.raise_as(when_empty, f"No data exported to {file_name}", )
    #]

