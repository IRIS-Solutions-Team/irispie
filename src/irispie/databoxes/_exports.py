"""
Exporting data to CSV sheets
"""


#[
from __future__ import annotations

import pickle as _pk

from collections.abc import (Iterable, )
from typing import (Any, )
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
        for date, data_row in zip(self.dates, data_array, ):
            yield \
                (str(date), ) \
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
        round: int = _DEFAULT_ROUND,
        delimiter: str = ",",
        csv_writer_settings: dict | None = {},
        when_empty: Literal["error", "warning", "silent"] = "warning",
    ) -> dict[str, Any]:
        """
················································································

==Export `Databox` to a CSV or spreadsheet file==

················································································
        """
        frequency_span = _resolve_frequency_span(self, frequency_span, )
        frequency_names = _resolve_frequency_names(self, frequency_span, names, )
        _catch_empty(frequency_span, frequency_names, when_empty, file_name, )
        #
        total_num_data_rows = _get_total_num_data_rows(frequency_span, )
        export_blocks = (
            _ExportBlock(
                self, frequency, frequency_span[frequency], frequency_names[frequency],
                total_num_data_rows, description_row, delimiter, numeric_format, nan_str, round,
            )
            for frequency in frequency_span.keys()
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


_DEFAULT_FREQUENCY_RANGE = {
    _dates.Frequency.YEARLY: ...,
    _dates.Frequency.HALFYEARLY: ...,
    _dates.Frequency.QUARTERLY: ...,
    _dates.Frequency.MONTHLY: ...,
    _dates.Frequency.DAILY: ...,
    _dates.Frequency.INTEGER: ...,
    _dates.Frequency.UNKNOWN: ...,
}


def _resolve_frequency_span(
    self,
    frequency_span: dict[_dates.Frequency | int: Iterable[_dates.Dater]] | None,
    /,
) -> tuple[dict[_dates.Frequency: tuple[_dates.Dater]], int]:
    """
    """
    #[
    # Remove Nones, ensure Frequency objects
    frequency_span = {
        _dates.Frequency(k): v
        for k, v in (frequency_span or _DEFAULT_FREQUENCY_RANGE).items()
        if v is not None
    }
    #
    # Resolve date span for each ...
    frequency_span = {
        k: v if v is not ... else self.get_span_by_frequency(k, )
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
    self,
    frequency_span: _dates.Frequency | None,
    custom_names: Iterable[str] | None,
    /,
) -> dict[_dates.Frequency: tuple[str, ...]]:
    """
    """
    #[
    #
    # Get all databox names for each frequency
    frequency_names = {
        k: tuple(self.get_series_names_by_frequency(k, ))
        for k in frequency_span.keys()
    }
    #
    # Filter names against custom names
    frequency_names = {
        k: tuple(n for n in v if n in custom_names)
        for k, v in frequency_names.items()
    } if custom_names is not None else frequency_names
    #
    return frequency_names


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

