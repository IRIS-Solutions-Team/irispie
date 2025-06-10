"""
Exporting data to CSV sheets
"""


#[

from __future__ import annotations

from typing import (TYPE_CHECKING, )
import pickle as _pk
import csv as _cs
import numpy as _np
import itertools as _it
import dataclasses as _dc
import functools as _ft
import documark as _dm

from ..dates import (Frequency, Period, EmptySpan, )
from .. import wrongdoings as _wrongdoings

if TYPE_CHECKING:
    from collections.abc import (Iterable, )
    from typing import (Any, Callable, )
    from ..databoxes.main import (Databox, )

#]


_DEFAULT_ROUND = 12
_DEFAULT_DATE_FORMATTER = str


@_dc.dataclass
class _ExportBlock:
    """
    """
    #[

    databox: Databox | None = None
    frequency: Frequency | None = None
    periods: tuple[Period] | None = None
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
        data_array = _get_data_array_for_names(self.databox, self.names, self.periods, )
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
        for date, data_row in zip(self.periods, data_array, ):
            yield \
                (date_formatter(date, ), ) \
                + tuple(x if not _np.isnan(x) else self.nan_str for x in _round(data_row).tolist()) \
                + ("", )
        #
        # Empty rows afterwards
        for _ in range(self.total_num_data_rows - len(self.periods), ):
            yield empty_row

    #]


class Inlay:
    """
    Databox inlay for writing databox time series to CSV file
    """
    #[

    @_dm.reference(category="import_export", )
    def to_csv_file(
        self,
        file_name: str,
        *,
        span: Iterable[Period] | None = None,
        frequency_span: dict | None = None,
        names: Iterable[str] | None = None,
        description_row: bool = False,
        frequency: Frequency | None = None,
        numeric_format: str = "g",
        nan_str: str = "",
        delimiter: str = ",",
        round: int = _DEFAULT_ROUND,
        date_formatter: Callable | None = None,
        csv_writer_settings: dict | None = {},
        when_empty: Literal["error", "warning", "silent"] = "warning",
        return_info: bool = False,
    ) -> dict[str, Any]:
        r"""
················································································

==Write Databox time series to a CSV file==


    self.to_csv_file(
        file_name,
        *,
        frequency_span=None,
        names=None,
        description_row=False,
        frequency=None,
        numeric_format="g",
        nan_str="",
        delimiter=",",
        round=None,
        date_formatter=None,
        csv_writer_settings={},
        when_empty="warning",
    )


### Input arguments ###


???+ input "file_name"
    Name of the CSV file where the data will be written.

???+ input "frequency_span"
    Specifies the frequencies and their corresponding date ranges for exporting
    data. If `None`, exports data for all available frequencies and their full
    date ranges in the databox.

???+ input "names"
    A list of series names to export. If `None`, exports all series for the 
    specified frequencies.

???+ input "description_row"
    If `True`, include a row of series descriptions in the CSV.

???+ input "frequency"
    Frequency of the data to export.

???+ input "numeric_format"
    The numeric format for data values, e.g., 'g', 'f', etc.

???+ input "nan_str"
    String representation for NaN values in the output.

???+ input "delimiter"
    Character to separate columns in the CSV file.

???+ input "round"
    Number of decimal places to round numeric values.

???+ input "date_formatter"
    Function to format date values. If `None`, SDMX string formatter is used.

???+ input "csv_writer_settings"
    Additional settings for the CSV writer.

???+ input "when_empty"
    Behavior when no data is available for export. Can be "error", "warning", or
    "silent".


### Returns ###


???+ returns "info"
    A dictionary with details about the export:

    * `names_exported`: Names of the series exported to the CSV file.


················································································
        """
        databox = self.shallow(source_names=names, )
        frequency_span = _resolve_frequency_span(databox, frequency_span, span, )
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
                periods=frequency_span[f],
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
        if return_info:
            return info
        else:
            return

    @_dm.no_reference
    def to_sheet(self, *args, **kwargs, ):
        r"""
        """
        return self.to_csv(*args, **kwargs, )

    # Legacy alias
    to_csv = to_csv_file

    @_dm.reference(category="import_export", )
    def to_pickle_file(
        self,
        file_name: str,
        **kwargs,
    ) -> None:
        r"""
................................................................................

==Write Databox to a pickle file==

    self.to_pickle(
        file_name,
        **kwargs, 
    )

### Input arguments ###

???+ input "file_name"
    Path to the pickle file where the data will be written.

???+ input "kwargs"
    Additional keyword arguments for the pickle writer.

### Returns ###

This method returns `None`.

................................................................................
        """
        with open(file_name, "wb+") as fid:
            _pk.dump(self, fid, **kwargs, )

    to_pickle = to_pickle_file

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
    Frequency.YEARLY: ...,
    Frequency.HALFYEARLY: ...,
    Frequency.QUARTERLY: ...,
    Frequency.MONTHLY: ...,
    Frequency.DAILY: ...,
    Frequency.INTEGER: ...,
    Frequency.UNKNOWN: ...,
}


def _resolve_frequency_span(
    databox: Databox,
    frequency_span: dict[Frequency | int: Iterable[Period]],
    span: Iterable[Period] | None = None,
    /,
) -> tuple[dict[Frequency: tuple[Period]], int]:
    """
    """
    #[
    # Argument span overrides frequency_span
    if span is None:
        frequency_span = (
            frequency_span
            if frequency_span is not None
            else _DEFAULT_FREQUENCY_SPAN
        )
    else:
        span = tuple(span)
        frequency = span[0].frequency
        frequency_span = {frequency: span}
    # Remove Nones, ensure Frequency objects
    frequency_span = {
        Frequency(k): v
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
        if v is not EmptySpan() or k is Frequency.UNKNOWN
    }
    #
    return frequency_span
    #]


def _resolve_frequency_names(
    databox: Databox,
    frequency_span: Frequency | None,
    /,
) -> dict[Frequency: tuple[str, ...]]:
    """
    """
    #[
    return {
        k: tuple(databox.get_series_names_by_frequency(k, ))
        for k in frequency_span.keys()
    }


def _get_total_num_data_rows(
    frequency_span: dict[Frequency: tuple[Period]],
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
    periods: Iterable[Period],
    /,
) -> _np.ndarray:
    """
    """
    empty_lead = _np.empty((len(periods), 0, ), dtype=_np.float64, )
    return _np.hstack([empty_lead] + [self[n].get_data(periods, ) for n in names], )


def _catch_empty(
    frequency_span: dict[Frequency: tuple[Period]],
    frequency_names: dict[Frequency: tuple[str, ...]],
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

