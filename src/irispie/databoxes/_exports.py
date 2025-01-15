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
    r"""
................................................................................
==Data Export Block==

The `_ExportBlock` class organizes and formats data from a `Databox` for 
CSV-like output. It supports configurations for series descriptions, numeric 
formatting, and custom delimiters.

Attributes are used to manage data representation, including handling periods, 
names, and formatting options.

### Input arguments ###
???+ input "databox"
The `Databox` instance containing the data for export.

???+ input "frequency"
The data frequency for the current export block.

???+ input "periods"
A tuple of periods defining the time span of the export block.

???+ input "names"
A tuple of series names included in the export block.

???+ input "total_num_data_rows"
The total number of rows of data to include.

???+ input "description_row"
Include a description row if `True`.

???+ input "delimiter"
The string delimiter for separating fields in the output.

???+ input "numeric_format"
Format specifier for numeric values.

???+ input "nan_str"
String representation for `NaN` values.

???+ input "round"
Number of decimal places to round numeric values.

???+ input "date_formatter"
Function to format date values for export.

### Returns ###
???+ returns
None: This class prepares data for iteration or export.

### Example ###
```python
export_block = _ExportBlock(
    databox=my_databox,
    frequency=Frequency.MONTHLY,
    periods=(Period(...), ...),
    names=("Series1", "Series2"),
    total_num_data_rows=10,
    description_row=True,
    delimiter=",",
    numeric_format="g",
    nan_str="",
    round=2,
    date_formatter=str
)
```
................................................................................
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
        r"""
................................................................................
==Iterate through export blocks==

Generates rows formatted for CSV output, including metadata, 
descriptions, and data values based on the configuration options.

### Input arguments ###
???+ input
    None

### Returns ###
???+ returns
    `Iterable`: Rows formatted for CSV output as tuples.

### Example ###
```python
    for row in export_block:
        print(row)
```
................................................................................
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
    r"""
................................................................................
==CSV Export Inlay==

Provides functionality to export `Databox` time series to CSV files. Includes 
metadata and descriptive rows as needed, with custom formatting options for 
delimiters, missing values, and numeric precision.

### Example ###
```python
    inlay = Inlay()
    export_info = inlay.to_csv("output.csv")
```
................................................................................
    """
    #[
    @_dm.reference(
        category="import_export",
    )
    def to_csv(
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
................................................................................
==Write Databox time series to a CSV file==

Exports time series data from a `Databox` to a structured CSV file. 
Supports custom formatting for metadata, numeric values, and output structure.

### Input arguments ###
???+ input "file_name"
    Name of the CSV file to write.

???+ input "span"
    Time span for exporting data.

???+ input "frequency_span"
    Mapping of frequencies to date ranges.

???+ input "names"
    Series names to export.

???+ input "description_row"
    Include a description row if `True`.

???+ input "frequency"
    Data frequency for export.

???+ input "numeric_format"
    Format for numeric values.

???+ input "nan_str"
    Representation for missing values.

???+ input "delimiter"
    Column delimiter in the CSV.

???+ input "round"
    Decimal rounding for numeric values.

???+ input "date_formatter"
    Formatter function for date values.

???+ input "csv_writer_settings"
    Additional CSV writer options.

???+ input "when_empty"
    Action when no data is available ("error", "warning", or "silent").

???+ input "return_info"
    Return metadata about the export if `True`.

### Returns ###
???+ returns
    Metadata about the exported data, including series names.

### Example ###
```python
    export_info = inlay.to_csv("data.csv")
```
................................................................................
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

    @_dm.reference(category="TBD", )
    def to_sheet(self, *args, **kwargs, ):
        r"""
        ................................................................................
        ==Alias for `to_csv`==

        This method serves as an alias for `to_csv`, offering identical functionality
        to write Databox time series to a CSV file.

        ### Input arguments ###
        ???+ input
            Accepts all parameters of the `to_csv` method.

        ### Returns ###
        ???+ returns
            Same return values as the `to_csv` method.

        ### Example ###
        ```python
            export_info = inlay.to_sheet("data.csv")
        ```
        ................................................................................
        """
        return self.to_csv(*args, **kwargs, )

    @_dm.reference(category="TBD", )
    def to_pickle(
        self,
        file_name: str,
        /,
        **kwargs,
    ) -> None:
        r"""
................................................................................
==Serialize Databox to a pickle file==

Saves the current `Databox` instance to a pickle file. This allows for 
data and state persistence, which can later be restored by deserializing 
the file.

### Input arguments ###
???+ input "file_name"
    The name of the file to write the pickle data.

???+ input "kwargs"
    Additional arguments passed to the `pickle.dump` function.

### Returns ###
???+ returns
    `None`: This method writes data to the specified pickle file.

### Example ###
```python
    inlay.to_pickle("data.pkl")
```
................................................................................
        """
        with open(file_name, "wb+") as fid:
            _pk.dump(self, fid, **kwargs, )
    #]


def _get_frequency_mark(frequency, ):
    """
................................................................................
==Retrieve Frequency Mark==

Get a unique mark string for the given frequency.

The mark helps identify frequency-specific data in the exported CSV.

### Input arguments ###
???+ input "frequency"
    The `Frequency` object representing the data frequency.

### Returns ###
???+ returns
    `str`: A unique string representing the frequency.

### Example ###
```python
    mark = _get_frequency_mark(Frequency.MONTHLY)
    print(mark)  # Output: "__monthly__"
```
................................................................................
    """
    return "__" + frequency.name.lower() + "__"


def _get_names_to_export(databox, frequency, names, ):
    """
................................................................................
==Retrieve Series Names to Export==

Validate and retrieve the series names to export for a specific frequency.

If no names are provided, defaults to all available names for the frequency.

### Input arguments ###
???+ input "databox"
    The `Databox` instance containing series data.

???+ input "frequency"
    The frequency for which to retrieve series names.

???+ input "names"
    Optional. Specific series names to validate and export.

### Returns ###
???+ returns
    `list[str]`: A list of valid series names for the frequency.

### Example ###
```python
    names = _get_names_to_export(my_databox, Frequency.MONTHLY, ["Sales", "Revenue"])
    print(names)
```
................................................................................
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
................................................................................
==Resolve Frequency Span==

Validate and expand the frequency span into usable tuples of periods.

This function ensures that the input spans are consistent and usable
for data export.

### Input arguments ###
???+ input "databox"
    The `Databox` instance containing series data.

???+ input "frequency_span"
    A mapping of frequencies to their respective periods.

???+ input "span"
    Optional. An iterable of periods overriding the frequency span.

### Returns ###
???+ returns
    `tuple[dict[Frequency, tuple[Period]], int]`: A tuple containing the 
    resolved frequency span and the count of unique frequencies.

### Example ###
```python
    frequency_span, count = _resolve_frequency_span(
        databox=my_databox,
        frequency_span={Frequency.MONTHLY: [..., ...]},
    )
```
................................................................................
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
................................................................................
==Resolve Frequency Names==

Fetch series names for each frequency in the span.

This function creates a dictionary mapping frequencies to tuples of
series names available in the `Databox`.

### Input arguments ###
???+ input "databox"
    The `Databox` containing the series data.

???+ input "frequency_span"
    A dictionary mapping frequencies to their respective date ranges.

### Returns ###
???+ returns
    `dict[Frequency, tuple[str, ...]]`: A mapping of frequencies to 
    tuples of series names.

### Example ###
```python
    names_by_frequency = _resolve_frequency_names(my_databox, frequency_span)
```
................................................................................
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
................................................................................
==Calculate Total Data Rows==

Determine the maximum number of data rows required across all frequencies.

This function computes the highest count of periods for any frequency
within the provided frequency span.

### Input arguments ###
???+ input "frequency_span"
    A dictionary mapping frequencies to tuples of periods.

### Returns ###
???+ returns
    `int`: The maximum count of rows required for export.

### Example ###
```python
    total_rows = _get_total_num_data_rows(frequency_span)
    print(total_rows)
```
................................................................................
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
................................................................................
==Fetch Descriptions for Names==

Retrieve series descriptions for the given series names.

This function queries the `Databox` to fetch descriptions associated 
with each series name.

### Input arguments ###
???+ input "self"
    The current `Databox` instance.

???+ input "names"
    An iterable of series names to fetch descriptions for.

### Returns ###
???+ returns
    `tuple[str, ...]`: A tuple of series descriptions.

### Example ###
```python
    descriptions = _get_descriptions_for_names(my_databox, ["Sales", "Profit"])
    print(descriptions)
```
................................................................................
    """
    return tuple(self[n].get_description() for n in names)


def _get_num_data_columns_for_names(
    self,
    names: Iterable[str],
    /,
) -> tuple[int, ...]:
    """
................................................................................
==Retrieve Number of Data Columns==

Determine the number of data columns for the given series names.

This function calculates the column count for each series name's data 
in the `Databox`.

### Input arguments ###
???+ input "self"
    The current `Databox` instance.

???+ input "names"
    An iterable of series names for which to retrieve column counts.

### Returns ###
???+ returns
    `tuple[int, ...]`: A tuple containing the number of data columns 
    for each series.

### Example ###
```python
    column_counts = _get_num_data_columns_for_names(my_databox, ["Sales", "Profit"])
    print(column_counts)
```
................................................................................
    """
    return tuple(self[n].shape[1] for n in names)


def _get_data_array_for_names(
    self,
    names: Iterable[str],
    periods: Iterable[Period],
    /,
) -> _np.ndarray:
    """
................................................................................
==Fetch Data Arrays==

Retrieve the data arrays for specified series names and periods.

Aggregates data for all specified series into a NumPy array, aligning 
with the provided periods.

### Input arguments ###
???+ input "self"
    The current `Databox` instance.

???+ input "names"
    An iterable of series names to fetch data for.

???+ input "periods"
    An iterable of periods for which to fetch data.

### Returns ###
???+ returns
    `_np.ndarray`: A NumPy array containing the aggregated data.

### Example ###
```python
    data = _get_data_array_for_names(my_databox, ["Sales"], [Period(...), ...])
    print(data)
```
................................................................................
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
................................................................................
==Handle Empty Data Cases==

Manage scenarios where no data is available for export.

Depending on the `when_empty` parameter, this function raises an error,
logs a warning, or silently skips further processing.

### Input arguments ###
???+ input "frequency_span"
    A dictionary mapping frequencies to tuples of periods.

???+ input "frequency_names"
    A dictionary mapping frequencies to tuples of series names.

???+ input "when_empty"
    Defines the behavior when no data is available. Can be:
    - `"error"`: Raises an exception.
    - `"warning"`: Logs a warning.
    - `"silent"`: Silently ignores the empty case.

???+ input "file_name"
    The name of the file intended for data export.

### Returns ###
???+ returns
    `None`: The behavior depends on the value of `when_empty`.

### Example ###
```python
    _catch_empty(frequency_span, frequency_names, "warning", "data.csv")
```
................................................................................
    """
    #[
    if not frequency_span or all(len(v) == 0 for v in frequency_names.values()):
        _wrongdoings.raise_as(when_empty, f"No data exported to {file_name}", )
    #]