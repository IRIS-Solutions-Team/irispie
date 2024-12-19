"""
Time series indexing inlay
"""


#[

from __future__ import annotations

import documark as _dm

from .. import dates as _dates

#]


class Inlay:
    """
    ................................................................................
    ==Inlay Class for Time Series Indexing==

    Provides a flexible interface for indexing and manipulating time series objects. 
    This class supports operations like time shifting, data extraction, assignment, 
    and recreation.

    ### Purpose ###
    The `Inlay` class serves as an abstraction layer for implementing and managing 
    indexing logic in time series objects, ensuring consistent behavior across 
    different indexing use cases.

    ### Key Features ###
    - Time series **shifting** by specific periods.
    - **Data extraction** for specific dates and variants.
    - **Data assignment** for modifying existing time series.
    - Recreation of new time series objects based on indexing.

    ### Methods ###
    The `Inlay` class implements the following key methods:
    - `__getitem__`: Enables data retrieval using indexing.
    - `__setitem__`: Allows data assignment using indexing.
    - `__call__`: Facilitates the creation of new series objects via callable syntax.
    - `indexing`: Reference implementation for supported indexing operations.

    ### Notes ###
    - This class provides the underlying functionality for indexing in time series 
      objects, enabling a natural and Pythonic interface.

    ................................................................................
    """
    #[


    @_dm.reference(
        category=None,
        call_name="Time series indexing",
        call_name_is_code=False,
        priority=30,
    )
    def indexing(self, /, ) -> None:
        """
················································································

Time `Series` objects can be indexed in four ways (note the square versus round
brackets):

| Indexing                                           | Description
|----------------------------------------------------|-------------
| `self[shift]`                                      | Time shift
| `self[dates]`, `self[dates, variants]`             | Data extraction
| `self[dates] = ...`, `self[dates, variants] = ...` | Data assignment
| `self(dates)`, `self(dates, variants)`             | Time `Series` recreation


### Time shift ###

```
self[shift]
```

Time shift is done by passing an integer to the `self[shift]` or
`self[shift]` indexing. The time shift syntax returns a new copy of the
original series, with the time periods shifted by `shift`.


### Data extractation ###

```
self[dates]
self[dates, variants]
```

The `dates` is a `Dater` or a tuple of `Daters` or a time `Span` object,
and `variants` is an integer or a tuple of integers or a `slice` object
specifying the variants. The data extraction syntax returns a
two-dimensional `numpy` array, with the time dimension running along the
rows and the variant dimension running along the columns.


### Data assignment ###

```
self[dates] = ...
self[dates, variants] = ...
```

The `dates` is a `Dater` or a tuple of `Daters` or a time `Span` object,
and `variants` is an integer or a tuple of integers or a `slice` object
specifying the variants. The data assignment syntax sets the data in the
time series.


### Time `Series` recreation ###

```
self(dates)
self(dates, variants)
```

The `dates` is a `Dater` or a tuple of `Daters` or a time `Span` object,
and `variants` is an integer or a tuple of integers or a `slice` object
specifying the variants. The time `Series` recreation syntax returns a new
time `Series` object based on the data selected by the `dates` and
`variants`.

················································································
        """
        raise NotImplementedError

    def __getitem__(
        self,
        index: int | tuple,
        /,
    ) -> _np.ndarray:
        """
    ................................................................................
    ==Data Retrieval Using Indexing==

    Retrieves data from the time series based on the provided index. Supports both 
    time shifting and data extraction.

    ### Input Arguments ###
    ???+ input "index"
        - **int**: Specifies a time shift.
        - **tuple**: Specifies `dates` and optionally `variants` for data extraction.

    ### Returns ###
    ???+ returns "numpy.ndarray"
        - **Time Shift**: A new time series object with shifted time periods.
        - **Data Extraction**: A 2D numpy array containing the extracted data.

    ### Example ###
    ```python
        shifted_series = series[3]  # Time shift
        data = series[(dates, variants)]  # Data extraction
    ```

    ### Notes ###
    - For time shifts, a new series object is created with the specified shift applied.
    - For data extraction, the `get_data` method is called with the provided dates and 
      variants.

    ................................................................................

        """
        if isinstance(index, int, ):
            # Time shift
            new = self.copy()
            new.shift(index, )
            return new
        else:
            # Extract data
            if isinstance(index, tuple, ):
                dates, variants = index
            else:
                dates, variants = index, None
            # dates = _dates.ensure_period_tuple(dates, frequency=self.frequency, )
            return self.get_data(dates, variants, )

    def __setitem__(
        self,
        index: int | tuple,
        data,
        /,
    ) -> None:
        """
            ................................................................................
    ==Data Assignment Using Indexing==

    Assigns data to specific periods and variants in the time series.

    ### Input Arguments ###
    ???+ input "index"
        - **int**: Specifies a single time period for assignment.
        - **tuple**: Specifies `dates` and optionally `variants` for assignment.

    ???+ input "data"
        The data to assign. Must match the shape of the specified `dates` and 
        `variants`.

    ### Returns ###
    ???+ returns "None"
        Modifies the time series in place with the assigned data.

    ### Example ###
    ```python
        series[dates] = new_data  # Assign data to specific dates
        series[(dates, variants)] = new_data  # Assign data to dates and variants
    ```

    ### Notes ###
    - If the index is not a tuple, it is converted to `(index, None)` for compatibility.
    - The `set_data` method is called internally for data assignment.

    ................................................................................
        """
        if not isinstance(index, tuple):
            index = (index, None, )
        return self.set_data(index[0], data, index[1], )

    def __call__(
        self,
        *index,
    ) -> Self:
        """
            ................................................................................
    ==Time Series Recreation==

    Creates a new `Series` object based on the specified `dates` and `variants`.

    ### Input Arguments ###
    ???+ input "*index"
        The indexing arguments (`dates` and optionally `variants`) used to recreate 
        the series.

    ### Returns ###
    ???+ returns "Self"
        A new `Series` object containing the selected data.

    ### Example ###
    ```python
        new_series = series(dates, variants)
        print(new_series)
    ```

    ### Notes ###
    - This method is used for creating a new `Series` object rather than modifying 
      the current object.
    - Internally, the `_get_data_and_recreate` method is used for recreation.

    ................................................................................
        """
        return self._get_data_and_recreate(*index, )

    #]

