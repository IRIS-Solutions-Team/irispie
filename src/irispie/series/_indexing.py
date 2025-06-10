"""
Time series indexing inlay
"""


#[

from __future__ import annotations

import documark as _dm

from .. import dates as _dates
from ._categories import CATEGORIES

#]


class Inlay:
    """
    """
    #[

    @_dm.reference(
        category=None,
        call_name=CATEGORIES["indexing"],
        call_name_is_code=False,
        priority=30,
    )
    def indexing(self, /, ) -> None:
        r"""
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
    ) -> _np.ndarray:
        """
        Get data self[dates] or self[dates, variants]
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
    ) -> None:
        """
        Set data self[dates] = ... or self[dates, variants] = ...
        """
        if not isinstance(index, tuple):
            index = (index, None, )
        return self.set_data(index[0], data, index[1], )

    def __call__(
        self,
        *index,
    ) -> Self:
        """
        Create a new time series based on date retrieved by self[dates] or self[dates, variants]
        """
        return self._get_data_and_recreate(*index, )

    #]

