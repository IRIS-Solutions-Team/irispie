"""
Time series indexing inlay
"""


#[
from __future__ import annotations

from .. import dates as _dates
#]


class Inlay:
    """
    """
    #[

    def __getitem__(
        self,
        index: int | tuple,
        /,
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
            dates = _dates.ensure_date_tuple(dates, frequency=self.frequency, )
            return self.get_data(dates, variants, )

    def __setitem__(
        self,
        index: int | tuple,
        data,
        /,
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

