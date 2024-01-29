"""
Time series indexing inlay
"""


#[
from __future__ import annotations
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
        # Time shift
        if isinstance(index, int, ):
            new = self.copy()
            new.shift(index, )
            return new
        # Extracting data
        if not isinstance(index, tuple, ):
            index = (index, None, )
        return self.get_data(*index, )

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

