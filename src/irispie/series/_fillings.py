"""
"""


#[
from __future__ import annotations
#]


class FillingMixin:
    """
    """
    def fill_missing(
        self,
        fill_range: Iterable[Dater],
        method: Literal["next", "previous", "nearest", "constant"],
        *args,
        /,
    ) -> None:
        """
        """
        fill_func = _METHOD_FACTORY[method]
        data = self.get_data(fill_range, )
        new_data = tuple(
            fill_func(column, method, *args, ).T
            for column in data.T
        )
        self.set_data(fill_range, new_data, )



_METHOD_FACTORY = {
    "next": _fill_next,
    "previous": _fill_previous,
    "nearest": _fill_nearest,
    "constant": _fill_constant,
}

