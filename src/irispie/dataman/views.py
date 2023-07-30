"""
Mixin classes for displaying data management objects
"""


#[
from __future__ import annotations

from numbers import (Number, )
import numpy as np_
import re as re_

from ..dataman import dates as da_
#]


_VERTICAL_ELLIPSIS = "⋮"
_REPEAT_SHORT_ROW = 2
_REPR_MAX_LEN = 70
_REPR_CONT = "..."
_REPR_INDENT = "    "
_REPR_SEPARATOR = ": "


def _get_series_row_str_(date, data, date_str_format, numeric_format, missing_str: str):
    """
    String representing one row of time series data including the date
    """
    #[
    date_str = ("{:"+date_str_format+"}").format(date)
    value_format = "{:"+numeric_format+"}"
    data_str = "".join([value_format.format(v) for v in data])
    data_str = data_str.replace("nan", "{:>3}".format(missing_str))
    return date_str + data_str
    #]


def _databank_repr(x, /, ) -> str:
    """
    String representing one record in a databank
    """
    #[
    if x is None:
        s = "None"
    elif x is ...:
        s = "..."
    elif isinstance(x, Number) or isinstance(x, da_.Dater):
        s = str(x)
    elif isinstance(x, str):
        s = f'"{x}"'
    elif isinstance(x, np_.ndarray) or isinstance(x, list) or isinstance(x, tuple):
        s = re_.sub("\n + ", " ", repr(x))
    elif hasattr(x, "_get_first_line_view_"):
        s = x._get_first_line_view_()
    else:
        s = repr(type(x))
    return s if len(s)<_REPR_MAX_LEN else s[0:_REPR_MAX_LEN] + _REPR_CONT
    #]


class ViewMixin:
    """
    """
    _short_rows_: int = 5
    #[
    def _get_header_view_(self, /, ):
        """
        """
        return [ 
            "", 
            self._get_first_line_view_(),
            f"Description: \"{self.get_description()}\"",
            "", 
        ]

    def _get_footer_view_(self, /, ):
        return ["", ]

    def _get_view_(self, /, ):
        """
        """
        header_view = self._get_header_view_()
        content_view = self._get_content_view_()
        footer_view = self._get_footer_view_()
        return header_view + content_view + footer_view

    def __invert__(self):
        """
        ~self for short view
        """
        header_view = self._get_header_view_()
        content_view = self._get_content_view_()
        if len(content_view) > 2*self._short_rows_:
            content_view = content_view[:self._short_rows_] + [self._get_short_row_()]*_REPEAT_SHORT_ROW + content_view[-self._short_rows_:]
        print("\n".join(header_view + content_view))

    def __repr__(self, /, ):
        """
        """
        return "\n".join(self._get_view_())

    def __str__(self, /, ):
        """
        """
        return repr(self)
    #]


class SeriesViewMixin(ViewMixin):
    """
    """
    #[
    def _get_first_line_view_(self, /, ):
        """
        """
        shape = self.data.shape
        return f"Series {self.frequency.letter} {self.start_date}:{self.end_date} {shape[0]}-by-{shape[1]}"

    def _get_content_view_(self, /, ):
        """
        """
        return [
            _get_series_row_str_(date, data_row, self._date_str_format, self._numeric_format, self._missing_str) 
            for row, (date, data_row) in enumerate(zip(self.range, self.data))
        ]

    def _get_short_row_(self):
        """
        Create string representing rows skipped in the short view of times series
        """
        shape = self.data.shape
        date_str = ("{:"+self._date_str_format+"}").format(_VERTICAL_ELLIPSIS)
        value_format = "{:"+self._short_str_format+"}"
        data_str = "".join([value_format.format(_VERTICAL_ELLIPSIS)] * shape[1])
        return date_str + data_str
    #]


class DatabankViewMixin(ViewMixin):
    """
    """
    #[
    def _get_first_line_view_(self, /, ):
        """
        """
        return f"Databank with {self._get_num_records():g} record(s)"

    def _get_content_view_(self, /, ):
        """
        """
        names = self._get_names()
        content = []
        if names:
            max_len = self._get_max_name_length_()
            content = [ _REPR_INDENT + str(k).rjust(max_len) + _REPR_SEPARATOR + _databank_repr(getattr(self, k)) for k in names ]
            content = content
        return content

    def _get_max_name_length_(self, /, ):
        """
        """
        return max(len(str(k)) for k in self._get_names())

    def _get_short_row_(self):
        """
        """
        max_len = self._get_max_name_length_()
        return _REPR_INDENT + _VERTICAL_ELLIPSIS.rjust(max_len) + " "*len(_REPR_SEPARATOR) + _VERTICAL_ELLIPSIS #]

