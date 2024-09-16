"""
"""


#[
from __future__ import annotations

from ..conveniences import views as _views
from .. import dates as _dates
#]


class Inlay(_views.Mixin, ):
    """
    """
    #[

    def _get_first_line_view(self, /, ):
        """
        """
        shape = self.data.shape
        span_str = _dates.get_printable_span(self.start, self.end, )
        missing_str = "*" if self.has_missing else " "
        return f"Series {self.frequency.letter} {span_str}{missing_str}{shape[0]}×{shape[1]}"

    def _get_header_separator(self, /, ):
        """"
        """
        return ""

    def _get_content_view(self, /, ):
        """
        """
        return tuple(
            _get_series_row_str_(date, data_row, self._date_str_format, self._numeric_format, self._missing_str) 
            for row, (date, data_row) in enumerate(zip(self.span, self.data))
        )

    def _get_short_row_(self):
        """
        Create string representing rows skipped in the short view of times series
        """
        shape = self.data.shape
        date_str = ("{:"+self._date_str_format+"}").format(_views._VERTICAL_ELLIPSIS)
        value_format = "{:"+self._short_str_format+"}"
        data_str = "".join([value_format.format(_views._VERTICAL_ELLIPSIS)] * shape[1])
        return date_str + data_str

    #]


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

