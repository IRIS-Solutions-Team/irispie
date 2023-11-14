"""
"""


#[
from __future__ import annotations
from numbers import (Number, )
import numpy as _np
import re as _re

from ..conveniences import views as _views
#]


_REPR_INDENT = "    "
_REPR_SEPARATOR = ": "
_REPR_MAX_LEN = 70
_REPR_CONT = "..."


class ViewMixin(_views.ViewMixin, ):
    """
    """
    #[
    def _get_first_line_view(self, /, ):
        """
        """
        return f"{self.__class__.__name__} with {self.get_num_items():g} item(s)"

    def _get_content_view_(self, /, ):
        """
        """
        names = self.get_names()
        if names:
            max_len = self._get_max_name_length_()
            content = tuple(_REPR_INDENT + str(k).rjust(max_len) + _REPR_SEPARATOR + _databox_repr(self[k]) for k in names)
        else:
            content = ()
        return content

    def _get_max_name_length_(self, /, ):
        """
        """
        return max(len(str(k)) for k in self.get_names())

    def _get_short_row_(self):
        """
        """
        max_len = self._get_max_name_length_()
        return _REPR_INDENT + _views._VERTICAL_ELLIPSIS.rjust(max_len) + " "*len(_REPR_SEPARATOR) + _VERTICAL_ELLIPSIS
    #]


def _databox_repr(x, /, ) -> str:
    """
    String representing one item in a databox
    """
    #[
    if hasattr(x, "_databox_repr"):
        s = f"<<{x._databox_repr()}>"
    elif x is None:
        s = "None"
    elif x is ...:
        s = "..."
    elif isinstance(x, Number):
        s = str(x)
    elif isinstance(x, str):
        s = f'"{x}"'
    elif isinstance(x, _np.ndarray) or isinstance(x, list) or isinstance(x, tuple):
        s = _re.sub(r"\n+ +", " ", repr(x))
    elif hasattr(x, "_get_first_line_view"):
        s = f"<{x._get_first_line_view()}>"
    else:
        s = repr(type(x))
    return s if len(s)<_REPR_MAX_LEN else s[0:_REPR_MAX_LEN] + _REPR_CONT
    #]

