"""
"""


#[

from __future__ import annotations

from numbers import Number
import numpy as _np
import re as _re

from ..conveniences import views as _views

#]


_REPR_INDENT = "⏐ "
_REPR_SEPARATOR = ": "
_REPR_MAX_LEN = 50
_REPR_CONT = "…"


class Inlay(_views.Mixin, ):
    """
    """
    #[

    def _get_first_line_view(self, /, ):
        """
        """
        return f"{self.__class__.__name__} with {self.num_items:g} item(s)"

    def _get_header_separator(self, /, ):
        """"
        """
        return _REPR_INDENT.rstrip()

    def _get_content_view(self, /, ) -> tuple[str, ...]:
        """
        """
        max_len = self._get_max_key_repr_len_()
        return tuple(
            _REPR_INDENT + _get_key_repr(k).rjust(max_len) + _REPR_SEPARATOR + _databox_repr(self[k])
            for k in self.keys()
        )

    def _get_max_key_repr_len_(self, /, ):
        """
        """
        key_reprs = tuple(_get_key_repr(k) for k in self.keys())
        return max(len(k) for k in key_reprs) if key_reprs else 0

    def _get_short_row_(self):
        """
        """
        max_len = self._get_max_key_repr_len_()
        return _REPR_INDENT + _views._VERTICAL_ELLIPSIS.rjust(max_len) + " "*len(_REPR_SEPARATOR) + _VERTICAL_ELLIPSIS

    def _get_footer_view_(self, /, ):
        return (_REPR_INDENT.rstrip(), )
    #]


def _databox_repr(x, /, ) -> str:
    """
    String representing one item in a databox
    """
    #[
    if hasattr(x, "_databox_repr"):
        s = f"<{x._databox_repr()}>"
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


def _get_key_repr(key: Any, /, ) -> str:
    #[
    if isinstance(key, str):
        return f'"{key}"'
    else:
        return str(key)
    #]

