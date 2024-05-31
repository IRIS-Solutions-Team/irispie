"""
Mixin classes for displaying data management objects
"""


#[
from __future__ import annotations
#]


_REPEAT_SHORT_ROW = 2
_VERTICAL_ELLIPSIS = "⋮"


class Mixin:
    """
    """
    #[

    _short_rows_: int = 5

    def _get_header_view_(self, /, ):
        """
        """
        return (
            "", 
            self._get_first_line_view(),
            f"Description: \"{self.get_description()}\"",
            self._get_header_separator(),
        )

    def _get_footer_view_(self, /, ):
        return ("", )

    def _get_view(self, /, ):
        """
        """
        header_view = self._get_header_view_()
        content_view = self._get_content_view()
        footer_view = self._get_footer_view_()
        return header_view + content_view + footer_view

    def __invert__(self):
        """
        ~self for short view
        """
        header_view = self._get_header_view_()
        content_view = self._get_content_view()
        if len(content_view) > 2*self._short_rows_:
            content_view = (
                content_view[:self._short_rows_]
                + (self._get_short_row_(), )*_REPEAT_SHORT_ROW
                + content_view[-self._short_rows_:]
            )
        print("\n".join(header_view + content_view))

    def __repr__(self, /, ):
        """
        """
        return "\n".join(self._get_view())

    def __str__(self, /, ):
        """
        """
        return repr(self)

    #]

