"""
Time series chartpacks
"""


#[
from __future__ import annotations

from typing import (Self, Callable, )
import re as _re
#]


class Chartpack:
    """
    """
    #[

    __slots__ = (
        "transforms",
        "charts",
    )

    #]


class Chart:
    """
    """
    #[

    _INPUT_STRING_PATTERN = _re.compile(r"^(?P<title>.*?:)(?P<expression>.*?)(?:\[(?P<transform>.*?)\])?$")

    __slots__ = (
        "title",
        "expression",
        "transform"
    )

    def __init__(
        self,
        expression: str,
        title: str | None = None,
        transform: str | Callable | None = None,
    ) -> None:
        """
        """
        self.expression = expression
        self.title = title or None
        self.transform = transform or None

    @classmethod
    def from_string(
        klass: type(Self),
        input_string: str,
    ) -> Self:
    pass

    #]

    + ["x", "inflation: pct_cpi [dev]"
