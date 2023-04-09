"""
"""


#[
from __future__ import annotations

import re as re_
#]


TIME_SHIFT_INSIDE = r"[\s\+\-\d]+"
_CURLY_TIME_SHIFT_PATTERN = re_.compile(r"(?<=[\w\?])\{(" + TIME_SHIFT_INSIDE + ")\}")


def standardize_time_shifts(source: str, /, ) -> str:
    return re_.sub(_CURLY_TIME_SHIFT_PATTERN, r"[\g<1>]", source, )


