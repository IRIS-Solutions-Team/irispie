"""
"""


#[
from __future__ import annotations

import re as _re
#]


TIME_SHIFT_INSIDE = r"[\s\+\-\d]+"
_CURLY_TIME_SHIFT_PATTERN = _re.compile(r"(?<=[\w\?])\{(" + TIME_SHIFT_INSIDE + r")\}")


def standardize_time_shifts(source: str, /, ) -> str:
    return _re.sub(_CURLY_TIME_SHIFT_PATTERN, r"[\g<1>]", source, )

