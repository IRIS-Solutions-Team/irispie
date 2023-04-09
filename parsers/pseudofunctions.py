"""
Resolve pseudofunctions in model source string
"""


#[
from __future__ import annotations

import re as re_

from ..parsers import (shifts as sh_, )
#]


def resolve_pseudofunctions(source: str, /, ) -> str:
    return re_.sub(_PSEUDOFUNCTION_PATTERN, _replace, source, )


def _replace(match, /, ) -> str:
    func_name = match.group(1)
    expression = match.group(2)
    func, default_shift = _PSEUDOFUNC_RESOLUTION[func_name]
    shift = _resolve_shift(match.group(3), default_shift)
    return func(expression, shift)


_NAME_MAYBE_WITH_SHIFT = re_.compile(
    r"(\b[a-zA-Z]\w*\b)" # Name starting with a letter
    + r"(?:" # Non-capturing...
    + r"\[(" + sh_.TIME_SHIFT_INSIDE + r")\]" # Square bracketed time shift
    + r")?" # ...optional
    + r"(?![\(\[])" # Not followed by ( or [
)


def _shift_all_names(source, by, /, ) -> str:
    def _replace(match, /, ) -> str:
        name, shift = match.group(1), match.group(2)
        shift = eval(shift) if shift else 0
        shift += int(by) if by else 0
        return (name + "[" + str(shift) + "]") if shift else name
    return re_.sub(_NAME_MAYBE_WITH_SHIFT, _replace, source, )


def _pseudo_shift(code, shift, /, ) -> str:
    return _shift_all_names(code, shift)


def _pseudo_diff(code, shift, /, ) -> str:
    return "(" + "(" + code + ")-(" + _shift_all_names(code, shift) + ")" + ")"


def _pseudo_difflog(code, shift, /, ) -> str:
    return "(" + "log(" + code + ")-log(" + _shift_all_names(code, shift) + ")" + ")"


def _pseudo_pct(code, shift, /, ) -> str:
    return "(" + "100*(" + code + ")/(" + _shift_all_names(code, shift) + ")-100" + ")"


def _pseudo_roc(code, shift, /, ) -> str:
    return "(" + "(" + code + ")/(" + _shift_all_names(code, shift) + ")" + ")"


def _pseudo_mov(code, shift, /, ) -> list[str]:
    if shift == 0:
        return ["0"], 0
    elif shift == 1 or shift == -1:
        return [code], 1
    else:
        step, total = (1, shift) if shift > 0 else (-1, -shift)
        return [
            "(" + _shift_all_names(code, sh) + ")" 
            for sh in range(0, shift, step)
        ], total


def _pseudo_movsum(code, shift, /, ) -> str:
    sequence, total = _pseudo_mov(code, shift)
    return "(" + "+".join(sequence) + ")"


def _pseudo_movavg(code, shift, /, ) -> str:
    sequence, total = _pseudo_mov(code, shift)
    return "(" + "(" + "+".join(sequence) + ")/" + str(total) + ")"


def _pseudo_movprod(code, shift, /, ) -> str:
    sequence, *_ = _pseudo_mov(code, shift)
    return "(" + "*".join(sequence) + ")"


_PSEUDOFUNCTION_NAME_PATTERN = r"\b(shift|diff|difflog|pct|roc|movavg|movsum|movprod)\b"
_PSEUDOFUNCTION_BODY_PATTERN = r"\(([^,\)]+)(?:,([^\)]+))?\)"
_PSEUDOFUNCTION_PATTERN = re_.compile(_PSEUDOFUNCTION_NAME_PATTERN + _PSEUDOFUNCTION_BODY_PATTERN)
_PSEUDOFUNC_RESOLUTION = {
    "shift": (_pseudo_shift, -1),
    "diff": (_pseudo_diff, -1),
    "difflog": (_pseudo_difflog, -1),
    "pct": (_pseudo_pct, -1),
    "roc": (_pseudo_roc, -1),
    "movsum": (_pseudo_movsum, -4),
    "movprod": (_pseudo_movprod, -4),
    "movavg": (_pseudo_movavg, -4),
}


def _resolve_shift(shift: str, default_shift: int, /, ) -> int:
    shift = shift.strip() if shift else ""
    return int(shift) if shift else default_shift


