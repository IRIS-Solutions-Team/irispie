"""
Resolve pseudofunctions in model source string
"""


#[
from __future__ import annotations

import re as re_

from ..parsers import (shifts as sh_, )
#]


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Exposure
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def resolve_pseudofunctions(source: str, /, ) -> str:
    """
    Substitute expanded pseudofunctions in source string
    """
    return re_.sub(_PSEUDOFUNC_PATTERN, _expand_pseudofunction, source, )


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Implementation
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def _expand_pseudofunction(match, /, ) -> str:
    func_name, expression, shift = match.group(1), match.group(2), match.group(3)
    func, default_shift = _PSEUDOFUNC_RESOLUTION[func_name]
    shift = _resolve_shift(shift, default_shift)
    return func(expression, shift)


def _resolve_shift(shift: str, default_shift: int, /, ) -> int:
    shift = shift.strip() if shift else ""
    return int(shift) if shift else default_shift


_NAME_MAYBE_WITH_SHIFT = re_.compile(
    # Name starting with a letter
    r"(\b[a-zA-Z]\w*\b)" 
    # Optional square bracketed time shift, capture only the shift
    # (part of sh.TIME_SHIFT_INSIDE)
    r"(?:\[(" + sh_.TIME_SHIFT_INSIDE + r")\])?" 
    # Not followed by ( or [
    r"(?![\(\[])" 
)


def _shift_all_names(
    source: str,
    by: int,
    /,
) -> str:
    """
    Shift all names in a source string by a given number of time steps
    """
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
    sequence, total = _pseudo_mov(code, shift, )
    return "(" + "+".join(sequence) + ")"


def _pseudo_movavg(code, shift, /, ) -> str:
    sequence, total = _pseudo_mov(code, shift, )
    return "(" + "(" + "+".join(sequence) + ")/" + str(total) + ")"


def _pseudo_movprod(code, shift, /, ) -> str:
    sequence, *_ = _pseudo_mov(code, shift)
    return "(" + "*".join(sequence) + ")"


_PSEUDOFUNC_RESOLUTION = {
    "shift":
        (_pseudo_shift, -1),
    "diff":
        (_pseudo_diff, -1),
    "difflog":
        (_pseudo_difflog, -1),
    "pct": 
        (_pseudo_pct, -1),
    "roc": 
        (_pseudo_roc, -1),
    "movsum":
        (_pseudo_movsum, -4),
    "movavg":
        (_pseudo_movavg, -4),
    "movprod":
        (_pseudo_movprod, -4),
}


# One of the pseudofunction names as a whole word
_PSEUDOFUNC_NAME_PATTERN = r"\b(" + "|".join(_PSEUDOFUNC_RESOLUTION.keys()) + r")\b"

# Groupped text with no comma and possibly ONE LEVEL of parentheses
_PSEUDOFUNC_EXPRESSION_PATTERN = r"((?:[^\(\),]+|\([^\(\),]+\))+)"

# Groupped text with no comma and no parentheses, optional
_PSEUDOFUNC_TIME_SHIFT_PATTERN = r"(?:,([^\(\),]+))?"

# Parenthesized arguments into pseudofunction
_PSEUDOFUNC_BODY_PATTERN = r"\(" + _PSEUDOFUNC_EXPRESSION_PATTERN + _PSEUDOFUNC_TIME_SHIFT_PATTERN+ r"\)"

# Full pseudofunction pattern
_PSEUDOFUNC_PATTERN = re_.compile(_PSEUDOFUNC_NAME_PATTERN + _PSEUDOFUNC_BODY_PATTERN)

