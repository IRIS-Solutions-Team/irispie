"""
"""


#[
from __future__ import annotations

from typing import (Protocol, )
import re as _re

from ..incidences import main as _incidence
#]


class TransformProtocol(Protocol, ):
    _LHS_PATTERN: _re.Pattern
    def __str__(self) -> str: ...


class TransformNone:
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"(\w+)")

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        return rhs_xtring

    def __str__(self, /, ) -> str:
        return "•"
    #]


class TransformLog:
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"log\((\w+)\)")

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        return f"exp({rhs_xtring})"

    def __str__(self, /, ) -> str:
        return "log"
    #]


class TransformDiff:
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"\(\((\w+)\)-\(\1\[-1\]\)\)")

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        lhs_token_lag_printed = lhs_token.shifted(-1, ).print_xtring()
        return f"{lhs_token_lag_printed}+({rhs_xtring})"

    def __str__(self, /, ) -> str:
        return "∆"
    #]


class TransformDiffLog:
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"\(log\((\w+)\)-log\(\1\[-1\]\)\)")

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        lhs_token_lag_printed = lhs_token.shifted(-1, ).print_xtring()
        return f"{lhs_token_lag_printed}*exp({rhs_xtring})"

    def __str__(self, /, ) -> str:
        return "∆log"
    #]


_ALL_TRANSFORMS = (
    TransformNone,
    TransformLog,
    TransformDiff,
    TransformDiffLog,
)


def recognize_transform(
    lhs: str,
) -> tuple[TransformProtocol | None, str | None]:
    """
    """
    #[
    for t in _ALL_TRANSFORMS:
        m = t._LHS_PATTERN.fullmatch(lhs)
        if m is not None:
            return t(), m.group(1)
    return None, None
    #]

