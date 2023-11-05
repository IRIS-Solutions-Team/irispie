"""
"""


#[
from __future__ import annotations

from typing import (Protocol, )
import re as _re

from ..incidences import main as _incidence
from .. import wrongdoings as _wrongdoings
#]


class LhsTransformProtocol(Protocol, ):
    _LHS_PATTERN: _re.Pattern

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
    ) -> str:
        ...


class _LhsTransform:
    """
    """
    pass


class LhsTransformNone(_LhsTransform, ):
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
    #]


class LhsTransformLog(_LhsTransform, ):
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

    #]


class LhsTransformDiff(_LhsTransform, ):
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

    #]


class LhsTransformDiffLog(_LhsTransform, ):
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

    #]


class LhsTransformRoc(_LhsTransform, ):
    """
    """
    #[

    _LHS_PATTERN = _re.compile(r"\(\((\w+)\)/\(\1\[-1\]\)\)")

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        lhs_token_lag_printed = lhs_token.shifted(-1, ).print_xtring()
        return f"{lhs_token_lag_printed}*({rhs_xtring})"

    #]


class LhsTransformPct(_LhsTransform, ):
    """
    """
    #[

    _LHS_PATTERN = _re.compile(r"\(100\*\((\w+)\)/\(\1\[-1\]\)-100\)")

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        lhs_token_lag_printed = lhs_token.shifted(-1, ).print_xtring()
        return f"{lhs_token_lag_printed}*(1+({rhs_xtring})/100)"

    #]


_ALL_LHS_TRANSFORMS = (
    LhsTransformNone,
    LhsTransformLog,
    LhsTransformDiff,
    LhsTransformDiffLog,
    LhsTransformRoc,
    LhsTransformPct,
)


def recognize_transform_in_equation(
    lhs: str,
    /,
) -> tuple[LhsTransformProtocol | None, str | None]:
    """
    """
    #[
    for t in _ALL_LHS_TRANSFORMS:
        m = t._LHS_PATTERN.fullmatch(lhs)
        if m is not None:
            return t(), m.group(1)
    raise _wrongdoings.IrisPieError(f"Could not parse this LHS expression: {lhs}")
    #]

