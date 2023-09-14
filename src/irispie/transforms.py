"""
"""


#[
from __future__ import annotations

from typing import (Protocol, )
from numbers import (Number, )
import re as _re
import numpy as _np

from .incidences import main as _incidence
#]


class TransformProtocol(Protocol, ):
    _LHS_PATTERN: _re.Pattern
    def __str__(self) -> str: ...


class Transform:
    """
    """
    #[
    def __init__(
        self,
        /,
        when_data: bool = False,
    ) -> None:
        self.when_data = when_data

    def __str__(self, /, ) -> str:
        when_data_symbol = "!" if not self.when_data else "?"
        return when_data_symbol + self._SYMBOL

    def __repr__(self, /, ) -> self:
        return self.__str__()

    def resolve_databank_name(
        self,
        base_name: str, 
        /,
    ) -> str:
        return self._DATABANK_NAME.replace("$", base_name, )
    #]


class TransformNone(Transform, ):
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"(\w+)")
    _DATABANK_NAME = "$"
    _SYMBOL = "••"

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        return rhs_xtring

    def eval_exogenized(
        self,
        exogenized_value: Number,
        lagged: Number,
        /,
    ) -> Number:
        return exogenized_value
    #]


class TransformLog(Transform, ):
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"log\((\w+)\)")
    _DATABANK_NAME = "log_$"
    _SYMBOL = "log"

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        return f"exp({rhs_xtring})"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        lagged: Number,
        /,
    ) -> Number:
        return _np.exp(exogenized_value)
    #]


class TransformDiff(Transform, ):
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"\(\((\w+)\)-\(\1\[-1\]\)\)")
    _DATABANK_NAME = "diff_$"
    _SYMBOL = "diff"

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        lhs_token_lag_printed = lhs_token.shifted(-1, ).print_xtring()
        return f"{lhs_token_lag_printed}+({rhs_xtring})"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        lagged: Number,
        /,
    ) -> Number:
        return lagged + exogenized_value
    #]


class TransformDiffLog(Transform, ):
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"\(log\((\w+)\)-log\(\1\[-1\]\)\)")
    _DATABANK_NAME = "difflog_$"
    _SYMBOL = "difflog"

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        lhs_token_lag_printed = lhs_token.shifted(-1, ).print_xtring()
        return f"{lhs_token_lag_printed}*exp({rhs_xtring})"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        lagged: Number,
        /,
    ) -> Number:
        return lagged * _np.exp(exogenized_value)
    #]


class TransformRoc(Transform, ):
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"\(\((\w+)\)/\(\1\[-1\]\)\)")
    _DATABANK_NAME = "roc_$"
    _SYMBOL = "roc"

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        lhs_token_lag_printed = lhs_token.shifted(-1, ).print_xtring()
        return f"{lhs_token_lag_printed}*({rhs_xtring})"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        lagged: Number,
        /,
    ) -> Number:
        return lagged * exogenized_value
    #]


class TransformPct(Transform, ):
    """
    """
    #[
    _LHS_PATTERN = _re.compile(r"\(100*\((\w+)\)/\(\1\[-1\]\)-100\)")
    _DATABANK_NAME = "pct_$"
    _SYMBOL = "pct"

    def create_eval_level_str(
        self,
        lhs_token: _incidence.Token,
        rhs_xtring: str,
        /,
    ) -> str:
        lhs_token_lag_printed = lhs_token.shifted(-1, ).print_xtring()
        return f"{lhs_token_lag_printed}*(1+({rhs_xtring})/100)"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        lagged: Number,
        /,
    ) -> Number:
        return lagged * (1+exogenized_value/100)
    #]


_ALL_TRANSFORMS = (
    TransformNone,
    TransformLog,
    TransformDiff,
    TransformDiffLog,
    TransformRoc,
    TransformPct,
)


RESOLVE_TRANSFORM = {
    None: TransformNone,
    "level": TransformNone,
    "none": TransformNone,
    "log": TransformLog,
    "diff": TransformDiff,
    "difflog": TransformDiffLog,
    "roc": TransformRoc,
    "pct": TransformPct,
}


def recognize_transform_in_equation(
    lhs: str,
    /,
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

