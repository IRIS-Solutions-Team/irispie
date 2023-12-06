"""
"""


#[
from __future__ import annotations

from typing import (Protocol, Callable, )
from numbers import (Real, )
import numpy as _np

from ..incidences import main as _incidence
#]


_WHEN_DATA_SYMBOL = {
    True: "?",
    False: "!",
    None: "",
}


class PlanTransformProtocol(Protocol, ):
    """
    """
    #[

    _DEFAULT_NAME_FORMAT: str
    _SYMBOL: str

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray | None,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        ...

    def __str__(self, ) -> str: ...

    #]


class PlanTransform:
    """
    """
    #[

    def __init__(
        self,
        /,
        when_data: bool | None = False,
        name_format: str | None = None,
        shift: int = -1,
    ) -> None:
        self.when_data = when_data or False
        self._name_format = name_format or self._DEFAULT_NAME_FORMAT
        self._shift = shift

    @property
    def symbol(self, /, ) -> str:
        return _WHEN_DATA_SYMBOL[self.when_data] + self._SYMBOL

    def __str__(self, /, ) -> str:
        return self.symbol

    def __repr__(self, /, ) -> self:
        return self.__str__()

    def resolve_databox_name(
        self,
        base_name: str, 
        /,
    ) -> str | None:
        """
        """
        return (
            self._name_format.format(base_name, )
            if self._name_format is not None else None
        )

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray | None,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        ...

    #]


class PlanTransformNone(PlanTransform, ):
    """
    """
    #[
    _DEFAULT_NAME_FORMAT = "{}"
    _SYMBOL = "[=]"

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        return exogenized_values_after[0]
    #]


class PlanTransformLog(PlanTransform, ):
    """
    """
    #[
    _DEFAULT_NAME_FORMAT = "log_{}"
    _SYMBOL = "[log]"

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        return _np.exp(exogenized_values_after[0])
    #]


class PlanTransformDiff(PlanTransform, ):
    """
    """
    #[
    _DEFAULT_NAME_FORMAT = "diff_{}"
    _SYMBOL = "[diff]"

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        return values_before[self._shift] + exogenized_values_after[0]
    #]


class PlanTransformDiffLog(PlanTransform, ):
    """
    """
    #[
    _DEFAULT_NAME_FORMAT = "diff_log_{}"
    _SYMBOL = "[diff_log]"

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        return values_before[self._shift] * _np.exp(exogenized_values_after[0])
    #]


class PlanTransformRoc(PlanTransform, ):
    """
    """
    #[
    _DEFAULT_NAME_FORMAT = "roc_{}"
    _SYMBOL = "[roc]"

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray | None,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        return values_before[self._shift] * exogenized_values_after[0]
    #]


class PlanTransformPct(PlanTransform, ):
    """
    """
    #[

    _DEFAULT_NAME_FORMAT = "pct_{}"
    _SYMBOL = "[pct]"

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        return values_before[self._shift] * (1 + exogenized_values_after[0]/100)

    #]


class PlanTransformFlat(PlanTransform, ):
    """
    """
    #[

    _DEFAULT_NAME_FORMAT = None
    _SYMBOL = "[flat]"

    def eval_exogenized(
        self,
        exogenized_values_after: _np.ndarray,
        values_before: _np.ndarray,
        /,
    ) -> Real:
        return values_before[self._shift]

    #]


CHOOSE_TRANSFORM_CLASS = {
    None: PlanTransformNone,
    "level": PlanTransformNone,
    "none": PlanTransformNone,
    "log": PlanTransformLog,
    "diff": PlanTransformDiff,
    "diff_log": PlanTransformDiffLog,
    "difflog": PlanTransformDiffLog,
    "roc": PlanTransformRoc,
    "pct": PlanTransformPct,
    "flat": PlanTransformFlat,
}


def resolve_transform(transform, **kwargs, ) -> PlanTransformProtocol:
    """
    """
    #[
    if transform is None or isinstance(transform, str):
        return CHOOSE_TRANSFORM_CLASS[transform](**kwargs, )
    else:
        return transform
    #]

