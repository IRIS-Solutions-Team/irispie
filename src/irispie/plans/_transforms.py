"""
"""


#[
from __future__ import annotations

from typing import (Protocol, Callable, )
from numbers import (Number, )
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

    _DATABOX_NAME: str
    _SYMBOL: str

    def eval_exogenized(
        self,
        exogenized_value: Number,
        lagged: Number,
    ) -> Number:
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
    ) -> None:
        self.when_data = when_data

    def __str__(self, /, ) -> str:
        return _WHEN_DATA_SYMBOL[self.when_data] + self._SYMBOL

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
            self._DATABOX_NAME.format(base_name, )
            if self._DATABOX_NAME is not None else None
        )

    #]


class PlanTransformNone(PlanTransform, ):
    """
    """
    #[
    _DATABOX_NAME = "{}"
    _SYMBOL = "[•]"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        values_before: _np.ndarray,
        /,
    ) -> Number:
        return exogenized_value
    #]


class PlanTransformLog(PlanTransform, ):
    """
    """
    #[
    _DATABOX_NAME = "log_{}"
    _SYMBOL = "[log]"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        values_before: _np.ndarray,
        /,
    ) -> Number:
        return _np.exp(exogenized_value)
    #]


class PlanTransformDiff(PlanTransform, ):
    """
    """
    #[
    _DATABOX_NAME = "diff_{}"
    _SYMBOL = "[diff]"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        values_before: _np.ndarray,
        /,
    ) -> Number:
        return values_before[-1] + exogenized_value
    #]


class PlanTransformDiffLog(PlanTransform, ):
    """
    """
    #[
    _DATABOX_NAME = "diff_log_{}"
    _SYMBOL = "[diff_log]"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        values_before: _np.ndarray,
        /,
    ) -> Number:
        return values_before[-1] * _np.exp(exogenized_value)
    #]


class PlanTransformRoc(PlanTransform, ):
    """
    """
    #[
    _DATABOX_NAME = "roc_{}"
    _SYMBOL = "[roc]"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        values_before: _np.ndarray,
        /,
    ) -> Number:
        return values_before[-1] * exogenized_value
    #]


class PlanTransformPct(PlanTransform, ):
    """
    """
    #[

    _DATABOX_NAME = "pct_{}"
    _SYMBOL = "[pct]"

    def eval_exogenized(
        self,
        exogenized_value: Number,
        values_before: _np.ndarray,
        /,
    ) -> Number:
        return values_before[-1] * (1 + exogenized_value/100)

    #]


class PlanTransformFlat(PlanTransform, ):
    """
    """
    #[

    _DATABOX_NAME = None
    _SYMBOL = "[flat]"

    def eval_exogenized(
        self,
        exogenized_value: None,
        values_before: _np.ndarray,
        /,
    ) -> Number:
        return values_before[-1]

    #]


class PlanTransformFactory(PlanTransformNone, ):
    """
    """
    #[

    def __init__(
        self,
        eval_exogenized: Callable,
        databox_name: str,
        /,
        symbol: str = "[custom]",
        when_data: bool | None = False,
    ) -> None:
        """
        """
        self.eval_exogenized = eval_exogenized
        self._DATABOX_NAME = databox_name
        self._SYMBOL = symbol
        self.when_data = when_data
    #]


_TRANFORM_CLASS_FACTORY = {
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


def resolve_transform(transform, ) -> PlanTransformProtocol:
    """
    """
    #[
    if transform is None or isinstance(transform, str):
        return _TRANFORM_CLASS_FACTORY[transform]()
    else:
        return transform
    #]

