"""
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Any, Callable, )
import dataclasses as _dc
import copy as _cp
import numpy as _np

from ..incidences import main as _incidence
from .. import equations as _equations
from .. import wrongdoings as _wrongdoings
from .. import makers as _makers

from . import _transforms as _transforms
#]


__all__ = (
    "Explanatory",
)


@_dc.dataclass
class Explanatory:
    """
    """
    #[

    _RESIDUAL_PREFIX = "res_"
    _IDENTITY_EQUAL_SIGN = "==="
    equation: _equations.Equation | None = None
    is_identity: bool | None = None
    _lhs_human: str | None = None
    lhs_name: str | None = None
    res_name: str | None = None
    res_name: str | None = None
    lhs_qid: int | None = None
    _rhs_human: str | None = None
    _lhs_transform: TransformProtocol | None = None
    _eval_level_str: str | None = None
    _eval_res_str: str | None = None
    _context: dict[str, Callable] | None = None
    all_names: tuple(str) | None = None
    eval_level: Callable | None = None
    eval_res: Callable | None = None

    def __init__(
        self,
        equation: _equations.Equation,
        /,
        context: dict[str, Callable] | None = None,
    ) -> None:
        """
        """
        self.equation = equation.copy()
        self._detect_identity()
        self._split_equation()
        self._parse_lhs()
        self._add_residual_to_rhs()
        self._collect_all_names()
        self._context = context

    def finalize(
        self,
        name_to_qid,
        /,
    ) -> None:
        """
        """
        self.equation.finalize(name_to_qid, )
        self.lhs_qid = name_to_qid[self.lhs_name]
        self.res_qid = name_to_qid[self.res_name] if self.res_name is not None else None
        rhs_xtring, *_ = _equations.xtring_from_human(self._rhs_human, name_to_qid, )
        lhs_xtring, *_ = _equations.xtring_from_human(self._lhs_human, name_to_qid, )
        lhs_token = _incidence.Token(self.lhs_qid, 0, )
        self._create_eval_level(lhs_token, rhs_xtring, )
        self._create_eval_res(lhs_xtring, rhs_xtring, )

    @property
    def min_shift(self, /, ) -> int:
        """
        """
        return _incidence.get_min_shift(self.equation.incidence, )

    @property
    def max_shift(self, /, ) -> int:
        """
        """
        return _incidence.get_max_shift(self.equation.incidence, )


    @property
    def description(self, /, ) -> str:
        """
        """
        return (
            str(self.equation.description)
            if self.equation.description is not None else ""
        )

    def print_equation(
        self,
        /,
        indent: int = 0,
        description: bool = True,
    ) -> str:
        """
        """
        indented = " " * indent
        print_description = description and self.equation.description
        return (
            ((f"{indented}\"{self.equation.description}\"\n") if print_description else "")
            +f"{indented}{self.equation.human}"
        )

    def _detect_identity(self, /, ) -> None:
        """
        """
        self.is_identity = self._IDENTITY_EQUAL_SIGN in self.equation.human

    def _split_equation(self, ) -> None:
        """
        """
        human = self.equation.human
        equal_sign = "=" if not self.is_identity else self._IDENTITY_EQUAL_SIGN
        split = human.split(equal_sign, )
        if len(split) != 2:
            raise _wrongdoings.IrisPieError(
                f"This equation is not in a LHS=RHS or LHS{self._IDENTITY_EQUAL_SIGN}RHS form: {self.equation.human}"
            )
        self._lhs_human, self._rhs_human = split

    def _parse_lhs(self, ) -> None:
        lhs_transform, lhs_name = _transforms.recognize_transform_in_equation(self._lhs_human, )
        self._lhs_transform = lhs_transform
        self.lhs_name = lhs_name

    def _add_residual_to_rhs(self, /, ) -> None:
        """
        """
        if self.is_identity:
            self.res_name = None
        else:
            self.res_name = f"{self._RESIDUAL_PREFIX}{self.lhs_name}"
            self._rhs_human += f"+{self.res_name}"
            self.equation.human += f"+{self.res_name}"

    def _collect_all_names(self, /, ) -> None:
        """
        """
        self.all_names = tuple(set(
            _equations.generate_names_from_human(self.equation.human, )
        )) + ((self.res_name, ) if self.res_name is not None else ())

    def _create_eval_level(
        self,
        lhs_token: str,
        rhs_xtring: str,
        /,
    ) -> None:
        args = ("x", "t", )
        body = self._lhs_transform.create_eval_level_str(lhs_token, rhs_xtring, )
        self.eval_level, self._eval_level_str, *_ = \
            _makers.make_lambda(args, body, self._context, )

    def _create_eval_res(
        self,
        lhs_xtring: str,
        rhs_xtring: str,
        /,
    ) -> None:
        if self.is_identity:
            return
        args = ("x", "t", )
        body = f"{lhs_xtring}-({rhs_xtring})"
        self.eval_res, self._eval_res_str, *_ = \
            _makers.make_lambda(args, body, self._context, )

    def __str__(self, /, ) -> str:
        """
        """
        indented = " " * 4
        return "\n".join((
            f"{indented}Explanatory object",
            f"{indented}Description: \"{self.description}\"",
            f"{indented}Equation: {self.equation.human}",
            f"{indented}LHS name: {self.lhs_name}",
            f"{indented}LHS transform: {str(self._lhs_transform)}",
            f"{indented}Residual name: {self.res_name}",
        ))

    def simulate(
        self,
        data: _np.ndarray,
        columns: int | _np.ndarray,
        /,
    ) -> dict[str, Any]:
        """
        """
        row = self.lhs_qid
        data[row, columns] = self.eval_level(data, columns, )
        info = {"is_finite": _np.isfinite(data[row, columns]), }
        return info

    def exogenize(
        self,
        data: _np.ndarray,
        columns: int | _np.ndarray,
        values: int | _np.ndarray,
        /,
    ) -> dict[str, Any]:
        """
        """
        lhs_row = self.lhs_qid
        res_row = self.res_qid
        data[lhs_row, columns] = values
        data[res_row, columns] = self.eval_res(data, columns, )
        is_finite = _np.isfinite(data[res_row, columns])
        info = {"is_finite": is_finite, }
        return info

    #]

