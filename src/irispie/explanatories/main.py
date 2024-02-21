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


_RESIDUAL_NAME_FORMAT = "res_{lhs_name}"


@_dc.dataclass
class Explanatory:
    """
    """
    #[

    _IDENTITY_EQUAL_SIGN = "==="
    equation: _equations.Equation | None = None
    is_identity: bool | None = None
    _lhs_human: str | None = None
    lhs_name: str | None = None
    residual_name: str | None = None
    lhs_qid: int | None = None
    _rhs_human: str | None = None
    _lhs_transform: TransformProtocol | None = None
    _eval_level_str: str | None = None
    _eval_residual_str: str | None = None
    _context: dict[str, Callable] | None = None
    all_names: tuple(str) | None = None
    eval_level: Callable | None = None
    eval_residual: Callable | None = None

    def __init__(
        self,
        equation: _equations.Equation,
        /,
        context: dict[str, Callable] | None = None,
        residual_name_format: str | None = None,
        **kwargs,
    ) -> None:
        """
        """
        self._residual_name_format = (
            residual_name_format
            if residual_name_format
            else _RESIDUAL_NAME_FORMAT
        )
        self.equation = equation.copy()
        self._detect_identity()
        self._split_equation()
        self._parse_lhs()
        self._add_residual_to_rhs()
        self._collect_all_names()

    def finalize(
        self,
        name_to_qid,
        /,
    ) -> None:
        """
        """
        self.equation.finalize(name_to_qid, )
        self.lhs_qid = name_to_qid[self.lhs_name]
        self.residual_qid = name_to_qid[self.residual_name] if self.residual_name is not None else None
        rhs_xtring, *_ = _equations.xtring_from_human(self._rhs_human, name_to_qid, )
        lhs_xtring, *_ = _equations.xtring_from_human(self._lhs_human, name_to_qid, )
        lhs_token = _incidence.Token(self.lhs_qid, 0, )
        self._create_eval_level(lhs_token, rhs_xtring, )
        self._create_eval_residual(lhs_xtring, rhs_xtring, )

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
            self.residual_name = None
        else:
            self.residual_name = self._residual_name_format.format(lhs_name=self.lhs_name, )
            self._rhs_human += f"+{self.residual_name}"
            self.equation.human += f"+{self.residual_name}"

    def _collect_all_names(self, /, ) -> None:
        """
        """
        self.all_names = tuple(set(
            _equations.generate_names_from_human(self.equation.human, )
        )) + ((self.residual_name, ) if self.residual_name is not None else ())

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

    def _create_eval_residual(
        self,
        lhs_xtring: str,
        rhs_xtring: str,
        /,
    ) -> None:
        if self.is_identity:
            return
        args = ("x", "t", )
        body = f"{lhs_xtring}-({rhs_xtring})"
        self.eval_residual, self._eval_residual_str, *_ = \
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
            f"{indented}Residual name: {self.residual_name}",
        ))

    def simulate(
        self,
        data: _np.ndarray,
        columns: int | _np.ndarray,
        values: int | _np.ndarray,
        /,
    ) -> dict[str, Any]:
        """
        """
        lhs_row = self.lhs_qid
        residual_row = self.residual_qid
        data[lhs_row, columns] = self.eval_level(data, columns, )
        is_finite = _np.isfinite(data[lhs_row, columns])
        return {
            "simulated_name": self.lhs_name,
            "is_finite": is_finite,
            "lhs_value": data[lhs_row, columns],
            "residual_value": data[residual_row, columns] if residual_row is not None else None,
        }

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
        residual_row = self.residual_qid
        data[lhs_row, columns] = values
        data[residual_row, columns] = self.eval_residual(data, columns, )
        is_finite = _np.isfinite(data[residual_row, columns])
        return {
            "simulated_name": self.residual_name,
            "is_finite": is_finite,
            "lhs_value": data[lhs_row, columns],
            "residual_value": data[residual_row, columns],
        }

    #]

