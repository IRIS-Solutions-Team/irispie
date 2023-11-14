"""
"""


#[
from __future__ import annotations

from typing import (Self, )
from numbers import (Number, )
import numpy as _np

from ..incidences import main as _incidence
from . import differentiators as ad_
from . import _rules as _rules

#]


class Atom(ad_.LoglyMixin):
    """
    Atomic value for invariance testing
    """
    _data_context: _np.ndarray | None = None
    _logly_context: dict[int, bool] | None = None
    _is_atom: bool = True
    #[
    def __init__(self) -> None:
        """
        """
        self._diff = None
        self._invariant = None
        self._logly = None
        self._logly_index = None

    @classmethod
    def no_context(
        cls: type,
        diff: _np.ndarray,
        invariant: _np.ndarray,
        logly: bool,
    ) -> Self:
        self = cls()
        self._diff = diff
        self._invariant = invariant
        return self

    @classmethod
    def in_context(
        cls: type,
        diff: _np.ndarray,
        token: _incidence.Token,
        *args,
    ) -> Self:
        """
        """
        self = cls()
        self._diff = diff==1
        self._invariant = _np.full((diff.shape[0], 1), True)
        self._logly_index = token.qid
        return self

    __add__ = _rules.add_invar

    __sub__ = _rules.sub_invar

    def __mul__(self, other: Self | Number):
        """
        Invariance of self(x)*other(x) or self(x)*other
        """
        if hasattr(other, "_is_atom"):
            new_diff = self._diff | other._diff
            if _np.all(self._diff==False):
                new_invariant = other._invariant
                new_logly = other.logly
            elif _np.all(other._diff==False):
                new_invariant = self._invariant
                new_logly = self.logly
            else:
                new_invariant = (
                    _np.logical_not(self._diff) & other._invariant
                    & _np.logical_not(other._diff) & self._invariant
                )
                new_logly = False
        else:
            new_diff = self._diff
            new_invariant = self._invariant
            new_logly = self.logly
        return Atom.no_context(new_diff, new_invariant, new_logly)

    def __rmul__(self, other):
        """
        Invariance of other*self(x)
        """
        return Atom.no_context(self._diff, self._invariant, self._logly)

    def __truediv__(self, other):
        """
        Invariance of self(x)/other(x) or self(x)/other
        """
        if hasattr(other, "_is_atom"):
            new_diff = self._diff | other._diff
            new_invariant = _np.all(other._diff, axis=0) & self._invariant
            new_logly = False
        else:
            new_diff = self._diff
            new_invariant = self._invariant
            new_logly = self._logly
        return Atom.no_context(new_diff, new_invariant, new_logly)

    def __rtruediv__(self, other: Number) -> Self:
        """
        Invariance of self(x) / other
        """
        new_diff = self._diff
        new_invariant = False & self._invariant
        return Atom.no_context(new_diff, new_invariant, False)

    __radd__ = __add__

    __rsub__ = __add__

    def _log_(self) -> Self:
        """
        Invariance of log(self(x))
        """
        new_diff = self._diff
        new_invariant = new_invariant if self._logly else (False & self._invariant)
        return Atom.no_context(new_diff, new_invariant, False)

    def _unary(self) -> Self:
        """
        Invariance of f(self(x))
        """
        new_diff = self._diff
        new_invariant = False & self._invariant
        return Atom.no_context(new_diff, new_invariant, False)

    def _binary(self, other: Self | Number) -> Self:
        """
        Invariance of f(self(x), other(x)) or f(self(x), other)
        """
        if hasattr(other, "_is_atom"):
            new_diff = self._diff | other._diff
        else:
            new_diff = self._diff
        new_invariant = False & self._invariant
        return Atom.no_context(new_diff, new_invariant, False)

    exp = _unary
    __pow__ = _binary
    #]


