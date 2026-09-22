r"""
"""

#[

from __future__ import annotations

import numpy as _np

#]


_EPS = _np.finfo(float).eps


DEFAULT_TOLERANCE = {

    # Relative distance from the unit circle within which an eigenvalue is
    # classified as a unit root; the value is the accuracy that can be
    # realistically expected from a QZ decomposition of a macroeconomic model
    "eigenvalue": _EPS**(5/9),

    # Relative loading on the unit-root components above which an element of a
    # solution vector is classified as nonstationary
    "stationarity": _EPS**(5/9),

    # Relative magnitude below which the entries of the solution matrices are
    # clipped to zero
    "clip": _EPS**(2/3),

    # Reciprocal condition number below which a matrix is considered
    # numerically singular; used for the Blanchard-Kahn rank condition, and
    # corresponding to a loss of more than about ten significant digits
    "rank": _EPS**(2/3),

    "equality": 1e-12,
}


class Mixin:
    r"""
    """
    #[

    def reset_tolerance(self, ) -> None:
        r"""
        """
        self._invariant.populate_tolerance()

    def override_tolerance(self, *args, **kwargs, ) -> dict[str, float]:
        r"""
        """
        tolerance = dict(self._invariant.tolerance, )
        update_tolerance = (args[0] if args else {}) | kwargs
        #
        invalid_keys = set(update_tolerance.keys()) - set(tolerance.keys())
        if invalid_keys:
            raise ValueError(f"Invalid tolerance keys: {invalid_keys}.", )
        #
        for key, value in update_tolerance.items():
            if value is not None:
                tolerance[key] = float(value)
        self._invariant.tolerance = tolerance
        return tolerance

    def get_tolerance(
        self,
        key: str | None = None,
    ) -> float | dict[str, float]:
        r"""
        """
        return self._invariant.tolerance[key] if key else self._invariant.tolerance

    #]

