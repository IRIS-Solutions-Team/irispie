r"""
"""

#[

from __future__ import annotations

#]


_DEFAULT_TOLERANCE = {
    "eigenvalue": 1e-12,
    "equality": 1e-12,
}


class InlayForInvariant:
    r"""
    """
    #[

    def reset_tolerance(self, ) -> None:
        r"""
        """
        self.tolerance = dict(_DEFAULT_TOLERANCE, )


class Inlay:
    r"""
    """
    #[

    def reset_tolerance(self, ) -> None:
        r"""
        """
        self._invariant.reset_tolerance()

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

