

#[
from __future__ import annotations

from . import solutions as _solutions
#]


def run_forward(
    solution: _solutions.Solution,
    initials: tuple[_np.ndarray, _np.ndarray, ],
    /,
) -> None:
    """
    """
    a0, P0 = initials
    Ta = solution.Ta
    Ka = solution.Ka
    Pa = solution.Pa

