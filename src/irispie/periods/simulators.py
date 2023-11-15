"""
Simulators for dynamic period-by-period systems
"""


#[
from __future__ import annotations

from ..simultaneous import main as _simultaneous
from .. import dataslates as _dataslates
#]


def simulate(
    model: _simultaneous.Simultaneous,
    dataslate: _dataslate.Dataslate,
    /,
    *,
    plan: _plans.PlanSimulate | None,
    anticipate: bool,
    deviation: bool,
) -> None:
    """
    """
    raise NotImplementedError

