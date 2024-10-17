
import pytest
import irispie as ir
import random as rn


rn.seed(0)

_ROUND_TO = 5

_SOLVER_SETTINGS = {
    "step_tolerance": 100,
}

_START_SIM = ir.qq(2020,1)
_END_SIM =  ir.qq(2020,4)

_INPUT_DB = ir.Databox()
_INPUT_DB["y"] = ir.Series(
    periods=_START_SIM-1>>_END_SIM,
    func=lambda: 100 + rn.uniform(0, 1)/10,
)
_INPUT_DB["x"] = ir.Series(
    periods=_START_SIM-1>>_END_SIM,
    values=100,
)
_INPUT_DB["x"][_START_SIM-1] = _INPUT_DB["x"][_START_SIM-1]



@pytest.mark.parametrize(
    ["func"], [ ("pct", ), ("log", ), ("diff", ), ("roc", ), ("diff_log", ) ],
)
def test_simulate(func: str ):

    source = r"""
    !transition_variables
        x
    !exogenous_variables
        y
    !equations
        {{func}}(x) = {{func}}(y);
    """

    m = ir.Simultaneous.from_string(
        source,
        context={"func": func},
    )

    s = m.simulate(
        _INPUT_DB,
        _START_SIM>>_END_SIM,
        method="period_by_period",
        solver_settings=_SOLVER_SETTINGS,
    )

    assert ir.round(s["x"], _ROUND_TO) == ir.round(s["y"], _ROUND_TO)


