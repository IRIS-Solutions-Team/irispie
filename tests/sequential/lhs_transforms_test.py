
import pytest
import irispie as ir
import random as rn


rn.seed(0)
_ROUND_TO = 5


@pytest.mark.parametrize(
    ["func"], [ ("pct", ), ("log", ), ("diff", ), ("roc", ), ("diff_log", ) ],
)
def test_simulate(func: str ):

    source = r"""
    !equations
        {{func}}(x) = {{func}}(y);
    """

    m = ir.Sequential.from_string(
        source,
        context={"func": func},
    )

    start_sim = ir.qq(2020,1)
    end_sim = ir.qq(2025,4)

    db = ir.Databox()
    db["y"] = ir.Series(
        periods=start_sim-1>>end_sim,
        func=lambda: 2 + rn.uniform(0, 1),
    )
    db["x"] = db["y"](start_sim-1)

    s = m.simulate(db, start_sim>>end_sim, )

    assert ir.round(s["x"], _ROUND_TO) == ir.round(s["y"], _ROUND_TO)

