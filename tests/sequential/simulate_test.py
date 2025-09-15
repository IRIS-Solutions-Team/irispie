

import irispie as ir

_SOURCE = r"""
!parameters

    c0_pct_x, ss_pct_x


!equations

    !for ? = <range(N)> !do

        pct(x?) = c0_pct_x * pct(x?[-1]) + (1 - c0_pct_x) * ss_pct_x;

        pct_x? = pct(x?);

        x?_eop = (x? + x?[+1]) / 2;

    !end
"""


N = 2
m, info, = ir.Sequential.from_string(
    _SOURCE,
    context={"N": N, },
    return_info=True,
)

d = ir.Databox()
for i in range(N, ):
    d[f"x{i}"] = ir.Series(start_date=ir.qq(2020,1)-2, values=(1,)*10, )
    d[f"pct_x{i}"] = ir.Series(start_date=ir.qq(2020,1)-2, values=(1,)*10, )

m.assign(c0_pct_x=0.8, ss_pct_x=0.5, )

span = ir.qq(2020,1,...,2025,4)

d["x0"].clip(None, ir.qq(2021,2), )

p = ir.SimulationPlan(m, span, )
p.exogenize(..., "x0", when_data=True, )
p.exogenize(ir.qq(2021,1,...,2021,4), "x1", transform="diff", )

d["diff_x1"] = ir.Series(start_date=ir.qq(2021,1), values=(0.3,)*4, )


def test_simulate_dates_equations():
    s0 = m.simulate(
        d, span,
        plan=p,
        when_nonfinite="silent",
    )
    for i in range(N, ):
        assert s0[f"x{i}_eop"].end == d[f"x{i}"].end - 1 


def test_simulate_equations_dates():
    s1 = m.simulate(
        d, span,
        plan=p,
        when_nonfinite="silent",
        execution_order="equations_dates",
        shocks_from_data=False,
    )
    for i in range(N, ):
        assert s1[f"x{i}_eop"].end == span[-1] - 1 


if __name__ == "__main__":
    test_simulate_dates_equations()
    test_simulate_equations_dates()

