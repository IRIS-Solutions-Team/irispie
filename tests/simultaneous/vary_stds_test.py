
import irispie as ir


source = r"""
!transition_variables
    a, b, c
!unanticipated_shocks
    shk_a, shk_b, shk_c
!transition_equations
    a = shk_a;
    b = shk_b;
    c = shk_c;
"""

m = ir.Simultaneous.from_string(source, linear=True, flatten=True, )

m.assign(
    std_shk_a=0.1,
    std_shk_b=0.2,
    std_shk_c=0.3,
)


simulation_start = ir.qq(2020,1)
simulation_end = ir.qq(2021,4)
simulation_span = simulation_start >> simulation_end


def test_scalar_overwrite():
    db0 = ir.Databox()
    db0["std_shk_a"] = 10
    db0["std_shk_b"] = 20
    db1 = m.vary_stds(None, db0, simulation_span, )
    assert all( i == 10 for i in db1["std_shk_a"].get_values() )
    assert all( i == 20 for i in db1["std_shk_b"].get_values() )
    assert all( i == 0.3 for i in db1["std_shk_c"].get_values() )


def test_scalar_multiplier():
    db0 = ir.Databox()
    db0["std_shk_a"] = 10
    db0["std_shk_b"] = 20
    db1 = m.vary_stds(db0, None, simulation_span, )
    assert all( i == 10*0.1 for i in db1["std_shk_a"].get_values() )
    assert all( i == 20*0.2 for i in db1["std_shk_b"].get_values() )
    assert all( i == 0.3 for i in db1["std_shk_c"].get_values() )


def test_series_overwrite():
    periods = (simulation_start+1, simulation_end-1, )
    db0 = ir.Databox()
    db0["std_shk_a"] = ir.Series(periods=periods, values=[10, 10], )
    db0["std_shk_b"] = ir.Series(periods=periods, values=[20, 20], )
    db1 = m.vary_stds(None, db0, simulation_span, )
    assert db1["std_shk_a"].get_values() ==  (0.1, 10, 0.1, 0.1, 0.1, 0.1, 10, 0.1)
    assert db1["std_shk_b"].get_values() ==  (0.2, 20, 0.2, 0.2, 0.2, 0.2, 20, 0.2)
    assert db1["std_shk_c"].get_values() ==  tuple(0.3 for _ in range(8))


def test_series_multipier():
    periods = (simulation_start+1, simulation_end-1, )
    db0 = ir.Databox()
    db0["std_shk_a"] = ir.Series(periods=periods, values=[10, 10], )
    db0["std_shk_b"] = ir.Series(periods=periods, values=[20, 20], )
    db1 = m.vary_stds(db0, None, simulation_span, )
    assert db1["std_shk_a"].get_values() ==  (0.1, 10*0.1, 0.1, 0.1, 0.1, 0.1, 10*0.1, 0.1)
    assert db1["std_shk_b"].get_values() ==  (0.2, 20*0.2, 0.2, 0.2, 0.2, 0.2, 20*0.2, 0.2)
    assert db1["std_shk_c"].get_values() ==  tuple(0.3 for _ in range(8))


if __name__ == "__main__":
    test_scalar_overwrite()
    test_scalar_multiplier()
    test_series_overwrite()
    test_series_multipier()

