
import irispie as ir

class X:
    pass

x = X()
x.simulate_can_be_exogenized = ["a", "b", "c"]
x.simulate_can_be_endogenized = ["res_a", "res_b", "res_c"]

rng = ir.qq(2020,1) >> ir.qq(2021,4)
p = ir.PlanSimulate(x, rng, )
p.exogenize(ir.qq(2020,2), "a", when_data=True, transform="log", )
p.endogenize(ir.qq(2020,2), "res_a", )
print(p.pretty)

