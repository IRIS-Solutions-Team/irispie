
import irispie as ir

model = """
!variables
    x, y, z

!shocks
    shk_x, shk_y, shk_z

!equations
    x = 0.8*x[-1] + (1-0.8)*1.5 + shk_x;
    y = x + shk_y;
    log(z) = 0.8*log(z[-1]) + (1-0.8)*log(3) + shk_z;

"""

m = ir.Model.from_string(model, flat=True, )


m.steady(optim_settings={"factor": 0.1}, )
m.check_steady()

print(m.get_steady(round=4, ), )

a = m.create_steady_array(num_columns=5, )

print(a)

