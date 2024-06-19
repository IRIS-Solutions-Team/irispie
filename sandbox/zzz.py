
import irispie as ir
import numpy as np

num_rhs = 5
num_lhs = 3

n = 20
#lhs = np.random.rand(2, n)
rhs = np.random.rand(num_rhs, n)
res = np.random.randn(num_lhs, n) * 0.05
intc = np.ones((1, n))

# a = np.diag([0.8, 0.8]);
# a[0, 1] = 0.6
# a[1, 0] = -0.6
a = np.random.rand(num_lhs, num_rhs, )
c = np.random.rand(num_lhs, 1)

lhs = a @ rhs + c + res

lhs_s, lhs_ms = ir.standardize(lhs)
rhs_s, rhs_ms = ir.standardize(rhs)

b_s = ir.ordinary_least_squares(lhs_s, rhs_s, )

b1, c1 = ir.destandardize_lstsq(b_s, lhs_ms, rhs_ms)


b2 = ir.ordinary_least_squares(lhs, np.vstack([intc, rhs]))

print(b2[:, 0] - c1)


v = ir.RedVAR(["gdp", "cpi", "stn"], order=2, )

