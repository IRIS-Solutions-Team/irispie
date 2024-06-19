
import irispie as _ir
import numpy as _np
from irispie.zeries import main as _zz
import time

x = _zz.Series(
    start_date=_ir.qq(1),
    values=[(1,None,None,None,10),(2,3),range(9)],
    num_variants=3,
)

y = _zz.Series(
    dates=_ir.qq(0)>>_ir.qq(2,4),
    func=_np.random.standard_normal,
)

#y[_ir.start+2>>_ir.start+3] = None

x0 = x.copy()
x1 = x.copy()

x0.underlay(y, method="by_span", )
x1.underlay(y, method="by_date", )
print(x0)
print(x1)

aa = _np.random.standard_normal((1000,1000))
aaa = _np.random.standard_normal((20,2))

q = _ir.Series(
    start_date=_ir.qq(1),
    values=aaa,
)

q0 = q.copy()
q0[_ir.start+2] = None
t, g = _ir.hpf(q, )

a = _ir.aggregate(q0, _ir.Freq.YEARLY, )

