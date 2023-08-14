

import sys
sys.path.append("../../src")
import irispie as _ip
import numpy as _np


x = _ip.Series.from_start_date_and_data(
    _ip.qq(2020, 1),
    range(10),
)

y = _ip.Series.from_start_date_and_data(
    x.start_date-5, _np.random.standard_normal((20,1))
)

fx = _ip.cum_diff(x, initial=y, )
fx2 = _ip.cum_diff(x, -2, initial=y, )

bx2  = _ip.cum_diff(x, initial=y, range=_ip.Ranger(None,None,-1))

w0 = _ip.Series.from_start_date_and_data(
    _ip.qq(2018,1),
    _np.random.standard_normal((36,1)),
)

dw0 =  _ip.diff(w0)

w1 = _ip.cum_diff(dw0, initial=0, range=_ip.Ranger(step=-1,))

dw = _ip.Series.from_start_date_and_data(
    _ip.qq(2020,1),
    _np.random.standard_normal((12,1)),
)

w2 = _ip.cum_diff(dw, "soy", initial=0, range=_ip.qq(2021,1)>>_ip.end)
w1 = _ip.cum_diff(dw, "eopy", initial=dw, range=_ip.qq(2021,1)>>_ip.end)

