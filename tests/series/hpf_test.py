
import sys
sys.path.append("../../../src")

import random as _ra

from irispie import *

#x = Series.from_func(qq(2020,1)>>qq(2025,4), _ra.gauss, num_columns=2)
x = 10+Series.from_func(qq(2020,1)>>qq(2025,4), _ra.gauss, )
#x = exp(x)

start_date = qq(2020,1)
end_date = qq(2025,4) + 8
change = Series.from_start_date_and_values(end_date, 0)
xt, xg = x.hpf(log=False, range=start_date>>end_date, )#change=change, )

