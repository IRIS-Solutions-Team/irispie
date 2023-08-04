
import sys
sys.path.append("../../../src")

import random as _ra

from irispie import *
from irispie.dataman import filters as _fi

x = Series.from_func(qq(2020,1)>>qq(2025,4), _ra.gauss, num_columns=2)

