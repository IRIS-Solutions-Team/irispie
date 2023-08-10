
import sys
sys.path.append("../../../src")

import irispie as _ip

y = _ip.Frequency.from_letter("y")
assert y is _ip.Frequency.YEARLY

h = _ip.Frequency.from_letter("h")
assert h is _ip.Frequency.HALFYEARLY

q = _ip.Frequency.from_letter("q")
assert q is _ip.Frequency.QUARTERLY

m = _ip.Frequency.from_letter("m")
assert m is _ip.Frequency.MONTHLY

d = _ip.Frequency.from_letter("d")
assert d is _ip.Frequency.DAILY

i = _ip.Frequency.from_letter("i")
assert i is _ip.Frequency.INTEGER

