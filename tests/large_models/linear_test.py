

import irispie as _ir


_SOURCE = r"""
!variables
    !for ? = <range(1000)> !do
        x?
    !end
!equations
    !for ?i = <range(1000)> !do
        x?i = !for ?j = <range(1,1000,70)> !do + x?j[-3] !end;
    !end
"""

def test():

    m = _ir.Model.from_string(_SOURCE, linear=True, )
    m.solve_steady()
    s = m.get_steady()

    for n in ("x"+str(i) for i in range(1000)):
        assert abs(s[n][0]) < 1e-10
        assert abs(s[n][1]) < 1e-10


if __name__ == "__main__":
    test()

