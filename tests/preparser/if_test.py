
import irispie as ir

_SOURCE = """
!equations{:abc :_D}
    !for ?x = 0, 1, 11, 2 !do
        !if aaa == ?x !then
            x = 1;
            !if aaa > 10 !then
                y = 1;
            !end
        !else
            !for ?w = a, b, c !do
                u?w?x = 2;
            !end
        !end
    !end
!equations
    z = 100;
!variables{:main}
    a
"""

def test():
    context = {"aaa": 11}
    _, info = ir.parsers.preparser.from_string(_SOURCE, context=context, )
    actual = info['preparsed_source']
    expected = """
!equations{:abc :_D}
                ua0 = 2;
                ub0 = 2;
                uc0 = 2;
                ua1 = 2;
                ub1 = 2;
                uc1 = 2;
            x = 1;
                y = 1;
                ua2 = 2;
                ub2 = 2;
                uc2 = 2;
!equations
    z = 100;
!variables{:main}
    a
"""
    assert actual.strip() == expected.strip()


if __name__ == "__main__":
    test()


