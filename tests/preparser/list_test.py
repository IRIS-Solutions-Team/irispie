
import irispie as ir


_SOURCE = r"""
a`n b`x c`z
!list(`n)
!list(`x)
!list(`z)
"""

def test():
    _, info = ir.parsers.preparser.from_string(_SOURCE, )
    actual = info['preparsed_source']
    expected = r"""
a b c
a
b
c
    """
    assert expected.strip() == actual.strip()



if __name__ == "__main__":
    test()

