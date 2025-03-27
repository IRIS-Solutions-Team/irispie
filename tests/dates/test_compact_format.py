
import irispie as ir


def test_yearly():
    t = ir.yy(2023)
    assert t.to_compact_string() == "23Y"


def test_halfyearly():
    t = ir.hh(2023, 1)
    assert t.to_compact_string() == "23H1"


def test_quarterly():
    t = ir.qq(2023, 1)
    assert t.to_compact_string() == "23Q1"


def test_monthly():
    t = ir.mm(2023, 1)
    assert t.to_compact_string() == "23M01"


def test_daily():
    t = ir.dd(2023, 1, 25)
    assert t.to_compact_string() == "23Jan25"

