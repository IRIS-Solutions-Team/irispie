
import irispie as ir


def test_regular_spans_shifted():
    x = ir.Series(periods=ir.qq(2020,1)>>ir.qq(2025,4), values=1, )
    y = ir.Series(periods=ir.qq(2021,4)>>ir.qq(2026,4), values=1, )
    (span, start, end) = ir.dates.get_encompassing_span(x, y, )
    assert span.start == x.start
    assert start == x.start
    assert span.end == y.end
    assert end == y.end


def test_regular_spans_nested():
    x = ir.Series(periods=ir.qq(2020,1)>>ir.qq(2026,4), values=1, )
    y = ir.Series(periods=ir.qq(2021,4)>>ir.qq(2025,4), values=1, )
    (span, start, end) = ir.dates.get_encompassing_span(x, y, )
    assert span.start == x.start
    assert start == x.start
    assert span.end == x.end
    assert end == x.end


def test_one_empty_series():
    x = ir.Series(periods=ir.qq(2020,1)>>ir.qq(2025,4), values=1, )
    y = ir.Series()
    (span, start, end) = ir.dates.get_encompassing_span(x, y, )
    assert span.start == x.start
    assert start == x.start
    assert span.end == x.end
    assert end == x.end


def test_both_empty_series():
    x = ir.Series()
    y = ir.Series()
    (span, start, end) = ir.dates.get_encompassing_span(x, y, )
    assert span.start is ir.start
    assert start is None
    assert span.end is ir.end
    assert end is None


