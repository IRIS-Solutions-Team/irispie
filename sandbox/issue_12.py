
import irispie as ir

test = ir.Series(dates=ir.qq(2010, 1) >> ir.qq(2010, 4), values=1)
test.plot(span=ir.qq(2009, 1) >> ir.qq(2010, 4))

