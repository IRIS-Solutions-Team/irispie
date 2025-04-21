

import irispie as ir

db = ir.Databox()
db["x"] = ir.Series(start=ir.qq(2020,1), values=range(20))

fig = ir.make_subplots((2,3), )

db["x"].plot(figure=fig, subplot=0, show_figure=False, )
db["x"].plot(figure=fig, subplot=1, show_figure=False, )

