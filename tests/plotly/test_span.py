
import irispie as ir


db = ir.Databox()

db["x"] = ir.Series(start=ir.qq(2020,1), values=range(20), )
db["y"] = ir.Series(start=ir.qq(2015,1), values=range(60), )


# db["x"].plot()
span = ir.qq(2015,1,...,2030,4)

info = db["x"].plot(
    span=span,
    return_info=True,
    freeze_span=True,
    highlight=ir.qq(2020,1)>>ir.qq(2030,4),
    show_figure=False,
)


