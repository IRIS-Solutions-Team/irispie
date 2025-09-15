
import sys
import irispie as ir
import numpy as np
from irispie.chartpacks import main as ch
import copy


a = ch._Chart.from_string("Inflation: cpi [pct]")
b = ch._Chart.from_string("Inflation: cpi ")
c = ch._Chart.from_string("cpi ")
d = ch._Chart.from_string("cpi [pct]")


transforms = {
    "pct": ir.pct,
}

span = ir.qq(2021,1,...,2023,2)
highlight_span = tuple(ir.qq(2021,4,...,2028,4))

x = ch.Chartpack(
    transforms=transforms,
    tiles=(3,3),
    span=span,
    highlight=highlight_span,
    legend=["abc", "def", ],
)

f = x.add_figure("Figure 1", )
f.add_charts(["Inflation: cpi [pct]", "Short rate: stn"], )
f.add_chart("GDP growh: gdp [pct]", )
f.add_chart("GDP growh: gdp [pct]", )

f = x.add_figure("Figure 2", show_legend=False, )
f.add_chart("GDP growh: gdp [pct]", )
f.add_chart("GDP growh: gdp [pct]", )
f.add_chart("GDP growh: gdp [pct]", )
f.add_chart("GDP growh: gdp [pct]", )
f.add_chart("GDP growh: gdp [pct]", )
f.add_chart("GDP growh: gdp [pct]", )
f.add_chart("GDP growh: gdp [pct]", )

db = ir.Databox()
db["cpi"] = ir.Series(num_variants=2, dates=ir.qq(2020,1,...,2025,4), func=np.random.standard_normal, )
db["gdp"] = ir.Series(num_variants=2, dates=ir.qq(2020,1,...,2025,4), func=np.random.standard_normal, )
db["stn"] = ir.Series(num_variants=2, dates=ir.qq(2020,1,...,2025,4), func=np.random.standard_normal, )


transforms = {
    "pct": None,
}


fs = x.plot(db, return_info=True, )
sys.exit()

fs2 = x.plot(db, transforms=transforms, )

sh = {
    "type": "rect",
    "xref": "x3",
    "x0": highlight_span[0].to_plotly_date(position="start", ),
    "x1": highlight_span[-1].to_plotly_date(position="end", ),
    "yref": "y3 domain",
    "y0": 0,
    "y1": 1,
    "fillcolor": "rgba(0, 0, 0, 0.15)",
    "line": {"width": 0, },
}


sh1 = copy.deepcopy(sh)

sh1["x0"] = (highlight_span[0]+1).to_plotly_date(position="start", )
sh1["fillcolor"] = "rgba(255,0,0,0.15)"
sh1["xref"] = "x2"
sh1["yref"] = "y2 domain"


#fs[0].update_xaxes({"range":(span[0].to_plotly_date(),span[-1].to_plotly_date()), "autorange":False}, )#row=2,col=1)
#fs[0].add_shape(sh, )
#fs[0].add_shape(sh1, )


