
import irispie as _ir
import numpy as _np
import statistics as _st
import sys

from irispie.series._arip import disaggregate_arip_data

num_years = 30
num_quarters = num_years * 4

start_date = _ir.qq(2021,1)

rng = _np.random.default_rng(0)

x0 = 1.01*_ir.exp(_ir.Series(
    dates=start_date>>start_date+num_quarters-1,
    func=rng.standard_normal) * 0.08
)
x0.cum_roc()

lx0 = _ir.log(x0)

form = "rate"
aggregation = "mean"
model = (form, aggregation, )

y0 = _ir.aggregate(x0, _ir.Freq.YEARLY, method=aggregation, )
y1 = _ir.aggregate(x0, _ir.Freq.YEARLY, method=_st.geometric_mean, )

x = _ir.disaggregate(y0, _ir.Freq.QUARTERLY, method="arip", model=model, )
x1 = _ir.disaggregate(y0, _ir.Freq.QUARTERLY, method="flat", )
# out.arip(_dates.Freq.QUARTERLY, "rate", "avg", )
# f = (x0 | x).plot()

low_from_to = y0.from_to
high_from_to = (low_from_to[0].convert(_ir.QUARTERLY, position="start", ), low_from_to[1].convert(_ir.QUARTERLY, position="end", ))
target_data = x0.get_data_from_to(high_from_to, )
target_data_0 = target_data.copy()
target_data_0[4:] = _np.nan
target_data_1 = target_data.copy()
target_data_1[0:3] = _np.nan
target_data_1[4:] = _np.nan
low_data = y0.get_data_from_to(low_from_to, )


high_data_0 = disaggregate_arip_data(
    (low_data, ),
    target_data_0,
    model,
    low_data.size,
    1, 4,
)

high_data_1 = disaggregate_arip_data(
    (low_data, ),
    target_data_1,
    model,
    low_data.size,
    1, 4,
)


