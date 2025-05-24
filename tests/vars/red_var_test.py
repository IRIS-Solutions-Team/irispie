
import sys
import numpy as np
import irispie as ir
from loguru import logger

db = ir.Databox.from_csv_file(
    "fred_data.csv",
    period_from_string=ir.Period.from_iso_string,
    description_row=True,
)

db["y"] = 100 * ir.log(db["GDPC"])
db["cpi"] = 100 * ir.log(db["CPI"])
db["ad_y"] = 4 * ir.diff(db["y"])
db["ad_cpi"] = 4 * ir.diff(db["cpi"])
db["rs"] = db["TB3M"]

db["y_tnd"], db["y_gap"], = ir.hpf(db["y"], )
db["ad2_y_tnd"] = 4 * ir.diff(ir.diff(db["y_tnd"]))

estim_start = ir.qq(1950, 1) + 50
estim_end = ir.qq(2022, 4)
estim_span = estim_start >> estim_end
num_estim_periods = len(estim_span)

db["covid_dummy"] = ir.Series(periods=estim_span, values=0, )
db["covid_dummy"][ir.qq(2020,2)] = 1

d1 = ir.MinnesotaDum(rho=0, mu2=num_estim_periods, )
d2 = ir.MeanDum(mu2=num_estim_periods, mean=[10,0,100])

endogenous_names = ["ad2_y_tnd", "y_gap", "cpi", "rs"]
v0 = ir.RedVAR(endogenous_names, )
# v0 = ir.RedVAR(endogenous_names, exogenous_names=("covid_dummy", ), )
est_db0 = v0.estimate(db, estim_span, omit_missing=True, )

v1 = ir.RedVAR(endogenous_names, order=2, )
est_db1 = v1.estimate(db, estim_span, omit_missing=True, dummies=d1, target_db=db, )

acov0 = v0.get_acov(up_to_order=2, )

acorr0 = v0.get_acorr(acov=acov0, )
mean0 = v0.get_mean()
mean1 = v1.get_mean()

sim_span = estim_end+1 >> estim_end+8
db["covid_dummy"][sim_span] = 0

s = v0.simulate(db, estim_end+1 >> estim_end+8, prepend_input=False, )

#############

num_variants = 1_000

rdb = v0.resample(
    est_db0,
    estim_span,
    "wild_bootstrap",
    num_variants=num_variants,
    show_progress=True,
)

rv0 = v0.copy()

rv0.estimate(
    rdb,
    estim_span,
    omit_missing=True,
    num_variants=num_variants,
    show_progress=True,
)

rc0 = rv0.get_acov(up_to_order=1, )
rr0 = rv0.get_acorr(acov=rc0, )

rs = rv0.simulate(
    db,
    estim_end+1 >> estim_end+8,
    prepend_input=True,
    show_progress=True,
)

