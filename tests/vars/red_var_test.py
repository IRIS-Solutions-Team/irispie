
import irispie as ir
import pytest
import os


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_FILE_PATH = os.path.join(_THIS_DIR, "fred_data.csv")

db = ir.Databox.from_csv_file(
    _DATA_FILE_PATH,
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
sim_span = estim_end+1 >> estim_end+8

db["covid_dummy"] = ir.Series(periods=estim_span, values=0, )
db["covid_dummy"][ir.qq(2020,2)] = 1

endogenous_names = ["ad2_y_tnd", "y_gap", "cpi", "rs"]
num_resamples = 1_000


def test_plain():
    v0 = ir.RedVAR(endogenous_names, order=2, )
    est_db0 = v0.estimate(db, estim_span, omit_missing=True, )
    acov0 = v0.get_acov(up_to_order=2, )
    acorr0 = v0.get_acorr(acov=acov0, )
    mean0 = v0.get_mean()
    db["covid_dummy"][sim_span] = 0
    sim_db0 = v0.simulate(db, estim_end+1 >> estim_end+8, prepend_input=False, )
    #
    rdb = v0.resample(
        est_db0,
        estim_span,
        "wild_bootstrap",
        num_variants=num_resamples,
    )
    #
    rv0 = v0.copy()
    #
    rv0.estimate(
        rdb,
        estim_span,
        omit_missing=True,
        num_variants=num_resamples,
    )
    #
    rc0 = rv0.get_acov(up_to_order=1, )
    rr0 = rv0.get_acorr(acov=rc0, )
    #
    rsim_db0 = rv0.simulate(
        db,
        estim_end+1 >> estim_end+8,
        prepend_input=True,
    )


d1 = ir.MinnesotaPriorObs(rho=0, mu2=num_estim_periods, )
d2 = ir.MeanPriorObs(mu2=num_estim_periods, mean=[0,0,0,0], )


def test_minnesota():
    v = ir.RedVAR(endogenous_names, order=2, )
    est_db = v.estimate(
        db, estim_span,
        omit_missing=True,
        prior_obs=d1,
        target_db=db,
    )
    mean = v.get_mean()


def test_mean():
    v = ir.RedVAR(endogenous_names, order=2, )
    est_db = v.estimate(
        db, estim_span,
        omit_missing=True,
        prior_obs=d2,
        target_db=db,
    )
    mean = v.get_mean()


def test_combined():
    v = ir.RedVAR(endogenous_names, order=2, )
    est_db = v.estimate(
        db, estim_span,
        omit_missing=True,
        prior_obs=(d1, d2),
        target_db=db,
    )
    mean = v.get_mean()

