
import pytest
import calendar as ca
import random as rn
import numpy as np
import irispie as ir

rn.seed(0)


daily_span = ir.dd(2020,1,1) >> ir.dd(2021,12,31)

parameter_names = ["method", "result_func"]

parameters = [
    ("sum", lambda num_days: (1 + num_days) * num_days / 2),
    ("mean", lambda num_days: (1 + num_days) / 2),
    (None, lambda num_days: (1 + num_days) / 2),
    ("first", lambda num_days: 1),
    ("last", lambda num_days: num_days),
]

# Daily to yearly

montly_span = ir.mm(2020,1) >> ir.mm(2021,12)
monthly_period_value = ((t, t.day) for t in daily_span)
periods, values = zip(*monthly_period_value)
DAILY_TO_MONTHLY_SERIES = ir.Series(periods=periods, values=values, )

@pytest.mark.parametrize(parameter_names, parameters, )
def test_daily_to_monthly(method, result_func, d=DAILY_TO_MONTHLY_SERIES, ):
    m = ir.aggregate(d, ir.MONTHLY, method=method, )
    for t in montly_span:
        _, num_days = ca.monthrange(t.year, t.period)
        assert m[t] == result_func(num_days, )


# Daily to yearly


def day_in_year(t):
    return t.serial - ir.daily_serial_from_ymd(t.year, 1, 1) + 1
yearly_span = ir.yy(2020) >> ir.yy(2021)
yearly_period_value = ((t, day_in_year(t)) for t in daily_span)
periods, values = zip(*yearly_period_value)
DAILY_TO_YEARLY_SERIES = ir.Series(periods=periods, values=values, )


@pytest.mark.parametrize(parameter_names, parameters, )
def test_daily_to_yearly(method, result_func, d=DAILY_TO_YEARLY_SERIES, ):
    m = ir.aggregate(d, ir.YEARLY, method=method, )
    for t in yearly_span:
        num_days = (ca.isleap(t.year) and 366) or 365
        assert m[t] == result_func(num_days, )


daily_span = ir.dd(2020,1,1) >> ir.dd(2021,12,31)
montly_span = ir.mm(2020,1) >> ir.mm(2021,12)
monthly_period_value = ((t, t.day if t.day != 15 else None) for t in daily_span)
periods, values = zip(*monthly_period_value)
DAILY_SERIES_MISSING = ir.Series(periods=periods, values=values, )


parameter_names = "method, result_func"

parameters = [
    ("sum", lambda num_days: sum(range(1, num_days + 1)) - 15),
    ("mean", lambda num_days: (sum(range(1, num_days + 1)) - 15) / (num_days - 1)),
    (None, lambda num_days: (sum(range(1, num_days + 1)) - 15) / (num_days - 1)),
    ("first", lambda num_days: 1),
    ("last", lambda num_days: num_days),
]

@pytest.mark.parametrize(parameter_names, parameters, )
def test_daily_to_monthly_missing(method, result_func, d=DAILY_SERIES_MISSING, ):
    m = ir.aggregate(d, ir.MONTHLY, method=method, remove_missing=True, )
    for t in montly_span:
        _, num_days = ca.monthrange(t.year, t.period)
        assert m[t] == result_func(num_days, )

