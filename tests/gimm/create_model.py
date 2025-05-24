
import sys
sys.path.append("../..")
import json
import os
import numpy as np_
import scipy as sp_
import warnings

from irispie import *


def glogp(*args):
    y0, sigma, nu, low, high = args
    low = y0 + low
    high = y0 + high;
    mu = sigma * np_.log( ((y0-low)/(high-low)) ** (-1/np_.exp(nu)) - 1 );
    return mu, sigma, nu, low, high


def glogc(x, *args):
    mu, sigma, nu, low, high = glogp(*args)
    nu1 = np_.exp(nu)
    z = (x - mu) / sigma
    y = (1 + np_.exp(-z)) ** (-nu1)
    return low + (high - low) * y


def glogd(x, *args):
    mu, sigma, nu, low, high = glogp(*args)
    nu1 = np_.exp(nu)
    sigma1 = 1 / sigma
    z = (x - mu) * sigma1
    dy = nu1 * sigma1 * (1 + exp(-z))**(-nu1-1) * np_.exp(-z);
    return (high - low) * dy


with open(os.path.join("parameters", "parameters.json"), "r") as fid:
    p = json.load(fid)

with open(os.path.join("parameters", "steady.json"), "r") as fid:
    s = json.load(fid)


source_files = [
  "model-source/macroWorld.model",
  "model-source/macroLocal.model",
  "model-source/macroAssets.model",

  "model-source/connectCreditCreation.model",
  "model-source/connectCreditRisk.model",
  "model-source/connectInterestRates.model",

  "model-source/bankLoanPerformance.model",
  "model-source/bankProvisioning.model",
  "model-source/bankCapital.model",
]


context = {
    "segments": ["hh"],
    "glogc": glogc,
    "glogd": glogd,
}


m = Model.from_file(
    source_files,
    context=context,
    linear=False,
    flat=False,
)

m.assign(**p)
# m.assign(**s)

m.assign(
    py = 1,
    y = 1,
)

q = m._get_quantities(kind=TRANSITION_VARIABLE | MEASUREMENT_VARIABLE)
q = filter_quantities_by_name(q, exclude_names=["y", "py"])

s = m.create_steady_evaluator(quantities=q, print_iter=True)

r1 = sp_.optimize.root(
    s.eval_with_jacobian,
    s.initial_guess,
    method="lm",
    jac=True,
)

