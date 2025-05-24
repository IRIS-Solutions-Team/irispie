
import sys
sys.path.append("../..")
from irispie import *
from irispie.parsers import preparser as pr_
import json
import numpy as np_


import nested
import model_functions as mf


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


with open("parameters.json", "r") as fid:
    p = Databank.from_dict(json.load(fid))


source_string = r"""
    !transition-variables
        q, z_unc, z, a, b

    !parameters
        ss_q, c2, c3, c4, c5

    !transition-equations
        q = glogc(-(z-1), ss_q, c2, c3, c4, c5);
        z_unc = 0.8*z_unc[-1] + (1-0.8)*1;
        z = maximum(z, 0);
        a = log(b);
        b = 1;
"""


context = {
    "segments": ["hh"],
    "glogc": glogc,
}


m = Model.from_string(
    source_string,
    context=context,
    linear=False,
    flat=True,
)

m.assign(
    ss_q = 0.02,
    q = 0.02,
    z_unc = 1,
    z = 1,
    b = 1,
    a = 0,
    c2 = p.c2_q_hh,
    c3 = p.c3_q_hh,
    c4 = p.c4_q_hh,
    c5 = p.c5_q_hh,
)

# m.alter_num_variants(3)

chk = m.check_steady(details=True, when_fails="error")

s = m.systemize()

