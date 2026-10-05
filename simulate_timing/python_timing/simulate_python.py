# Code by OGResearch
"""
Time irispie's stacked-time `simulate` on the GPM "Near term" increment --
the `fo` step. Run read_model.py first; it writes results/gpm_model.h5.

The model is REBUILT from that h5 exactly as the ogi `fo` action rebuilds it
(utils/model_h5.load_model: parse the source, assign, check the saved steady
state, solve) -- the h5 holds no solved object, so this is a cost every
Python action pays and MATLAB's load of a .mat does not.

Needs only irispie (plus numpy, plotly) and the utils/ folder next to this
file; no ogi. Mirrors `simulate_increment` in general/algos/algo_forecast.py:
the input database is the one the `fo` action simulated (tunes already
overlaid, endogenous history clipped), and the plan swaps every tuned
endogenous variable with its shock.

    python simulate_python.py

Writes results/python_sim.csv (the variables in config.json) and, when
../matlab_timing/results/matlab_sim.csv exists too, results/compare.html
with both overlaid.
"""

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# irispie's solver printer emits U+2016, which a cp1252 console cannot encode
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import irispie as ir
from utils.model_functions import model_context_functions
from utils.model_h5 import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs")
RESULTS = os.path.join(HERE, "results")
MATLAB_RESULTS = os.path.join(HERE, "..", "matlab_timing", "results")

# the same solver settings algo_forecast.py passes
SOLVER_SETTINGS = {"func_tolerance": 1e-4, "step_tolerance": float("inf")}


def scalar(series: ir.Series, period) -> float:
    """First variant of a series at one period, NaN outside its span."""
    return float(np.asarray(series[period], dtype=float).ravel()[0])


def quarter(text: str):
    """"2026Q2" -> ir.qq(2026, 2)."""
    year, q = text.upper().split("Q")
    return ir.qq(int(year), int(q))


# ----------------------------------------------------------------- inputs
with open(os.path.join(HERE, "config.json")) as f:
    cfg = json.load(f)
span = quarter(cfg["first_fore"]) >> quarter(cfg["last_fore"])

t0 = time.perf_counter()
m = load_model(os.path.join(RESULTS, "gpm_model.h5"),
               context=model_context_functions(), linear=False)
print(f"load model (parse+check+solve): {time.perf_counter() - t0:8.2f} s")
dbin = ir.Databox.from_pickle_file(os.path.join(INPUTS, "dbin.pkl"))
dtunes = ir.Databox.from_pickle_file(os.path.join(INPUTS, "dtunes.pkl"))

# ------------------------------------------------------------------- plan
# a tune on an endogenous variable is imposed by swapping it with its shock
# in every period the tune has a value (algo_forecast.py, simulate_increment)
plan = ir.SimulationPlan(m, span)
transition_variables = set(m.get_names(kind=ir.TRANSITION_VARIABLE))
n_swaps = 0
for endo, exo in cfg["endo_by_exo"].items():
    if endo not in dtunes or endo not in transition_variables:
        continue
    swap = (plan.swap_anticipated
            if exo in cfg["anticipated_shocks"] else plan.swap_unanticipated)
    for period in span:
        if not np.isnan(scalar(dtunes[endo], period)):
            swap(period, (endo, exo))
            n_swaps += 1
print(f"plan: {n_swaps} swaps")

# --------------------------------------------------------------- simulate
t0 = time.perf_counter()
dbout = m.simulate(
    dbin,
    span,
    plan=plan,
    method="stacked",
    prepend_input=True,
    when_fails="error",
    solver_settings=SOLVER_SETTINGS,
)
elapsed = time.perf_counter() - t0
print(f"simulate (stacked):             {elapsed:8.2f} s")

# ----------------------------------------------------------------- output
names = cfg["plot_variables"]
periods = list(span)
with open(os.path.join(RESULTS, "python_sim.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["period"] + names)
    for t in periods:
        w.writerow([str(t)] + [scalar(dbout[n], t) for n in names])
with open(os.path.join(RESULTS, "python_timing.txt"), "w") as f:
    f.write(f"simulate_seconds={elapsed:.3f}\n")

# ------------------------------------------------------------------- plot
matlab_csv = os.path.join(MATLAB_RESULTS, "matlab_sim.csv")
if os.path.exists(matlab_csv):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    with open(matlab_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    fig = make_subplots(rows=2, cols=3, subplot_titles=names)
    x = [str(t) for t in periods]
    for i, n in enumerate(names):
        r, c = divmod(i, 3)
        py = np.array([scalar(dbout[n], t) for t in periods])
        ml = np.array([float(row[n]) for row in rows])
        fig.add_trace(go.Scatter(x=x,
                                 y=ml,
                                 name="MATLAB",
                                 line=dict(color="red"),
                                 showlegend=(i == 0)),
                      row=r + 1,
                      col=c + 1)
        fig.add_trace(go.Scatter(x=x,
                                 y=py,
                                 name="Python",
                                 line=dict(color="green", dash="dash"),
                                 showlegend=(i == 0)),
                      row=r + 1,
                      col=c + 1)
        print(f"{n:14s} max |py - matlab| = {np.nanmax(np.abs(py - ml)):.3e}")
    fig.update_layout(title="GPM stacked simulate: MATLAB vs Python", height=600)
    fig.write_html(os.path.join(RESULTS, "compare.html"))
    print("wrote results/compare.html")
else:
    print(
        "no ../matlab_timing/results/matlab_sim.csv yet -- run simulate_matlab.m to get the overlay"
    )
