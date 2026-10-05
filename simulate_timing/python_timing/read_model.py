# Code by OGResearch
"""
Read, steady-state and solve the GPM model with irispie -- the `rm` step.

Needs only irispie (plus numpy/scipy) and the utils/ folder next to this
file; no ogi. Mirrors general/algos/algo_read_model.py for a nonlinear model:
parse inputs/gpm.model with the context functions, assign the parameters,
steady() first, then solve().

    python read_model.py

Writes results/gpm_model.h5 the way the ogi `rm` action does (model
source + parameters + steady state, NOT a solved object -- see
utils/model_h5.py), which simulate_python.py rebuilds the model from.

STEADY_GUESS, below, chooses how the steady-state solver is started. Edit
the constant and rerun; see the note there.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# irispie's solver printer emits U+2016, which a cp1252 console cannot encode
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import irispie as ir
from utils.irispie_preparser_patch import patch_irispie_preparser
from utils.model_functions import model_context_functions
from utils.model_h5 import save_model

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs")
RESULTS = os.path.join(HERE, "results")

# ---------------------------------------------------- NOTE ON THE STEADY GUESS
# How the steady-state solver is started. This is an EXTENSION of the Python
# implementation; MATLAB's IRIS needs none of it.
#
# Motivation. GPM's inflation targets are near-unit-root AR(1)s,
#     D4L_CPI_TAR = rho*D4L_CPI_TAR[-1] + (1-rho)*ss_D4L_CPI_TAR + shocks
# with rho = 0.9999 (JP, MX) and 0.99999 (RC). irispie's non-flat steady
# state solves for a level AND a change, writing x[-1] as x - dx, so the
# residual is (1-rho)*(x - ss) + rho*dx: one equation in two unknowns with
# the level pinned only through (1-rho) ~ 1e-4. From irispie's default guess
# (1/9) the Levenberg solver either stalls (Block 69, D4L_CPI_TAR_JP) or lets
# dx absorb the level error and leaves the level wrong -- and every gap
# variable downstream with it. IRIS gets through the same singular Jacobian
# with sstate(..., 'PseudoinvWhenSingular', true), which irispie lacks.
#
# The real rm action (gpm_setparam.py) therefore seeds the solver by
# assigning D4L_CPI_TAR_<cc> = ss_D4L_CPI_TAR_<cc>, and RS_<cc> = RS_UNC_<cc>
# = 5 to start away from the smax0_ kink; those assignments are already in
# inputs/gpm_parameters.json. Later actions (sm, fo) additionally start from
# the steady state the rm action SAVED (inputs/gpm_steady_guess.json: level
# and change of all 1115 quantities, the outcome of rm's own steady()), so
# their solver starts at the answer.
#
#   "setparam"   what gpm_setparam.py seeds, nothing else   -- the rm action
#   "saved"      the rm action's saved steady state on top  -- sm / fo
#   "none"       strip the seeding: irispie's default guess everywhere
#
# Measured 2026-10-05: "setparam" and "saved" converge to the same steady
# state (19 s vs 10 s); "none" fails after ~2 s with "Steady state
# calculations failed to converge in [Variant 0][Block 69]". A fourth
# variant, ir.SteadyPlan.fix_levels on the D4L_CPI_TAR_<cc> without the
# rate seeding, was still inside steady() after 45 CPU-minutes and was
# killed. The question for irispie: can the steady solver handle a rho -> 1
# AR(1) in a non-flat steady state without the user seeding it?
STEADY_GUESS = "setparam"

# ----------------------------------------------------------------- inputs
with open(os.path.join(INPUTS, "gpm_parameters.json")) as f:
    p = json.load(f)

# the lists and strings (!for loops, switches) go to the parser as context,
# the numbers are assigned as parameters -- as algo_read_model.py does
context = model_context_functions()
parameters = {}
for k, v in p.items():
    if isinstance(v, (bool, list, str, dict)):
        context[k] = v
    else:
        parameters[k] = v

if STEADY_GUESS == "none":
    seeded = [k for k in parameters
              if k.startswith(("RS_", "RS_UNC_", "D4L_CPI_TAR_"))]
    for k in seeded:
        del parameters[k]
    print(f"removed {len(seeded)} seeding entries from the parameters")

# ---------------------------------------------------- parse, steady, solve
patch_irispie_preparser()

t0 = time.perf_counter()
m = ir.Simultaneous.from_file(
    os.path.join(INPUTS, "gpm.model"), linear=False, context=context)
print(f"parse:   {time.perf_counter() - t0:8.2f} s")

m.assign(parameters)
if STEADY_GUESS == "saved":
    with open(os.path.join(INPUTS, "gpm_steady_guess.json")) as f:
        # a [level, change] pair must reach `assign` as a tuple
        m.assign({k: tuple(v) if isinstance(v, list) else v
                  for k, v in json.load(f).items()})
elif STEADY_GUESS not in ("setparam", "none"):
    raise ValueError(f"unknown STEADY_GUESS {STEADY_GUESS!r}")
print(f"steady guess: {STEADY_GUESS}")

t0 = time.perf_counter()
m.steady()
print(f"steady:  {time.perf_counter() - t0:8.2f} s")
m.check_steady(when_fails="error", tolerance=1e-6)

t0 = time.perf_counter()
m.solve(tolerance=1e-10)
print(f"solve:   {time.perf_counter() - t0:8.2f} s")

# ----------------------------------------------------------------- output
os.makedirs(RESULTS, exist_ok=True)
t0 = time.perf_counter()
save_model(m, os.path.join(RESULTS, "gpm_model.h5"))
print(f"save h5: {time.perf_counter() - t0:8.2f} s   results/gpm_model.h5")
