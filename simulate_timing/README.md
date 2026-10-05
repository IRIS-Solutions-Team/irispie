# GPM read-model and simulate timing: MATLAB vs Python

Runs the two model steps of the GPM round on both engines, from the model
file onwards, with the inputs the gpm202608 actions used:

1. **read model** (`rm`): parse the model, assign parameters, steady state,
   first-order solution, save the model.
2. **simulate** (`fo`): one stacked-time nonlinear simulation of the
   "Near term" increment, with the action's plan and solver settings.

Neither side needs ogi. MATLAB needs IRIS, Python needs irispie-pe 0.79.6
(plus numpy, scipy, plotly). The `inputs/` folders are PRE-GENERATED: they
were exported once from the gpm202608 round by the OGResearch ogi
infrastructure (its `prepare_*_inputs` scripts, kept there), so nothing
here needs to be prepared -- run the two steps below as they are.

```text
matlab_timing/
  read_model.m              step 1: Model.fromFile + sstate + solve  -> results/gpm_model.mat
  simulate_matlab.m         step 2: simulate(..., 'Method', 'Stacked') -> results/matlab_sim.csv
  smax0_.m                  copy of modules/codes/matlab/general/utils/smax0_.m (the model calls it)
  config.json               span, increment, plotted variables, endo_by_exo, anticipated_shocks
  inputs/                   gpm.mod, gpm_parameters.mat, gpm_forecast_in.mat, gpm_forecast_increments.mat
python_timing/
  read_model.py             step 1: from_file + steady + solve + save -> results/gpm_model.h5
  simulate_python.py        step 2: load_model + simulate(method="stacked") -> results/python_sim.csv, compare.html
  utils/                    copies of the ogi bits, so the folder stands alone:
                            model_functions.py   context functions the equations call (erf, smax0_, ...)
                            irispie_preparser_patch.py   !if/!else fix gpm.model needs on irispie 0.79.6
                            model_h5.py          ogi's save_model / load_model: the <mc>_model.h5 format
  config.json               same content as the MATLAB one
  inputs/                   gpm.model, gpm_parameters.json, gpm_steady_guess.json, dbin.pkl, dtunes.pkl
```

## Run

```text
>> run('...\matlab_timing\read_model.m')
>> run('...\matlab_timing\simulate_matlab.m')

python python_timing/read_model.py
python python_timing/simulate_python.py
```

Whichever simulate runs second finds the other's CSV, prints the max
absolute difference per variable and plots both (MATLAB figure, or
`python_timing/results/compare.html`).

The IRIS release is the one `gpm_starter.m` (Documents/MATLAB) adds to the
path, `IRIS-Toolbox-Release-20221026`; the scripts start it from `IRIS_DIR`
when no IRIS is on the path yet.

## What is held equal

* Same model file (`gpm.mod` / `gpm.model`) and the same complete parameter
  set: the `rm` action's `presetparam` attribute plus `gpm_setparam`, exported
  once from ogi.
* Same input database: the saved `dbin` of the last increment (tunes
  overlaid, endogenous history clipped, swapped shocks blanked).
* Same plan: every tuned endogenous variable in `endo_by_exo` is swapped with
  its shock in each period the tune has a value, anticipated where the shock
  is in `anticipated_shocks`. Both scripts print the swap count (197).
* Same solver settings as the actions: `sstate` with the action's
  `solverOpt` in MATLAB, `steady()` defaults in irispie; simulate with
  `FunctionTolerance 1e-4`, `StepTolerance Inf` (MATLAB keeps its
  fast/robust option pair; irispie takes one set).

## The model file between the two steps

MATLAB saves the SOLVED object (`gpm_model.mat`) and `simulate_matlab.m`
just loads it. The ogi Python design saves `gpm_model.h5` with the model
source, the parameters and the steady state, NOT a solved object; every
consumer rebuilds the model from it (parse, assign, check the saved steady
state, solve). `utils/model_h5.py` is that saver and loader, copied from
ogi, and `simulate_python.py` goes through it exactly as the `fo` action
does. So the "load model" line on the Python side is a real rebuild and is
paid by every action; see the docstring of `model_h5.py` for why the h5 and
not a pickle.

## Measured 2026-10-05 (R2022a / IRIS 20221026 vs irispie-pe 0.79.6)

| step | MATLAB | Python |
|---|---|---|
| parse | 7.8 s | 2.7 s |
| steady state | 6.2 s | 3.2-3.5 s |
| first-order solve | 3.7 s | 7.5-12.9 s |
| save model | - | 0.3 s |
| load model before simulate | 1.4 s (load .mat) | 9.0 s (rebuild from .h5) |
| simulate (10 frames, 1 Newton step each) | 44-49 s | 19-23 s |

Max difference on the six plotted series is about 1e-4, i.e. the 1e-4
function tolerance plus the small input differences between the two rounds'
smoothers.

## The steady guess (Python only)

`python_timing/read_model.py` has a `STEADY_GUESS` constant, with a note
explaining it. GPM's near-unit-root inflation targets (rho = 0.9999) cannot
be pinned by irispie's level-plus-change steady state from the default
guess, so the real `rm` action seeds them (`gpm_setparam.py`), and `sm`/`fo`
start from the steady state `rm` saved. IRIS needs none of this
(`PseudoinvWhenSingular`). Edit the constant to test:

* `"setparam"` (default): what `gpm_setparam.py` seeds. Converges.
* `"saved"`: the `rm` action's saved steady state on top. Converges, same result.
* `"none"`: seeding stripped. Fails after ~2 s in Block 69 (D4L_CPI_TAR_JP).

## Adjusting

* Variables: `plot_variables` in both config.json files (2x3 panels, six names).
* Span: `first_fore`/`last_fore` in config.json AND `rn` in the MATLAB script.
* Another increment or another round needs the inputs re-exported from
  ogi (`gpm_migration/simulate_timing/*/prepare_*_inputs`), plus `r(end)`
  in the MATLAB script if the increment is not the last one.
