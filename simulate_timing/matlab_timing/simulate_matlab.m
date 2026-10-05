% Code by OGResearch
% Time IRIS's stacked `simulate` on the GPM "Near term" increment.
%
% Run read_model.m first; it writes results/gpm_model.mat.
%
% Mirrors simulate_increment in algo_forecast.m: the simulation-ready input database the `fo` action saved
% (gpm_forecast_in.mat, tunes already overlaid, endogenous history clipped),
% and a plan that swaps every tuned endogenous variable with its shock.
%
% Needs only IRIS (started from IRIS_DIR below unless already on the path) -- no ogi. Run from anywhere:
%   >> run('C:\SVN\ogi\gpm_migration\simulate_timing\matlab_timing\simulate_matlab.m')
%
% Writes results/matlab_sim.csv (the variables listed in config.json) and,
% when ../python_timing/results/python_sim.csv exists too, plots both overlaid.
% Everything adjustable is in config.json; the span below must match it.

% the IRIS release the actions run on, as set in gpm_starter.m (Documents/MATLAB)
IRIS_DIR = 'C:/IRIS/IRIS-Toolbox-Release-20221026';
if ~exist('qq', 'file'), run(fullfile(IRIS_DIR, 'irisstartup.m')); end

here    = fileparts(mfilename('fullpath'));
inputs  = fullfile(here, 'inputs');
% the model calls smax0_; a copy of modules/codes/matlab/general/utils/smax0_.m sits in this folder
addpath(here);
results = fullfile(here, 'results');
if ~exist(results, 'dir'), mkdir(results); end

cfg = jsondecode(fileread(fullfile(here, 'config.json')));

% forecast span -- keep in step with first_fore / last_fore in config.json
rn = qq(2026, 2) : qq(2031, 3);

% ----------------------------------------------------------------- inputs
tic;
load(fullfile(results, 'gpm_model.mat'), 'm');         % solved by read_model.m
fprintf('load model:                 %8.2f s\n', toc);
load(fullfile(inputs, 'gpm_forecast_in.mat'), 'dbin');
load(fullfile(inputs, 'gpm_forecast_increments.mat'), 'r');
dtunes = r(end).dtunes;

% ------------------------------------------------------------------- plan
% (algo_forecast.m:584-616)
anticipated_shocks = cellstr(cfg.anticipated_shocks)';
P = Plan(m, rn, 'Anticipate', false);
P = anticipate(P, true, anticipated_shocks);
endonames  = access(m, 'transition-variables');
endostuned = intersect(intersect(endonames, fieldnames(dtunes)), fieldnames(cfg.endo_by_exo));
nswaps = 0;
for i = 1:numel(endostuned)
  endo = endostuned{i};
  exo  = cfg.endo_by_exo.(endo);
  ii = intersect(rn, find(~isnan(dtunes.(endo)))');
  for it = ii
    P = swap(P, it, {endo, exo});
    nswaps = nswaps + 1;
  end
end
fprintf('plan: %d swaps\n', nswaps);

% the same two solver option sets algo_forecast.m passes (fast, then robust)
s1 = solver.Options('Iris-Newton', 'SkipJacobUpdate', 2, 'FunctionTolerance', 1e-4, ...
                    'StepTolerance', Inf, 'FunctionNorm', Inf);
s2 = solver.Options('Iris-Newton', 'SkipJacobUpdate', 0, 'FunctionTolerance', 1e-4, ...
                    'StepTolerance', Inf, 'FunctionNorm', Inf);
solverSettings = [s1, s2];

% --------------------------------------------------------------- simulate
tic;
[dbout, info] = simulate(m, dbin, rn, ...
                         'Plan', P, ...
                         'Method', 'Stacked', ...
                         'Blocks', false, ...
                         'StartIterationsFrom', 'Data', ...
                         'Solver', solverSettings, ...
                         'SuccessOnly', true, ...
                         'PrependInput', true);
elapsed = toc;
fprintf('simulate (stacked):         %8.2f s   success=%d\n', elapsed, all(info.Success));

% ----------------------------------------------------------------- output
names = cellstr(cfg.plot_variables)';
fid = fopen(fullfile(results, 'matlab_sim.csv'), 'w');
fprintf(fid, 'period,%s\n', strjoin(names, ','));
dates = dat2str(rn);
for t = 1:numel(rn)
  vals = cellfun(@(n) dbout.(n)(rn(t)), names);
  fprintf(fid, '%s,%s\n', dates{t}, strjoin(arrayfun(@(v) sprintf('%.12g', v), vals, 'UniformOutput', false), ','));
end
fclose(fid);
fid = fopen(fullfile(results, 'matlab_timing.txt'), 'w');
fprintf(fid, 'simulate_seconds=%.3f\n', elapsed);
fclose(fid);

% ------------------------------------------------------------------- plot
pyfile = fullfile(here, '..', 'python_timing', 'results', 'python_sim.csv');
if exist(pyfile, 'file')
  py = readtable(pyfile);
  figure('Name', 'GPM stacked simulate: MATLAB vs Python');
  for i = 1:numel(names)
    subplot(2, 3, i);
    ml = dbout.(names{i})(rn);
    plot(1:numel(rn), ml, 'r-', 1:numel(rn), py.(names{i}), 'g--');
    title(names{i}, 'Interpreter', 'none');
    if i == 1, legend('MATLAB', 'Python'); end
    fprintf('%-14s max |matlab - py| = %.3e\n', names{i}, max(abs(ml - py.(names{i}))));
  end
else
  disp('no ../python_timing/results/python_sim.csv yet -- run simulate_python.py to get the overlay');
end
