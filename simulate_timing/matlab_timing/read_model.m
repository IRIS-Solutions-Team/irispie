% Code by OGResearch
% Read, steady-state and solve the GPM model with IRIS -- the `rm` step.
%
% Needs only IRIS (started from IRIS_DIR below unless already on the path)
% and smax0_.m in this folder; no ogi. Mirrors algo_read_model.m:213-238 for
% a nonlinear model: Model.fromFile with the parameters assigned, then
% sstate(..., 'Growth', true) with the action's solver options, then solve.
%
%   >> run('C:\SVN\ogi\gpm_migration\simulate_timing\matlab_timing\read_model.m')
%
% Writes results/gpm_model.mat, which simulate_matlab.m loads.
%
% There is no steady-state guess to choose here: IRIS's sstate gets through
% the near-unit-root inflation targets on its own, thanks to
% 'PseudoinvWhenSingular' (see the note in ../python_timing/read_model.py).

% the IRIS release the actions run on, as set in gpm_starter.m (Documents/MATLAB)
IRIS_DIR = 'C:/IRIS/IRIS-Toolbox-Release-20221026';
if ~exist('qq', 'file'), run(fullfile(IRIS_DIR, 'irisstartup.m')); end

here    = fileparts(mfilename('fullpath'));
inputs  = fullfile(here, 'inputs');
results = fullfile(here, 'results');
if ~exist(results, 'dir'), mkdir(results); end
addpath(here);   % smax0_.m, which the model calls

% ----------------------------------------------------------------- inputs
load(fullfile(inputs, 'gpm_parameters.mat'), 'p');

% ---------------------------------------------------- parse, steady, solve
tic;
m = Model.fromFile(fullfile(inputs, 'gpm.mod'), 'linear', false, 'assign', p);
fprintf('parse:   %8.2f s\n', toc);

% the action's solver options (algo_read_model.m:230-235)
solverOpt = { 'IRIS-Newton', ...
              'SpecifyObjectiveGradient', false, ...   % no analytical Jacobian
              'StepTolerance', Inf, ...
              'LastStepSizeOptim', 0, ...
              'PseudoinvWhenSingular', true, ...
              'Display', 'Iter' };
tic;
m = sstate(m, 'Growth', true, 'Solver', solverOpt);
fprintf('sstate:  %8.2f s\n', toc);

tic;
m = solve(m);
fprintf('solve:   %8.2f s\n', toc);

% ----------------------------------------------------------------- output
save(fullfile(results, 'gpm_model.mat'), 'm');
disp('wrote results/gpm_model.mat');
