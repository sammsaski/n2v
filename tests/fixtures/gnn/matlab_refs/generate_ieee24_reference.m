function generate_ieee24_reference(varargin)
% generate_ieee24_reference - dump GNNV reach bounds for cross-tool diff
%
% Loads an IEEE-24 GNN checkpoint, runs NNV GNN reach with approx-star on a
% set of (instance, eps) pairs, and saves the output (lb, ub) per instance to
% a single .mat file consumed by the n2v cross-tool tests
% (tests/soundness/test_cross_tool_{gnn,sage,gine}.py).  The bundled
% references were generated from the full checkpoints and copied here by
% ../make_fixtures.py.
%
% Usage:
%   generate_ieee24_reference()                                   % defaults
%   generate_ieee24_reference('instances', [1 2 3])               % subset
%   generate_ieee24_reference('eps_values', [1e-3 1e-2])          % subset
%   generate_ieee24_reference('output', '/path/to/ref.mat')       % override
%   generate_ieee24_reference('checkpoint', '/path/to/sage_pf_ieee24.mat')
%
% NNV with GNN support (gnn2nnv, GraphStar) must already be on the MATLAB
% path (run startup_nnv.m once first).
%
% Author: Anne Tumlin

%% ---- arg parsing -------------------------------------------------------
p = inputParser;
addParameter(p, 'checkpoint', ...
    fullfile(fileparts(mfilename('fullpath')), '..', 'ieee24_pf', 'gcn_pf_ieee24.mat'), @ischar);
addParameter(p, 'output', ...
    fullfile(fileparts(mfilename('fullpath')), 'gcn_ieee24_reference.mat'), @ischar);
addParameter(p, 'instances', [1 2 3 4 5], @isnumeric);   % 1-based MATLAB indices
addParameter(p, 'eps_values', [1e-3, 1e-2, 5e-2], @isnumeric);
addParameter(p, 'lp_solver', 'linprog', @ischar);
parse(p, varargin{:});
opt = p.Results;

fprintf('Loading checkpoint: %s\n', opt.checkpoint);
[gnn, test_data] = gnn2nnv(opt.checkpoint);

n_inst = numel(opt.instances);
n_eps = numel(opt.eps_values);

% Each (instance, eps) cell contains lb / ub matrices of shape (N, F_out).
ref_lb = cell(n_inst, n_eps);
ref_ub = cell(n_inst, n_eps);
ref_center = cell(n_inst, n_eps);
times_sec = zeros(n_inst, n_eps);

for ii = 1:n_inst
    inst_idx = opt.instances(ii);
    X = double(test_data.X_all{inst_idx});
    [N, F] = size(X);

    for jj = 1:n_eps
        eps_val = opt.eps_values(jj);
        fprintf('instance %d, eps %.1e ... ', inst_idx, eps_val);

        LB = -eps_val * ones(N, F);
        UB =  eps_val * ones(N, F);
        gs_in = GraphStar(X, LB, UB);

        reach_opt = struct();
        reach_opt.reachMethod = 'approx-star';
        reach_opt.lp_solver = opt.lp_solver;
        reach_opt.numCores = 1;

        t = tic;
        gs_out = gnn.reach(gs_in, reach_opt);
        times_sec(ii, jj) = toc(t);

        [lb_out, ub_out] = gs_out.getRanges(opt.lp_solver);
        ref_lb{ii, jj} = lb_out;
        ref_ub{ii, jj} = ub_out;
        ref_center{ii, jj} = gs_out.V(:, :, 1);

        fprintf('done (%.2fs), bound width mean=%.3e max=%.3e\n', ...
                times_sec(ii, jj), mean(ub_out(:) - lb_out(:)), ...
                max(ub_out(:) - lb_out(:)));
    end
end

reference = struct();
reference.checkpoint   = opt.checkpoint;
reference.instances    = opt.instances(:);
reference.eps_values   = opt.eps_values(:);
reference.lp_solver    = opt.lp_solver;
reference.lb           = ref_lb;
reference.ub           = ref_ub;
reference.center       = ref_center;
reference.times_sec    = times_sec;
reference.matlab_version = version();
reference.nnv_method   = 'approx-star';
reference.generated_at = char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));

fprintf('\nSaving %s\n', opt.output);
save(opt.output, '-struct', 'reference');
fprintf('Done.\n');
end
