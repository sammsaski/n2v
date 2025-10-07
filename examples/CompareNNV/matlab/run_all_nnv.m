%% Run All NNV Verification Experiments
%%
%% This script runs all verification scripts sequentially.
%% Experiments can be skipped by adding them to skipExperiments list.

clear; clc;

fprintf('Running all NNV verification experiments...\n\n');

% Add NNV to path (required when running in Docker container)
nnvPath = '/home/matlab/nnv/code/nnv';
if exist(nnvPath, 'dir')
    fprintf('Adding NNV to MATLAB path...\n');
    addpath(genpath(nnvPath));
    fprintf('NNV path added.\n\n');
end

% Change to CompareNNV directory
matlabDir = fileparts(mfilename('fullpath'));
global baseDir;  % Make baseDir available to child scripts
baseDir = fullfile(matlabDir, '..');
scriptsDir = fullfile(matlabDir, 'scripts');
cd(baseDir);

%% Configuration: Experiments to Skip
% Skip computationally intractable experiments (CNN exact reachability)
% These have thousands of ReLU neurons making exact analysis infeasible
skipExperiments = {
    'verify_cnn_conv_relu_exact',   % 608 ReLUs - exact is intractable
    'verify_cnn_avgpool_exact',     % 3168 ReLUs - exact is intractable
    'verify_cnn_maxpool_exact',     % 3168 ReLUs - exact is intractable
};

fprintf('Will skip %d intractable experiments\n\n', length(skipExperiments));

experiments = {
    'verify_fc_mnist_exact',
    'verify_fc_mnist_approx',
    'verify_fc_mnist_relax_star_area_0_25',
    'verify_fc_mnist_relax_star_area_0_50',
    'verify_fc_mnist_relax_star_area_0_75',
    'verify_fc_mnist_relax_star_range_0_25',
    'verify_fc_mnist_relax_star_range_0_50',
    'verify_fc_mnist_relax_star_range_0_75',
    'verify_fc_mnist_small_exact',
    'verify_fc_mnist_small_approx',
    'verify_cnn_conv_relu_exact',
    'verify_cnn_conv_relu_approx',
    'verify_cnn_conv_relu_relax_star_area_0_25',
    'verify_cnn_conv_relu_relax_star_area_0_50',
    'verify_cnn_conv_relu_relax_star_area_0_75',
    'verify_cnn_conv_relu_relax_star_range_0_25',
    'verify_cnn_conv_relu_relax_star_range_0_50',
    'verify_cnn_conv_relu_relax_star_range_0_75',
    'verify_cnn_avgpool_exact',
    'verify_cnn_avgpool_approx',
    'verify_cnn_avgpool_relax_star_area_0_25',
    'verify_cnn_avgpool_relax_star_area_0_50',
    'verify_cnn_avgpool_relax_star_area_0_75',
    'verify_cnn_avgpool_relax_star_range_0_25',
    'verify_cnn_avgpool_relax_star_range_0_50',
    'verify_cnn_avgpool_relax_star_range_0_75',
    'verify_cnn_maxpool_exact',
    'verify_cnn_maxpool_approx',
    'verify_cnn_maxpool_relax_star_area_0_25',
    'verify_cnn_maxpool_relax_star_area_0_50',
    'verify_cnn_maxpool_relax_star_area_0_75',
    'verify_cnn_maxpool_relax_star_range_0_25',
    'verify_cnn_maxpool_relax_star_range_0_50',
    'verify_cnn_maxpool_relax_star_range_0_75',
    'verify_toy_fc_4_3_2_zono',
    'verify_toy_fc_4_3_2_box',
    'verify_toy_fc_8_4_2_zono',
    'verify_toy_fc_8_4_2_box',
};

numExperiments = length(experiments);
fprintf('Total experiments: %d\n\n', numExperiments);

results = cell(numExperiments, 1);
times = zeros(numExperiments, 1);

for expIdx = 1:numExperiments
    fprintf('\n[%d/%d] Running: %s\n', expIdx, numExperiments, experiments{expIdx});
    fprintf('%s\n', repmat('=', 1, 70));

    % Check if experiment should be skipped
    if any(strcmp(experiments{expIdx}, skipExperiments))
        results{expIdx} = 'SKIPPED';
        times(expIdx) = 0;
        fprintf('SKIPPED: Experiment marked as intractable\n');
        continue;
    end

    try
        t = tic;
        run(fullfile(scriptsDir, [experiments{expIdx}, '.m']));
        times(expIdx) = toc(t);
        results{expIdx} = 'SUCCESS';
        fprintf('Completed in %.2f seconds\n', times(expIdx));
    catch ME
        times(expIdx) = toc(t);
        results{expIdx} = ['ERROR: ', ME.message];
        fprintf('FAILED: %s\n', ME.message);
    end
end

%% Summary
fprintf('\n\n%s\n', repmat('=', 1, 70));
fprintf('SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 70));

succeeded = sum(cellfun(@(x) strcmp(x, 'SUCCESS'), results));
skipped = sum(cellfun(@(x) strcmp(x, 'SKIPPED'), results));
failed = numExperiments - succeeded - skipped;

fprintf('Succeeded: %d/%d\n', succeeded, numExperiments);
fprintf('Skipped:   %d/%d\n', skipped, numExperiments);
fprintf('Failed:    %d/%d\n', failed, numExperiments);
fprintf('Total time: %.2f seconds\n', sum(times));

fprintf('\nDetails:\n');
for expIdx = 1:numExperiments
    fprintf('  %s: %s (%.2fs)\n', experiments{expIdx}, results{expIdx}, times(expIdx));
end