%% Robustness verification of a NN (L infinity adversarial attack)
%  if f(x) = y, then forall x' in X s.t. ||x - x'||_{\infty} <= eps,
%  then f(x') = y = f(x)

% Load network 
modelName = '/Users/samuel/milos/rgit/nnv2/nnv_py/examples/CompareMATLAB/outputs/fc_mnist.onnx';
netonnx = importONNXNetwork(modelName, "InputDataFormats", "BCSS", "OutputDataFormats", "BC");

% Create NNV model
net = matlab2nnv(netonnx);
net.OutputSize = 10;
numClasses = net.OutputSize;

%% 
% Load the data
load('/Users/samuel/milos/rgit/nnv2/nnv_py/examples/CompareMATLAB/test_sample.mat');
l = label + 1; % label gets loaded from the data

%%

% Verification settings
reachOptions = struct;
% reachOptions.reachMethod = "relax-star-area";
% reachOptions.relaxFactor = 0.5;
reachOptions.reachMethod = "approx-star";

%% ROBUST SAMPLE
eps = 1/255;

fprintf('Starting verification with epsilon %d \n', eps);

% Perform L_inf attack
lb_min = zeros(size(image)); % minimum allowed values for lower bound is 0
ub_max = ones(size(image)); % maximum allowed values for upper bound is 1
lb_clip = max((image-eps),lb_min);
ub_clip = min((image+eps), ub_max);
IS = ImageStar(lb_clip, ub_clip); % this is the input set we will use

output = net.evaluate(image);
[~, P] = max(output);

LB_output = net.evaluate(lb_clip);
[~, LBPred] = max(LB_output);

UB_output = net.evaluate(ub_clip);
[~, UBPred] = max(UB_output);

%%
t = tic;

% NEED THIS HERE SO MET EXISTS
try
    % run verification algorithm
    fprintf("Verification algorithm starting.\n")
    temp = net.verify_robustness(IS, reachOptions, l);
    fprintf("Verification algorithm finished.\n")
            
catch ME
    met = ME.message;
    temp = -1;
    fprintf(met);
end

p = profile('info');

res = temp;
time = toc(t);

fprintf("\n");
fprintf("Result : %d \n", res);
fprintf("Time: %f\n", time);

fprintf("Computing reach sets...\n")
% Get the reachable sets 
R = net.reachSet{end};

fprintf("Done computing reach sets! \n")


fprintf("Get the ranges for ub/lb \n")
% Get ranges for each output index
[lb_out, ub_out] = R.getRanges;
lb_out = squeeze(lb_out);
ub_out = squeeze(ub_out);

fprintf("Now to plotting! \n");

% Get middle point for each output and range sizes
mid_range = (lb_out + ub_out)/2;
range_size = ub_out - mid_range;

% Label for x-axis
x = [0 1 2 3 4 5 6 7 8 9];

% Visualize set ranges and evaluation points
f = figure;
errorbar(x, mid_range, range_size, '.', 'Color', 'b', 'LineWidth', 2);
hold on;
xlim([-0.5, 9.5]);
scatter(x, output, 30, 'x', 'MarkerEdgeColor', 'r');
title('Reachable Outputs');
xlabel('Label');
ylabel('Reachable Output Range on the Input Set');
% Save the figure
saveas(f, "outputs/approx/matlab_reach_stmnist_plot.png");

%% Save Results for Comparison with Python

% Create output directory if it doesn't exist
if ~exist('outputs/approx', 'dir')
    mkdir('outputs/approx');
end

% Save results to MAT file
results = struct();
results.test_label = label;
results.epsilon = eps;
results.reach_method = reachOptions.reachMethod;
% results.relax_factor = reachOptions.relaxFactor;
results.nominal_output = output;
results.lb_image_output = LB_output;
results.ub_image_output = UB_output;
results.nominal_pred = P;
results.lb_pred = LBPred;
results.ub_pred = UBPred;
results.output_lb = lb_out;
results.output_ub = ub_out;
results.mid_range = mid_range;
results.range_size = range_size;
results.num_output_stars = length(R);
results.computation_time = time;
results.robust = res;

save('outputs/approx/matlab_verify_fc_results.mat', 'results');
fprintf('✅ Results saved to: outputs/approx/matlab_verify_fc_results.mat\n');

% Save detailed text report
fid = fopen('outputs/approx/matlab_verify_fc_results.txt', 'w');
fprintf(fid, 'NNV-MATLAB VERIFICATION RESULTS\n');
fprintf(fid, 'Baseline implementation\n');
fprintf(fid, '======================================================================\n\n');

fprintf(fid, 'TEST CONFIGURATION:\n');
fprintf(fid, '  Test sample label: %d\n', label);
fprintf(fid, '  L∞ epsilon: %.10f\n', eps);
fprintf(fid, '  Reach method: %s\n', reachOptions.reachMethod);
% fprintf(fid, '  Relax factor: %.2f\n\n', reachOptions.relaxFactor);

fprintf(fid, 'EVALUATION RESULTS:\n');
fprintf(fid, '  Nominal prediction: %d\n', P);
fprintf(fid, '  Lower bound prediction: %d\n', LBPred);
fprintf(fid, '  Upper bound prediction: %d\n\n', UBPred);

fprintf(fid, 'NOMINAL OUTPUT LOGITS:\n');
for i = 1:numClasses
    if i == P
        marker = ' <-- PREDICTED';
    else
        marker = '';
    end
    fprintf(fid, '  Class %d: %20.10f%s\n', i-1, output(i), marker);
end
fprintf(fid, '\n');

fprintf(fid, 'REACHABILITY ANALYSIS:\n');
fprintf(fid, '  Number of output stars: %d\n', length(R));
fprintf(fid, '  Computation time: %.6f seconds\n', time);
if res == 1
    fprintf(fid, '  Robustness result: VERIFIED ROBUST (1)\n\n');
else
    fprintf(fid, '  Robustness result: NOT ROBUST (-1)\n\n');
end

fprintf(fid, 'OUTPUT BOUNDS:\n');
fprintf(fid, '%-10s %-20s %-20s %-15s\n', 'Class', 'Lower Bound', 'Upper Bound', 'Width');
fprintf(fid, '----------------------------------------------------------------------\n');
for i = 1:numClasses
    width = ub_out(i) - lb_out(i);
    if i == l
        marker = ' <-- TRUE CLASS';
    else
        marker = '';
    end
    fprintf(fid, '%-10d %-20.10f %-20.10f %-15.10f%s\n', i-1, lb_out(i), ub_out(i), width, marker);
end

fclose(fid);
fprintf('✅ Results saved to: outputs/approx/matlab_verify_fc_results.txt\n');

fprintf('\n======================================================================\n');
fprintf('COMPARISON INSTRUCTIONS:\n');
fprintf('======================================================================\n');
fprintf('1. Run the Python script: verify_fc.py\n');
fprintf('2. Compare the following values:\n');
fprintf('   - Output bounds (lb_out, ub_out)\n');
fprintf('   - Robustness result (res)\n');
fprintf('   - Number of output stars\n');
fprintf('   - Nominal output logits\n');
fprintf('\nExpected: Bounds should match within numerical precision (< 1e-6)\n');
fprintf('======================================================================\n');

%%

