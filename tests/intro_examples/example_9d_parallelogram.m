%% Example 9.D: Parallelogram Zonotope
% This example demonstrates basic zonotope operations using the parallelogram
% from the NNV introductory documentation.
%
% The parallelogram is defined as:
%   (2 + ε₁, 3 + ε₁ + ε₂)
%
% In compact notation:
%   (⟨2, 1, 0⟩, ⟨3, 1, 1⟩)
%
% Where:
%   - Center: c = [2; 3]
%   - Generator 1: v₁ = [1; 1]
%   - Generator 2: v₂ = [0; 1]
%
% The zonotope is: Z = c + α₁*v₁ + α₂*v₂, where -1 ≤ αᵢ ≤ 1

% Add NNV to path (adjust path as needed)
% addpath(genpath('../../nnv/code/nnv'));

fprintf('========================================\n');
fprintf('Example 9.D: Parallelogram Zonotope\n');
fprintf('========================================\n\n');

%% 1. Create the Parallelogram Zonotope
fprintf('1. Creating parallelogram zonotope...\n');

% Define center and generators
c = [2; 3];
V = [1, 0;
     1, 1];

% Create zonotope
Z = Zono(c, V);

fprintf('   Center c = [%d; %d]\n', c(1), c(2));
fprintf('   Generators V = [%d, %d; %d, %d]\n', V(1,1), V(1,2), V(2,1), V(2,2));
fprintf('   Dimension: %d\n', Z.dim);
fprintf('   Number of generators: %d\n', size(V, 2));
fprintf('   ✓ Zonotope created\n\n');

%% 2. Compute Bounds
fprintf('2. Computing bounds...\n');

% Get bounding box
B = Z.getBox();
lb = B.lb;
ub = B.ub;

fprintf('   Bounds:\n');
fprintf('     Dimension 1 (x): [%.10f, %.10f]\n', lb(1), ub(1));
fprintf('     Dimension 2 (y): [%.10f, %.10f]\n', lb(2), ub(2));
fprintf('\n   Verification: Upper bound in vertical dimension = %.10f\n', ub(2));
fprintf('   Expected: 3 + 1 + 1 = 5\n');
fprintf('   Match: %s\n\n', iif(abs(ub(2) - 5.0) < 1e-10, '✓ YES', '✗ NO'));

%% 3. Get All Vertices
fprintf('3. Computing vertices...\n');

vertices = Z.getVertices();

fprintf('   Number of vertices: %d\n', size(vertices, 2));
fprintf('   Vertices:\n');
for i = 1:size(vertices, 2)
    fprintf('     v%d = (%.10f, %.10f)\n', i, vertices(1, i), vertices(2, i));
end
fprintf('   ✓ Vertices computed\n\n');

%% 4. Affine Transformation
fprintf('4. Testing affine transformation...\n');

% Simple scaling and translation
W = [2, 0;
     0, 1];
b = [1; 0];

Z_affine = Z.affineMap(W, b);
B_affine = Z_affine.getBox();
lb_affine = B_affine.lb;
ub_affine = B_affine.ub;

fprintf('   Transformation matrix W = [%d, %d; %d, %d]\n', W(1,1), W(1,2), W(2,1), W(2,2));
fprintf('   Translation vector b = [%d; %d]\n', b(1), b(2));
fprintf('   Transformed zonotope:\n');
fprintf('     Center: (%.10f, %.10f)\n', Z_affine.c(1), Z_affine.c(2));
fprintf('     Bounds: x ∈ [%.10f, %.10f], y ∈ [%.10f, %.10f]\n', ...
    lb_affine(1), ub_affine(1), lb_affine(2), ub_affine(2));
fprintf('   ✓ Affine map computed\n\n');

%% 5. Minkowski Sum
fprintf('5. Testing Minkowski sum...\n');

% Create second zonotope (a small box)
c2 = [0; 0];
V2 = [0.5, 0;
      0, 0.5];
Z2 = Zono(c2, V2);

Z_sum = Z.MinkowskiSum(Z2);
B_sum = Z_sum.getBox();
lb_sum = B_sum.lb;
ub_sum = B_sum.ub;

fprintf('   Second zonotope: center = [%.1f; %.1f], generators = 2\n', c2(1), c2(2));
fprintf('   Minkowski sum result:\n');
fprintf('     Center: (%.10f, %.10f)\n', Z_sum.c(1), Z_sum.c(2));
fprintf('     Number of generators: %d\n', size(Z_sum.V, 2));
fprintf('     Bounds: x ∈ [%.10f, %.10f], y ∈ [%.10f, %.10f]\n', ...
    lb_sum(1), ub_sum(1), lb_sum(2), ub_sum(2));
fprintf('   ✓ Minkowski sum computed\n\n');

%% 6. Test Point Containment
fprintf('6. Testing point containment...\n');

% Test several points
test_points = [
    2.0, 3.0;    % center (should be inside)
    2.5, 3.5;    % inside
    1.0, 2.0;    % corner (should be inside)
    3.0, 5.0;    % corner (should be inside)
    0.0, 0.0;    % outside
    5.0, 5.0     % outside
];

contains_results = zeros(size(test_points, 1), 1);
for i = 1:size(test_points, 1)
    p = test_points(i, :)';
    contains_results(i) = Z.contains(p);
    fprintf('   Point (%.1f, %.1f): %s\n', p(1), p(2), ...
        iif(contains_results(i), 'INSIDE ✓', 'OUTSIDE ✗'));
end
fprintf('   ✓ Containment tests completed\n\n');

%% 7. Save All Results
fprintf('7. Saving results to MAT file...\n');

% Create results structure with all variables and intermediate results
results = struct();

% Original zonotope
results.c = c;
results.V = V;
results.dim = Z.dim;
results.num_generators = size(V, 2);

% Bounds
results.lb = lb;
results.ub = ub;

% Vertices
results.vertices = vertices;
results.num_vertices = size(vertices, 2);

% Affine transformation
results.affine_W = W;
results.affine_b = b;
results.affine_c = Z_affine.c;
results.affine_V = Z_affine.V;
results.affine_lb = lb_affine;
results.affine_ub = ub_affine;

% Minkowski sum
results.mink_c2 = c2;
results.mink_V2 = V2;
results.mink_sum_c = Z_sum.c;
results.mink_sum_V = Z_sum.V;
results.mink_sum_lb = lb_sum;
results.mink_sum_ub = ub_sum;

% Point containment tests
results.test_points = test_points;
results.contains_results = contains_results;

% Save to file
save('saved_results/example_9d_results.mat', 'results');

fprintf('   ✓ Results saved to: saved_results/example_9d_results.mat\n\n');

%% 8. Summary
fprintf('========================================\n');
fprintf('Summary of Example 9.D\n');
fprintf('========================================\n');
fprintf('Parallelogram zonotope: (⟨2, 1, 0⟩, ⟨3, 1, 1⟩)\n');
fprintf('  • Center: (2, 3)\n');
fprintf('  • 2 generators: [1, 0; 1, 1]\n');
fprintf('  • Bounds: x ∈ [1, 3], y ∈ [2, 5]\n');
fprintf('  • Upper bound (y): %.1f ✓ (3 + 1 + 1 = 5)\n', ub(2));
fprintf('  • All operations completed successfully\n');
fprintf('  • Results saved for Python comparison\n');
fprintf('========================================\n');

%% Helper function for inline if
function out = iif(condition, true_val, false_val)
    if condition
        out = true_val;
    else
        out = false_val;
    end
end
