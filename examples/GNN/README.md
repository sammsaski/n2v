# GNN Verification

Reachability-based verification of graph neural networks for node-level
tasks.  Node features are perturbed inside an L_inf box, the perturbation is
propagated as a `GraphStar` through the graph-convolution layers, and output
properties are checked against the reach set.  When reachability cannot prove
a property, a falsifier searches for a concrete counterexample.

The implementation is a translation of NNV's MATLAB GNN verification
(GraphStar, GCN/SAGE/GINE layers) and is cross-checked against it in
`tests/soundness/test_cross_tool_*.py`.

## Examples

### `verify_pyg.py`: verify a PyTorch Geometric model

```bash
pip install -e ".[gnn]"      # installs torch_geometric
python examples/GNN/verify_pyg.py
```

Converts a PyG GCN with `GraphNeuralNetwork.from_pyg`, compares approx-star and
exact reach, and runs verify-or-falsify on a node-robustness property.

### `verify_gnn.py`: end-to-end pipeline on an IEEE power-flow model

```bash
python examples/GNN/verify_gnn.py                    # IEEE-24 SAGE (bundled)
python examples/GNN/verify_gnn.py --eps 0.01 --delta 0.05
python examples/GNN/verify_gnn.py \
    --checkpoint tests/fixtures/gnn/ieee118_pf/gine_pretrain_pf_ieee118.mat --targets 0 40 80
```

Loads a `.mat` checkpoint, checks forward parity against the saved PyTorch
predictions, runs full-graph and per-target k-hop subgraph reach, verifies
per-node local robustness (`|f(X') - f(X)| <= delta`), falsifies a spec the
model violates, and checks soundness by sampling.  The IEEE-24 and IEEE-118
GCN, SAGE, and GINE checkpoints bundled in `tests/fixtures/gnn/` all work.

## Usage

```python
from n2v.nn import GraphNeuralNetwork
from n2v.sets import GraphStar

# from a PyTorch Geometric model (GCNConv / SAGEConv / GINEConv layers)
net = GraphNeuralNetwork.from_pyg(model, data.edge_index, data.num_nodes,
                                  edge_attr=data.edge_attr,  # GINEConv only
                                  relu_hidden=True, relu_output=False)
# or from a .mat checkpoint (schema: n2v/utils/gnn_loader.py)
net = GraphNeuralNetwork.from_mat("model.mat")

X = data.x.numpy()
gs = GraphStar.from_bounds(X - eps, X + eps)
out = net.reach(gs)                              # list of output GraphStars
res = net.verify(gs, spec_lb, spec_ub)           # 'verified' / 'falsified' / 'unknown'
per_node = net.reach_subgraph(gs, target_nodes)  # k-hop subgraphs, scales to large graphs
```

`relu_hidden` / `relu_output` describe where the model's `forward` applies
ReLU; activations are not part of the PyG layer modules.

## Supported

| Component | Reach | Notes |
| --- | --- | --- |
| `GCNConv` | approx, exact | normalized or raw adjacency, optional edge weights |
| `SAGEConv` | approx, exact | `aggr` mean / add, with or without root weight |
| `GINEConv` (PyG) and Hu et al. GIN+E | approx | `nn` must be `Linear-ReLU-Linear`; edge-feature perturbation via `gine_graph_star` |
| ReLU between layers | approx (triangle), exact (splitting) | exact yields a union of GraphStars |
| Sum / mean graph pooling | exact (linear) | `n2v.nn.layer_ops.graph_pool_reach` |
| k-hop subgraph reach | same as full graph | per-target results are identical to full-graph reach |
| verify / falsify | | box node-feature specs; falsifier returns concrete witnesses |

Exact reach splits on every ReLU that can change sign, so the number of sets
grows exponentially with the number of such neurons; use it on small graphs.

## Limitations

* Node-level tasks only in `GraphNeuralNetwork`; graph-level models (pooling +
  readout) must be composed from the layer ops.
* No max pooling, no structural (edge add/remove) perturbations.
* Edge-feature perturbation is not exposed through `GraphNeuralNetwork`.
* The approx ReLU classifies neurons with interval bounds rather than NNV's
  per-neuron LP, so bounds are looser than NNV's at larger `eps` (sound, up to
  ~20x wider on individual outputs at `eps = 0.05` on IEEE-24 GCN).

## Components

* `n2v.sets.GraphStar`: node-feature Star set (`n2v/sets/graph_star.py`)
* `n2v.nn.GraphNeuralNetwork`: model wrapper (`reach`, `verify`, `reach_subgraph`)
* `n2v.nn.layer_ops.{gcn,sage,gine,graph_pool}_reach`: per-layer reach
* `n2v.utils.load_gnn_mat`, `n2v.utils.convert_pyg`: model loaders
* `n2v.utils.{gnn_verify,gnn_falsify,subgraph}`: specs, falsification, k-hop extraction
