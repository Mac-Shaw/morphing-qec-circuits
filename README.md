# morphing-qec-circuits
A package for morphing QEC circuits, as described in arXiv:2407.16336 and arXiv:2604.09797.

Contains helper functions to generate stim circuits corresponding to morphing QEC circuits. Contains a "MorphingSpecifications" class that stores the defining features of a morphing QEC circuit (mid-cycle code stabilisers, contraction circuits). This is then used to generate stim circuits corresponding to memory and stability experiments using the morphing specification. Also contains helper functions to calculate the static code distance and circuit-level distance. Finally, there are a number of morphing circuits for Bivariate Bicycle codes, colour codes, and toric codes, as described in `arXiv:2407.16336` and `arXiv:2604.09797`.

# Installation

This is a package but I haven't published it to PyPI (yet). So you need to:
  1. `git clone` the repository, then
  2. run `pip install .` to install the package.

Required packages:
- `numpy`, `galois`, and `numba` for general calculations.
- `gurobipy` for calculating code/circuit distances and for minimum-weight decoding (requires valid Gurobi license)
- `stim` and `surface_sim` for describing and simulating Clifford quantum circuits. `surface_sim` is available at `https://github.com/MarcSerraPeralta/surface-sim/tree/main`.
- `mle_decoder` for minimum-weight decoding. `mle_decoder` is available at `https://github.com/MarcSerraPeralta/mle-decoder`.

# Usage

See `example_notebook/Example_notebook.ipynb` for examples of usage.

# Contact

For troubleshooting issues, or general questions, contact me (Mackenzie Shaw) at `m.h.shaw@tudelft.nl` (this email will be deactivated after November 2026).
