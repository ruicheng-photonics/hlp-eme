# Examples

This folder provides examples of inverse Bragg grating (IBG) simulations for both straight and spiral waveguides:

- `straight_ibg_sim.ipynb`
- `spiral_ibg_sim.ipynb`

The straight-grating notebook includes three examples:

1. A single-channel square filter
2. A two-channel linear-edge filter
3. A Gaussian-apodized grating

To run a specific example, uncomment the corresponding parameter block in the **Grating Parameters** section.

For a first-time run, we recommend starting with the Gaussian-apodized grating example. It is the shortest example and typically requires the least computational time.

## Grating Profiles

To model and simulate the grating structures, discrete complex coupling profiles $q_n$ sampled at a one-grating-period pitch ($\Lambda$) are required.

Pre-computed grating profiles for the **single-channel square filter** and **two-channel linear-edge filter** are provided in the `grating_libs/` directory. These profiles were synthesized via the discrete layer-peeling algorithm (see [LPA](https://github.com/SiEPIC/SiEPIC-Tools/tree/master/Simulation/Bragg_Layer_Peeling) for implementation details) and subsequently re-sampled to match the one-period grid spacing.

Note: The current pipeline assumes profiles sampled strictly at one grating period. Future updates will extend the functions in `grating_struct_funs/` to support arbitrary spatial sampling resolutions.

## Notes

**IMPORTANT**: Lumerical 2022 R2 or later is strongly recommended. Starting from Lumerical 2022 R2, solver tasks can be distributed across multiple concurrent processes, which can substantially improve simulation efficiency. See the [EME Benchmark](https://optics.ansys.com/hc/en-us/articles/10138214556179-EME-Performance-Benchmarks) for details.