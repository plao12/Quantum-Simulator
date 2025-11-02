# Quantum-Simulator

A Julia-based simulator for quantum molecular junctions: models of one-to-three molecular levels connected to two leads (left/right). Implements Green's function-based spectral and transport calculations on CPU and optional GPU backends, plus common interaction kernels (Heisenberg, Ising, anisotropy, Dzyaloshinskii–Moriya).

## Features
- Green's-function solvers for single-, two- and three-level models (CPU and GPU variants). See implementations:
  - [`SingleLevelCPU.main_computation`](src/cpu/single_level/green_function/green_functions.jl)
  - [`TwoLevelsCPUNoU.main_computation`](src/cpu/two_levels_no_U/green_function/green_functions.jl)
  - [`ThreeLevelsCPUNoU.main_computation`](src/cpu/three_levels_no_U/green_function/green_functions.jl)
  - GPU variant: [`SingleLevelNoUGPU.main_computation!`](src/gpu/single_level_no_U/green_function/green_functions.jl)
- Interaction kernels (CPU): [`Heisenberg.heisenberg!`](common/cpu/heisenberg/heisenberg.jl), [`Ising.ising!`](common/cpu/ising/ising.jl), [`UniaxialAnisotropy.anisotropy!`](common/cpu/uniaxial_anisotropy/uniaxial_anisotropy.jl), [`DzyaloshinskiiMoriya.dzyaloshinskii_moriya!`](common/cpu/dzyaloshinskii-moriya/dzyaloshinskii-moriya.jl) — NOTE: Dzyaloshinskii–Moriya kernel file is present but not yet activated/functional.
- Quantum statistics helpers: [`QuantumStatistics.bose_einstein_d`](common/cpu/quantum_statistics/quantum_statistics.jl), [`QuantumStatistics.fermi_dirac_d`](common/cpu/quantum_statistics/quantum_statistics.jl)
- Modular launcher + runner:
  - Project entry & helpers: [`Launcher`](launcher.jl)
  - Simulation runner: [`Runner.run_simulation`](runner.jl)
  - Interaction caller: [`InteractionCaller.init_interactions`](common/caller.jl)
  - Greens-function caller: [`GreensCaller`](src/caller.jl)

## Repository layout (key files)
- [launcher.jl](launcher.jl) — project entry, model selection, parameter setup
- [runner.jl](runner.jl) — execution loop, CPU/GPU dispatch
- [common/](common/) — shared interaction and utilities (CPU/GPU)
- [src/](src/) — greens-function implementations per device/level
- [usage_examples.ipynb](usage_examples.ipynb) — example workflows and plots
- [images/](images/) — SVG illustrations used in examples

## Quick start (Julia REPL)
1. Start Julia in the project root.
2. Run:
```julia
include("launcher.jl")
using .Launcher

model = Launcher.init_model(device="cpu", level=1, U_interaction=false)
data, indexes = Launcher.init_parameters_simulation(
    model;
    parameters=(Tr=(3.0,), U=(0.0,), λ=(0.5,), eV=(-10.0,), ϵ=(0.0,)),
    E=(start=-10.0, stop=10.0, length=500)
)
Launcher.run_simulation(model, data, indexes)
```

## Examples and visualization
- Interactive examples and plots are in [usage_examples.ipynb](usage_examples.ipynb). Plots call the simulation results and the plotting helpers in the notebook.

## Notes
- GPU code is conditional on CUDA availability; the project checks CUDA at runtime and falls back to CPU if unavailable. See [`common/caller.jl`](common/caller.jl) and [`runner.jl`](runner.jl) for dispatch logic.
- Greens-function modules export `main_computation` (or `main_computation!` for GPU). See the per-level files in [src/](src/).
