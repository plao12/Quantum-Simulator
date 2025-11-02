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

## Installation

Requirements
- Julia (recommended 1.8+)
- Git
- For GPU support: an NVIDIA GPU with compatible drivers and the CUDA toolkit installed (system-level).

Linux
1. Install system dependencies (example for Debian/Ubuntu):
   ```bash
   sudo apt update
   sudo apt install -y build-essential git wget ca-certificates
   ```
2. Install Julia (recommended: official binaries):
   ```bash
   # download latest stable tarball, adjust version as needed
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.x-linux-x86_64.tar.gz
   tar -xzf julia-1.10.*.tar.gz
   sudo mv julia-1.10.* /opt/julia-1.10
   sudo ln -s /opt/julia-1.10/bin/julia /usr/local/bin/julia
   ```
   Option: use your distro package manager but official binaries are recommended for up-to-date releases.
3. (Optional) GPU support — install NVIDIA drivers and CUDA following your distribution instructions. Verify drivers:
   ```bash
   nvidia-smi
   ```
4. Clone and instantiate the project:
   ```bash
   git clone <repo-url>
   cd Quantum-Simulator
   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
   ```
5. Verify CUDA from Julia (if using GPU):
   ```bash
   julia --project=. -e 'using Pkg; Pkg.add("CUDA"); using CUDA; println(CUDA.has_cuda()); CUDA.versioninfo()'
   ```

Windows
1. Install Git and Julia:
   - Download and run the Julia installer from https://julialang.org/downloads/ and check "Add Julia to PATH" during installation.
   - Install Git (https://git-scm.com/) or via Chocolatey: `choco install git`.
2. (Optional) GPU support — install NVIDIA drivers and the CUDA toolkit from NVIDIA. On Windows, ensure the driver and toolkit versions are compatible.
   - Verify with `nvidia-smi` from an elevated PowerShell or Command Prompt.
   - If using WSL2: follow NVIDIA and Microsoft instructions to enable CUDA in WSL2.
3. Clone and instantiate the project (PowerShell or Git Bash):
   ```powershell
   git clone <repo-url>
   cd Quantum-Simulator
   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
   ```
4. Verify CUDA in Julia:
   ```powershell
   julia --project=. -e "using Pkg; Pkg.add(\"CUDA\"); using CUDA; println(CUDA.has_cuda()); CUDA.versioninfo()"
   ```

Quick start (project)
- After installation, run examples from the Quick start section:
  ```julia
  include("launcher.jl")
  using .Launcher
  # ...example usage...
  ```

Troubleshooting
- If package instantiation fails, run in the Julia REPL:
  ```julia
  using Pkg
  Pkg.instantiate()
  Pkg.resolve()
  Pkg.precompile()
  ```
- For GPU issues, ensure system CUDA and driver versions match the requirements of CUDA.jl. Use `CUDA.versioninfo()` for diagnostics.
- On Windows, if building native packages fails, ensure required build tools (MSVC/Build Tools) are installed.

