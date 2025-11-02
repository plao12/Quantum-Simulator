"""
# Runner Module: Handles simulation execution
"""
module Runner
using ProgressMeter

export run_simulation

# Try to load CUDA safely
const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

"""
    run_simulation(model, data::NamedTuple, indexes::Vector{NTuple{5, Int}}) -> Nothing

Run a simulation using the provided `model` and `data` over a set of index tuples.

# Arguments
- `model`: A structure representing the simulation model. Must contain a `device` field (string) indicating whether to use `"gpu"` or `"cpu"` execution.
- `data::NamedTuple`: A NamedTuple containing data inputs required by the simulation. Its structure must be compatible with the model and runner functions.
- `indexes::Vector{NTuple{5, Int}}`: A vector of 5-element tuples, where each tuple represents a specific combination of indices for which the simulation should be run.

# Behavior
- Selects a GPU or CPU runner based on `model.device`.
- Runs the simulation for each set of indices in `indexes`.
- Uses multi-threading (`Threads.@threads`) for CPU execution and sequential iteration for GPU.

# Side Effects
- May update internal model state or write results depending on the behavior of the runner functions (not shown here).

# Returns
- `nothing`: This function returns nothing.
"""
function run_simulation(
    model,
    data::NamedTuple,
    indexes::Vector{NTuple{5,Int}};
)::Nothing
    # Initialize data and parameter indices
    prog = Progress(length(indexes))
    if model.U_interaction
        # Number operator expected value initialization
        # for self consistent model
        fill!(data.results[1].n, 0.5)
        fill!(data.results[2].n, 0.5)
    end

    # Choose runner based on GPU flag
    gpu_flag = model.device == "gpu"
    runner = gpu_flag ? create_gpu_runner(model, data) : create_cpu_runner(model, data)
    if gpu_flag
        for idx in eachindex(indexes)
            runner(indexes[idx])
            next!(prog)
        end
    else
        Threads.@threads for idx in eachindex(indexes)
            runner(indexes[idx])
            next!(prog)
        end
    end
    return nothing
end

function create_cpu_runner(model, data::NamedTuple)::Function
    return function runner(idx)
        (iTr, iU, iλ, ieV, iϵσ) = idx

        p = model.greens_function.init_p(
            ħω₀=4 / 5,
            λ=data.parameters.λ[iλ],
            U=data.parameters.U[iU],
            ϵσ=data.parameters.ϵ[iϵσ],
            ρ=[0.6, 0.6],
            T=[3.0, data.parameters.Tr[iTr]],
            eV=data.parameters.eV[ieV],
            Λ₀=0.0,
            Γ₀=1.0
        )
        for i in 1:2
            result = data.results[i]
            E = result.E
            nE = result.nE

            # Views into result arrays
            Aω_view = view(result.Aω, :, :, iTr, iU, iλ, ieV, iϵσ)
            Bω_view = view(result.Bω, :, :, iTr, iU, iλ, ieV, iϵσ)
            n_view = view(result.n, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)

            # Views for Green's functions
            G1less_view = view(result.G1less, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)
            G1great_view = view(result.G1great, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)
            G0less_view = view(result.G0less, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)
            G0great_view = view(result.G0great, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)

            model.greens_function.main_c(
                E, nE, p,
                Aω_view, Bω_view,
                G1less_view, G1great_view,
                G0less_view, G0great_view,
                n_view
            )
        end
        if model.level == 1
            model.interactions.anisotropy(
                data.results[1].E, data.results[2].E,
                data.results[1].nE, data.results[2].nE,
                view(data.results[1].G1less, :, iTr, iU, iλ, ieV, iϵσ), view(data.results[1].G1great, :, iTr, iU, iλ, ieV, iϵσ),
                view(data.results[2].G1less, :, iTr, iU, iλ, ieV, iϵσ), view(data.results[2].G1great, :, iTr, iU, iλ, ieV, iϵσ),
                view(data.interactions.anisotropy, iTr, iU, iλ, ieV, iϵσ)
            )
        else
            model.interactions.ising(
                view(data.results[1].E, :), view(data.results[2].E, :),
                data.results[1].nE, data.results[2].nE,
                view(data.results[1].G1less, :, :, :, iTr, iU, iλ, ieV, iϵσ), view(data.results[1].G1great, :, :, :, iTr, iU, iλ, ieV, iϵσ),
                view(data.results[2].G1less, :, :, :, iTr, iU, iλ, ieV, iϵσ), view(data.results[2].G1great, :, :, :, iTr, iU, iλ, ieV, iϵσ),
                view(data.interactions.ising, :, iTr, iU, iλ, ieV, iϵσ)
            )

            model.interactions.heisenberg(
                view(data.results[1].E, :), view(data.results[2].E, :),
                data.results[1].nE, data.results[2].nE,
                view(data.results[1].G1less, :, :, :, iTr, iU, iλ, ieV, iϵσ), view(data.results[1].G1great, :, :, :, iTr, iU, iλ, ieV, iϵσ),
                view(data.results[1].G0less, :, :, :, iTr, iU, iλ, ieV, iϵσ), view(data.results[1].G0great, :, :, :, iTr, iU, iλ, ieV, iϵσ),
                view(data.results[2].G1less, :, :, :, iTr, iU, iλ, ieV, iϵσ), view(data.results[2].G1great, :, :, :, iTr, iU, iλ, ieV, iϵσ),
                view(data.results[2].G0less, :, :, :, iTr, iU, iλ, ieV, iϵσ), view(data.results[2].G0great, :, :, :, iTr, iU, iλ, ieV, iϵσ),
                view(data.interactions.heisenberg, :, iTr, iU, iλ, ieV, iϵσ)
            )
        end
    end
end

function create_gpu_runner(model, data::NamedTuple)::Function

    # Transfer energy grids to GPU
    E1_gpu = CuArray(data.results[1].E)
    E2_gpu = CuArray(data.results[2].E)

    return function runner(idx)
        (iTr, iU, iλ, ieV, iϵσ) = idx

        p = model.greens_function.init_p(
            ħω₀=4 / 5,
            λ=data.parameters.λ[iλ],
            U=data.parameters.U[iU],
            ϵσ=data.parameters.ϵ[iϵσ],
            ρ=[0.6, 0.6],
            T=[3.0, data.parameters.Tr[iTr]],
            eV=data.parameters.eV[ieV],
            Λ₀=0.0,
            Γ₀=1.0
        )

        for (i, E_gpu) in enumerate((E1_gpu, E2_gpu))
            result = data.results[i]
            nE = result.nE

            # Views (CPU-side) for output arrays
            Aω_view = view(result.Aω, :, :, iTr, iU, iλ, ieV, iϵσ)
            Bω_view = view(result.Bω, :, :, iTr, iU, iλ, ieV, iϵσ)
            n_view = view(result.n, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)

            G1less_view = view(result.G1less, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)
            G1great_view = view(result.G1great, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)
            G0less_view = view(result.G0less, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)
            G0great_view = view(result.G0great, (model.level == 1 ? (:,) : (:, :, :))..., iTr, iU, iλ, ieV, iϵσ)

            # GPU version uses GPU energy array
            model.greens_function.main_c(
                E_gpu, nE, p,
                Aω_view, Bω_view,
                G1less_view, G1great_view,
                G0less_view, G0great_view,
                n_view
            )
        end
        if model.level == 1
            model.interactions.anisotropy(
                E1_gpu, E2_gpu,
                data.results[1].nE, data.results[2].nE,
                data.results[1].G1less[:, iTr, iU, iλ, ieV, iϵσ], data.results[1].G1great[:, iTr, iU, iλ, ieV, iϵσ],
                data.results[2].G1less[:, iTr, iU, iλ, ieV, iϵσ], data.results[2].G1great[:, iTr, iU, iλ, ieV, iϵσ],
                view(data.interactions.anisotropy, iTr, iU, iλ, ieV, iϵσ)
            )
        else
            1
        end
    end
end

end # module Runner