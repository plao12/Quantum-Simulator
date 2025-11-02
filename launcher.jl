"""
# Molecular-Clean: A Julia package for molecular simulations
"""
module Launcher

include(joinpath("common", "caller.jl"))
include(joinpath("src", "caller.jl"))
include(joinpath("runner.jl"))

using .GreensCaller
using .InteractionCaller
using .Runner

const device_options = ["cpu", "gpu"]
const level_options = Dict(
    1 => "single_level",
    2 => "two_levels",
    3 => "three_levels"
)

function help()::Nothing
    println("""
    Options
    device:     Whether to select device (cpu or gpu)
    level:      molecular level (1–3):
                    1 → single_level / single_level_no_U
                    2 → two_levels / 
                    3 → three_levels / 
    U_interaction:    Whether to enable interaction term U (true or false)
    """
    )
    return nothing
end

struct Model
    device::String
    level::Int
    U_interaction::Bool
    path::String
    greens_function::NamedTuple
    interactions::NamedTuple
end

function init_model(; device::String, level::Int, U_interaction::Bool)::Model
    device = lowercase(device)

    function input_checker()::Bool
        state = true
        if !(device in device_options)
            println("Invalid device option. Choose from: $(device_options).")
            state = false
        end

        if !(level in keys(level_options))
            println("Invalid level option. Choose from: $(keys(level_options)).")
            state = false
        end

        if !isa(U_interaction, Bool)
            println("U_interaction must be a boolean value (true or false).")
            state = false
        end
        return state
    end

    if !input_checker()
        error("Input validation failed. Please check your inputs.")
    end

    aux_path = U_interaction ? level_options[level] : level_options[level] * "_no_U"
    model_path = joinpath("src", device, aux_path, "green_function", "green_functions.jl")
    if !isfile(model_path)
        error("Model path does not exist: $model_path")
    end
    return Model(
        device,
        level,
        U_interaction,
        model_path,
        init_greens_functions(device, level, U_interaction),
        init_interactions(device, level)
    )
end

function init_parameters_simulation(
    model::Model;
    parameters::NamedTuple=(
        Tr=(4.0,),
        U=model.U_interaction ? (0.5,) : (0.0,),
        λ=(0.5,),
        eV=(0.0,),
        ϵ=(0.0,)
    ),
    E::NamedTuple=(start=-10.0, stop=10.0, length=100)
)::Tuple{NamedTuple,Vector{NTuple{5,Int}}}

    function rand_space(start::T, stop::T, number::Int; factor::T=0.05)::Vector{T} where {T<:Float64}
        number ≥ 2 || throw(ArgumentError("Number of points must be at least 2."))
        Δ = (stop - start) / (number - 1)
        points = [start + i * Δ for i in 0:(number-1)]
        for i in 2:(number-1)
            points[i] += factor * Δ * rand([-1.0, 1.0]) * rand()
        end
        return points
    end

    function init_index(nparameters::NTuple{N,Int})::Vector{NTuple{N,Int}} where {N}
        ranges = (1:p for p in nparameters)
        return vec(collect(Iterators.product(ranges...)))
    end

    function init_results(E::Vector{Float64}, nE::Int64, nparameters::NTuple, ndots::Int)::NamedTuple
        n_AB = ((nE, 2)..., nparameters...)
        if ndots == 1
            n_GF = ((nE,)..., nparameters...)
            n_n = n_AB[2:end]
        else
            n_GF = ((ndots, ndots, nE)..., nparameters...)
            n_n = ((ndots, ndots, 2)..., nparameters...)
        end
        return (
            E=E,
            nE=nE,
            Aω=zeros(Float64, n_AB),
            Bω=zeros(Float64, n_AB),
            G1less=zeros(ComplexF64, n_GF),
            G1great=zeros(ComplexF64, n_GF),
            G0less=zeros(ComplexF64, n_GF),
            G0great=zeros(ComplexF64, n_GF),
            n=zeros(Float64, n_n)
        )
    end

    function init_interactions(ndots::Int, nparameters::NTuple{N,Int})::NamedTuple where {N}
        triangular(x::Int) = div(x * (x + 1), 2)
        ans_triangula = triangular(ndots)
        if ndots == 1
            return (anisotropy=zeros(Float64, nparameters),)
        else
            return (
                ising=zeros(Float64, ans_triangula, nparameters...),
                heisenberg=zeros(Float64, ans_triangula, nparameters...)
            )
        end

    end

    nparameters = Tuple(length(p) for p in parameters)

    # Simulation parameter ranges
    indexes = init_index(nparameters)
    margin = 50
    # Initialize results
    data = (
        init_results(rand_space(E.start, E.stop, E.length), E.length, nparameters, model.level),
        init_results(rand_space(E.start, E.stop, E.length + margin), E.length + margin, nparameters, model.level)
    )

    # Interaction storage
    interactions = init_interactions(model.level, nparameters)

    return (parameters=parameters, results=data, interactions=interactions), indexes
end

end # end Launcher
