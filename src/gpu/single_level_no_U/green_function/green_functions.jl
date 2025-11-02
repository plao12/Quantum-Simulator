module SingleLevelNoUGPU

include(joinpath("..", "..", "..", "..", "common", "gpu", "quantum_statistics", "quantum_statistics.jl"))
include(joinpath("..", "..", "..", "..", "common", "gpu", "integration_methods", "integration.jl"))
using .Integration
using .QuantumStatistics
using SpecialFunctions: besseli
using CUDA

export main_computation!

# Constants
const e = 1.6021773e-19             # Coulombs
const ħ = 4.135669e-12 / (2 * π)    # meVs
const kb = 8.61739e-2               # meVs/K
const μB = 5.788383e-2              # Bohr's Magneton (meV/T)
const g = 2.0                       # Giromagnetic ratio

# Define the parameters struct
struct parameters{T<:Float64}
    ħω₀::T                  # photon energy
    λ::T                    # electron-photon coupling
    U::T                    # electron-electron interaction
    ϵσ::T                   # electron energy
    ρ::Array{T,1}           # Spin polarization
    Λ::Array{T}             # Self-energy parameters
    Λl::Array{T}            # Left self-energy
    Λr::Array{T}            # Right self-energy
    Γ::Array{T}             # Spin-independent broadening
    Γl::Array{T}            # Left broadening
    Γr::Array{T}            # Right broadening
    Σ::Array{Complex{T}}    # Self Energy
    T::Array{T,1}           # Temperature
    μ::Array{T}             # Chemical potential
    μl::Array{T}            # Chemical potential left up down
    μr::Array{T}            # Chemical potential right up down
end

# Function for renormalization
function renormalization(x::T, y::T, z::T)::T where {T<:AbstractFloat}
    round(x - z * y, digits=5)
end

# Set the parameters
function init_parameters(;
    ħω₀=4 / 5, λ=0.0, U=0.0, ϵσ=0.0,
    ρ=[0.6, 0.6], T=[3.0, 3.0],
    eV=0.0, Λ₀=0.0, Γ₀=1.0
)::parameters
    U = renormalization(U, λ^2 / ħω₀, 2.0)
    ϵσ = renormalization(ϵσ, λ^2 / ħω₀, 1.0)

    Λl = Λ₀ * [1 + ρ[1] 1 - ρ[1]]
    Λr = Λ₀ * [1 + ρ[2] 1 - ρ[2]]
    Λ = Λl .+ Λr

    Γl = Γ₀ * [1 + ρ[1] 1 - ρ[1]]
    Γr = Γ₀ * [1 + ρ[2] 1 - ρ[2]]
    Γ = Γl .+ Γr
    Σ = Λ .- 0.5im * Γ
    μ = [1.0 1.0] .+ 0.5 * eV * [1.0 -1.0]
    μl = [μ[1] μ[1]]
    μr = [μ[2] μ[2]]
    parameters(ħω₀, λ, U, ϵσ, ρ, Λ, Λl, Λr, Γ, Γl, Γr, Σ, T, μ, μl, μr)
end

function compute_Gret!(ans, E, nE, n, ħω₀, ϵσ, amplitude, factor, Σ)
    i = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if i <= nE
        auxup = 0.0 + 0.0im
        auxdown = 0.0 + 0.0im
        Ei = E[i]
        Σup = Σ[1]
        Σdown = Σ[2]
        for l in 1:(2n+1)
            nl = -(n + 1 - l)
            f = factor[l]
            auxup += f / ((Ei) - ħω₀ * nl - (ϵσ + Σup))
            auxdown += f / ((Ei) - ħω₀ * nl - (ϵσ + Σdown))
        end
        ans[i, 1] = auxup * amplitude
        ans[i, 2] = auxdown * amplitude
    end
    return nothing
end

function compute_Gadv!(ans, Gret, nE)
    i = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if i <= nE
        ans[i, 1] = conj(Gret[i, 1])
        ans[i, 2] = conj(Gret[i, 2])
    end
    return nothing
end

function compute_Aω!(ans, Gret, nE)
    i = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if i <= nE
        ans[i, 1] = (-1 / π) * imag(Gret[i, 1])
        ans[i, 2] = (-1 / π) * imag(Gret[i, 2])
    end
    return nothing
end

function compute_Gless(ans, Gret, Gadv, nE, pl, pr, Γl, Γr)
    i = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if i <= nE
        for l ∈ 1:2
            auxGret = Gret[i, l]
            auxGadv = Gadv[i, l]
            aux = (pl[i, l] * auxGret * Γl[l] * auxGadv) + (pr[i, l] * auxGret * Γr[l] * auxGadv)
            aux *= 1im
            ans[i, l] = aux
        end
    end
    return nothing
end

function compute_Ggreat!(ans, Gret, Gadv, nE, pl, pr, Γl, Γr)
    i = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if i <= nE
        for l ∈ 1:2
            auxGret = Gret[i, l]
            auxGadv = Gadv[i, l]
            aux = ((1.0 - pl[i, l]) * auxGret * Γl[l] * auxGadv) + ((1.0 - pr[i, l]) * auxGret * Γr[l] * auxGadv)
            aux *= -1im
            ans[i, l] = aux
        end
    end
    return nothing
end

function compute_Bω!(ans, Ggreat, Gless, nE)
    i = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if i <= nE
        for j ∈ 1:2
            ans[i, j] = (-1 / 2π) * imag(Ggreat[i, j] .- Gless[i, j])
        end
    end
    return nothing
end

# Function to compute Green's functions
function main_computation(E, nE::Int, p::parameters,
    Aω, Bω, G1less, G1great, G0less, G0great, n_result
)::Nothing
    threads = 256
    blocks = Int(ceil(nE / threads))
    shared_mem_size = threads * sizeof(Float64)

    sizenE = (nE, 2)
    Gret = CUDA.zeros(ComplexF64, sizenE)
    Gadv = CUDA.zeros(ComplexF64, sizenE)
    Aω_gpu = CUDA.zeros(Float64, sizenE)

    Gless = CUDA.zeros(ComplexF64, sizenE)
    Ggreat = CUDA.zeros(ComplexF64, sizenE)
    Bω_gpu = CUDA.zeros(Float64, sizenE)

    pl = CUDA.zeros(Float64, nE, 2)
    pr = CUDA.zeros(Float64, nE, 2)

    @cuda threads = threads blocks = blocks fermi_dirac_d!(pl, kb, E, nE, CuArray(p.μl), p.T[1])
    @cuda threads = threads blocks = blocks fermi_dirac_d!(pr, kb, E, nE, CuArray(p.μr), p.T[2])

    β = 1 / (kb * 0.5 * sum(p.T))
    nB = bose_einstein_d(p.ħω₀, β)
    aux1 = (p.λ / p.ħω₀)^2
    amplitude = exp(-aux1 * (1 + 2 * nB))
    z = 2 * aux1 * (sqrt(nB * (nB + 1)))
    n = 10
    aux2 = CuArray([exp(-0.5 * i * p.ħω₀ * β) * besseli(abs(i), z) for i ∈ -n:n])
    @cuda threads = threads blocks = blocks compute_Gret!(Gret, E, nE, n, p.ħω₀, p.ϵσ, amplitude, aux2, CuArray(p.Σ))
    # Spectral function A(ω)
    @cuda threads = threads blocks = blocks compute_Aω!(Aω_gpu, Gret, nE)
    @cuda threads = threads blocks = blocks compute_Gadv!(Gadv, Gret, nE)
    Anor = CUDA.zeros(Float64, 1)
    @cuda threads = threads blocks = blocks shmem = shared_mem_size trapezoidal1d!(Anor, E, Aω_gpu[:, 1] .+ Aω_gpu[:, 2], nE)
    Gret ./= Anor
    Gadv ./= Anor
    Aω_gpu ./= Anor
    Aω .= Array(Aω_gpu)
    @cuda threads = threads blocks = blocks compute_Gless(Gless, Gret, Gadv, nE, pl, pr, CuArray(p.Γl), CuArray(p.Γr))
    @cuda threads = threads blocks = blocks compute_Ggreat!(Ggreat, Gret, Gadv, nE, pl, pr, CuArray(p.Γl), CuArray(p.Γr))
    # Nonequilibrium Spectral function B(ω)
    @cuda threads = threads blocks = blocks compute_Bω!(Bω_gpu, Ggreat, Gless, nE)
    Bnor = CUDA.zeros(Float64, 1)
    @cuda threads = threads blocks = blocks shmem = shared_mem_size trapezoidal1d!(Bnor, E, Bω_gpu[:, 1] .+ Bω_gpu[:, 2], nE)
    Gless ./= Bnor
    Ggreat ./= Bnor
    Bω_gpu ./= Bnor
    Bω .= Array(Bω_gpu)
    G1less .= Array(0.5 * (Gless[:, 1] .- Gless[:, 2]))
    G1great .= Array(0.5 * (Ggreat[:, 1] .- Ggreat[:, 2]))
    G0less .= Array(0.5 * (Gless[:, 1] .+ Gless[:, 2]))
    G0great .= Array(0.5 * (Ggreat[:, 1] .+ Ggreat[:, 2]))
    return nothing
end
end