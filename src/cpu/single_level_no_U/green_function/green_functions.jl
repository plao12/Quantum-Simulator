module SingleLevelNoUCPU

include(joinpath("..", "..", "..", "..", "common", "cpu", "quantum_statistics", "quantum_statistics.jl"))
include(joinpath("..", "..", "..", "..", "common", "cpu", "integration_methods", "integration.jl"))
using .Integration
using .QuantumStatistics
using SpecialFunctions: besseli

export main_computation

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
    ϵσ::T                   # molecular energy
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

# PhSet the parameters
function init_parameters(;
    ħω₀=4 / 5, λ=0.0, U=0.0, ϵσ=0.0,
    ρ=[0.6, 0.6], T=[3.0, 3.0],
    eV=0.0, Λ₀=0.0, Γ₀=1.0,
    Bz=[g * μB]
)::parameters
    U = renormalization(U, λ^2 / ħω₀, 2.0)
    ϵσ = renormalization(ϵσ, λ^2 / ħω₀, 1.0)
    ϵσ += 0.0 * Bz[1]

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

# Kernel function for Green's function calculation
function kernel(ϵ::Array{Float64}, aux::Array{Complex{Float64}})::Array{Complex{Float64}}
    1 ./ (ϵ .- aux)
end

# Function to compute Green's functions
function main_computation(E, nE::Int, p::parameters,
    Aω, Bω, G1less, G1great, G0less, G0great, n_result
)::Nothing
    sizenE = (nE, 2)
    Gret = zeros(Complex{Float64}, sizenE)
    Gadv = zeros(Complex{Float64}, sizenE)

    Gless = zeros(Complex{Float64}, sizenE)
    Ggreat = zeros(Complex{Float64}, sizenE)

    β = 1 / (kb * 0.5 * sum(p.T))
    pl = fermi_dirac_d(E, p.μl, kb, p.T[1])
    pr = fermi_dirac_d(E, p.μr, kb, p.T[2])

    nB = bose_einstein_d(p.ħω₀, β)
    aux1 = (p.λ / p.ħω₀)^2
    amplitude = exp(-aux1 * (1 + 2 * nB))
    z = 2 * aux1 * (sqrt(nB * (nB + 1)))

    n = 10
    aux2 = [exp(-0.5 * i * p.ħω₀ * β) * besseli(abs(i), z) for i ∈ -n:n]
    aux3 = p.ϵσ .+ p.Σ
    aux4 = E .- p.ħω₀ * reshape(range(-n, n), 1, 2n + 1)
    for l ∈ 1:(2n+1)
        Gret += aux2[l] * kernel(aux4[:, l], aux3)
    end
    Gret .*= amplitude

    # Spectral function A(ω)
    Aω .= (-1 / π) * imag(Gret)
    Anor = trapezoidal_1d(E, Aω[:, 1] + Aω[:, 2])[1]
    Gret ./= Anor
    Gadv .= conj(Gret)
    Aω ./= Anor

    # Nonequilibrium Spectral function B(ω)
    aux5 = @. Gret * p.Γl * Gadv
    aux6 = @. Gret * p.Γr * Gadv

    Gless .= @. (pl * aux5) + (pr * aux6)
    Gless .*= 1im

    Ggreat .= @. ((1 - pl) * aux5) + ((1 - pr) * aux6)
    Ggreat .*= -1im

    Bω .= (-1 / 2π) * imag(Ggreat .- Gless)
    Bnor = trapezoidal_1d(E, Bω[:, 1] + Bω[:, 2])[1]
    Gless ./= Bnor
    Ggreat ./= Bnor
    Bω ./= Bnor

    local_nbar = (1 / 2π) * reshape(trapezoidal_1d(E, imag(Gless)), 1, 2)
    n_result[1] = local_nbar[1]
    n_result[2] = local_nbar[2]

    G1less .= 0.5 .* (Gless[:, 1] .- Gless[:, 2])
    G1great .= 0.5 .* (Ggreat[:, 1] .- Ggreat[:, 2])
    G0less .= 0.5 .* (Gless[:, 1] .+ Gless[:, 2])
    G0great .= 0.5 .* (Ggreat[:, 1] .+ Ggreat[:, 2])
    return nothing
end

end