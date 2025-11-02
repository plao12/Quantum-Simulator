module TwoLevelsCPUNoU

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
    ϵσ::Array{T}            # molecular energy
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
    γ::Array{T}             # Coupling
end

# Function for renormalization
function renormalization(x::T, y::T, z::T)::T where {T<:AbstractFloat}
    round(x - z * y, digits=5)
end

# Set the parameters
function init_parameters(;
    ħω₀=4 / 5, λ=0.0, U=0.0, ϵσ=0.0,
    ρ=[0.6, 0.6], T=[3.0, 3.0],
    eV=0.0, Λ₀=0.0, Γ₀=1.0,
    Bz=[g * μB, -g * μB]
)::parameters
    U = renormalization(U, λ^2 / ħω₀, 2.0)
    ϵσ = renormalization(ϵσ, λ^2 / ħω₀, 1.0)

    auxϵσ = zeros(Float64, 2, 2, 2)
    γ = zeros(Float64, 2, 2, 2)

    Λl = zeros(Float64, 2, 2, 2)
    Λr = zeros(Float64, 2, 2, 2)

    Γl = zeros(Float64, 2, 2, 2)
    Γr = zeros(Float64, 2, 2, 2)

    auxϵσ[1, 1, :] .= [ϵσ + 8.5*Bz[1], ϵσ - 8.5*Bz[1]]
    auxϵσ[2, 2, :] .= [ϵσ + 8.5*Bz[2], ϵσ - 8.5*Bz[2]]

    γ[1, 2, :] .= 1.0
    γ[2, 1, :] .= 1.0

    Λl[1, 1, 1] = 1 + ρ[1]
    Λl[1, 1, 2] = 1 - ρ[1]

    Λr[2, 2, 1] = 1 + ρ[2]
    Λr[2, 2, 2] = 1 - ρ[2]

    Λ = Λ₀ * (Λl + Λr)

    Γl[1, 1, 1] = 1 + ρ[1]
    Γl[1, 1, 2] = 1 - ρ[1]

    Γr[2, 2, 1] = 1 + ρ[2]
    Γr[2, 2, 2] = 1 - ρ[2]

    Γ = Γ₀ * (Γl + Γr)

    Σ = Λ - 0.5im * Γ

    μ = [1.0 1.0] .+ 0.5 * eV * [1.0 -1.0]
    μl = [μ[1] μ[1]]
    μr = [μ[2] μ[2]]
    parameters(ħω₀, λ, U, auxϵσ, ρ, Λ, Λl, Λr, Γ, Γl, Γr, Σ, T, μ, μl, μr, γ)
end

# Kernel function for Green's function calculation
function kernel(ϵ::Float64, aux::Array{Complex{Float64}})::Array{Complex{Float64},3}
    ans = copy(aux)
    ans[1, 1, :] .+= ϵ
    ans[2, 2, :] .+= ϵ
    det = @. (ans[1, 1, :] * ans[2, 2, :]) - (ans[1, 2, :] * ans[2, 1, :])
    ans[:, :, 1] /= det[1]
    ans[:, :, 2] /= det[2]
    ans
end

function trace(aux)
    ans = 0.0
    for i ∈ 1:size(aux)[1]
        ans += aux[i, i]
    end
    ans
end

# Function to compute Green's functions
function main_computation(E, nE::Int, p::parameters,
    Aω, Bω, G1less, G1great, G0less, G0great, n_result
)::Nothing
    sizenE = (2, 2, nE, 2)
    Gret = zeros(Complex{Float64}, sizenE)
    Gadv = zeros(Complex{Float64}, sizenE)

    Gless = zeros(Complex{Float64}, sizenE)
    Ggreat = zeros(Complex{Float64}, sizenE)

    β = 1 / (kb * sum(p.T))
    pl = fermi_dirac_d(E, p.μl, kb, p.T[1])
    pr = fermi_dirac_d(E, p.μr, kb, p.T[2])

    nB = bose_einstein_d(p.ħω₀, β)

    aux1 = (p.λ / p.ħω₀)^2
    amplitude = exp(-aux1 * (1 + 2 * nB))
    z = 2 * aux1 * (sqrt(nB * (nB + 1)))

    n = 20
    aux2 = [exp(-0.5 * i * p.ħω₀ * β) * besseli(abs(i), z) for i ∈ -n:n]
    aux3 = reshape(E, 1, nE) .- p.ħω₀ * range(-n, n)

    # Partial matrix inversion
    aux4 = @. -(p.ϵσ + p.γ + p.Σ)
    aux4[1, 2, :] .*= -1
    aux4[2, 1, :] .*= -1
    auxcopy = aux4[1, 1, :]
    aux4[1, 1, :] .= aux4[2, 2, :]
    aux4[2, 2, :] .= auxcopy

    for i ∈ 1:nE
        for l ∈ 1:(2n+1)
            Gret[:, :, i, :] += aux2[l] .* kernel(aux3[l, i], aux4)
        end
        #Equilibrium
        Gret[:, :, i, :] .*= amplitude
        Gadv[:, :, i, :] .= conj(Gret[:, :, i, :])
        Aω[i, 1] = (-1 / π) * imag(trace(Gret[:, :, i, 1]))
        Aω[i, 2] = (-1 / π) * imag(trace(Gret[:, :, i, 2]))

        #Nonequilibrium
        aux5 = @. Gret[:, :, i, :] * p.Γl * Gadv[:, :, i, :]
        aux6 = @. Gret[:, :, i, :] * p.Γr * Gadv[:, :, i, :]
        Gless[:, :, i, 1] = @. 1im * ((pl[i, 1] * aux5[:, :, 1]) + (pr[i, 1] * aux6[:, :, 1]))
        Gless[:, :, i, 2] = @. 1im * ((pl[i, 2] * aux5[:, :, 2]) + (pr[i, 1] * aux6[:, :, 2]))
        Ggreat[:, :, i, 1] = @. -1im * (((1 - pl[i, 1]) * aux5[:, :, 1]) + ((1 - pr[i, 1]) * aux6[:, :, 1]))
        Ggreat[:, :, i, 2] = @. -1im * (((1 - pl[i, 2]) * aux5[:, :, 2]) + ((1 - pr[i, 2]) * aux6[:, :, 2]))

        Bω[i, 1] = (-1 / π) * imag(trace(Ggreat[:, :, i, 1]) - trace(Gless[:, :, i, 1]))
        Bω[i, 2] = (-1 / π) * imag(trace(Ggreat[:, :, i, 2]) - trace(Gless[:, :, i, 2]))
    end
    #Spectral function A(ω)
    Anor = trapezoidal_1d(E, Aω[:, 1] .+ Aω[:, 2])[1]
    Gret ./= Anor
    Gadv ./= Anor
    Aω ./= Anor

    #Nonequilibrium Spectral function B(ω)
    Bnor = trapezoidal_1d(E, Bω[:, 1] .+ Bω[:, 2])
    Gless ./= Bnor
    Ggreat ./= Bnor
    Bω ./= Bnor

    for i ∈ 1:2, j ∈ 1:2
        n_result[i, j, :] .= (1 / 2π) .* trapezoidal_1d(E, imag(Gless[i, j, :, :]))
    end

    G1less .= 0.5 .* (Gless[:, :, :, 1] .- Gless[:, :, :, 2])
    G1great .= 0.5 .* (Ggreat[:, :, :, 1] .- Ggreat[:, :, :, 2])
    G0less .= 0.5 .* (Gless[:, :, :, 1] .+ Gless[:, :, :, 2])
    G0great .= 0.5 .* (Ggreat[:, :, :, 1] .+ Ggreat[:, :, :, 2])
    return nothing
end

end