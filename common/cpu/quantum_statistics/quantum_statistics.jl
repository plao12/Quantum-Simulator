module QuantumStatistics

export bose_einstein_d, fermi_dirac_d

# Bose-Einstein distribution function
"""
    bose_einstein_d(ε::T, β::T)::T where {T<:Real}

    This function computes the Bose-Einstein distribution function.

    ε: Energy (in units of k_B * T)
    β: Inverse temperature (1 / (k_B * T))

    Returns the value of the Bose-Einstein distribution at the given energy and temperature.
"""
function bose_einstein_d(ε::T, β::T)::T where {T<:Float64}
    exp_val = exp(ε * β)
    if abs(exp_val - 1.0) < 1e-10  # Handle small values of ε * β
        return 1 / (ε * β)  # Approximation for small ε * β
    else
        return 1 / (exp_val - 1)
    end
end

# Fermi-Dirac distribution function
"""
    fermi_dirac_d(ε, μ::Array{T}, kb::T, Te::T)::Array{T} where {T<:Real}

    This function computes the Fermi-Dirac distribution function.

    ε: Energy array (in units of eV or appropriate units)
    μ: Chemical potential array (up and down) (in the same units as ε) 
    kb: Boltzmann constant
    Te: Lead temperature 

    Returns an array of Fermi-Dirac distribution values for each energy.
"""
function fermi_dirac_d(ε, μ::Array{T}, kb::T, Te::T)::Array{T} where {T<:Float64}
    β = 1 / (kb * Te)  # Inverse temperature
    x = @. (ε - μ) * β
    return @. 1 / (exp(x) + 1)  # Broadcasting the operation
end

end
