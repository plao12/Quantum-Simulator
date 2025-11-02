module QuantumStatistics

using CUDA

export bose_einstein_d, fermi_dirac_d!

# Bose-Einstein distribution function
function bose_einstein_d(ϵ, β)::Float64
    exp_val = exp(ϵ * β)
    if exp_val ≈ 1.0
        1 / (ϵ * β)  # Handle small values of ϵ * β
    else
        1 / (exp_val - 1)
    end
end

# Fermi-Dirac distribution function
function fermi_dirac_d!(ans, kb, ϵ, n, μ, T)
    i = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    if i <= n
        β = 1 / (kb * T)
        for j ∈ 1:2
            ans[i, j] = 1 / (exp((ϵ[i] - μ[j]) * β) + 1)
        end
    end
    return nothing
end

end
