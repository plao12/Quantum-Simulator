"""
Module for the Ising interaction computation.
"""
module Ising

include(joinpath("..", "integration_methods", "integration.jl"))
using .Integration

export ising!

function allowed_combinations(dim::Int)::Array{NTuple{2,Int},1}
    ans = []
    for i ∈ 1:dim, j ∈ i:dim
        push!(ans, (i, j))
    end
    return ans
end

function ising!(
    E1, E2,
    nE1::Int, nE2::Int,
    G1less1, G1great1,
    G1less2, G1great2,
    ans
)::Nothing
    dim = size(G1less1, 1)
    combinations = allowed_combinations(dim)
    aux = zeros(Float64, nE1, nE2)
    #println("Ising combinations: ", combinations)

    for k in eachindex(combinations)
        m, n = combinations[k]
        @inbounds for i in 1:nE1, j in 1:nE2
            aux1 = real(G1great1[m, n, i] * G1less2[n, m, j] - G1less1[m, n, i] * G1great2[n, m, j])
            aux2 = real(G1great2[m, n, i] * G1less1[n, m, j] - G1less2[m, n, i] * G1great1[n, m, j])
            aux3 = (aux1 - aux2) / (E1[i] - E2[j] + 1e-10)
            if isnan(aux3) || isinf(aux3)
                aux[i, j] = 0.0
            else
                aux[i, j] = aux3
            end
        end
        ans[k] = 0.5 * trapezoidal_2d(E1, E2, aux) / (2π)^2
    end
    return nothing
end

end  # module Ising
