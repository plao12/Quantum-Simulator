module DzyaloshinskiiMoriya

include(joinpath("..", "integration_methods", "integration.jl"))
using .Integration

export dzyaloshinskii_moriya!

function allowed_combinations(dim::Int)::Array{NTuple{2, Int}, 1}
    ans = []
    for i ∈ 1:dim, j ∈ i:dim
        j != i ? push!(ans, (i, j)) : nothing
    end
    return ans
end

function dzyaloshinskii_moriya!(
    E1,
    nE::Int, 
    G1less, G1great,
    G0less, G0great,
    ans
)::Nothing
    dim = size(G1less1, 1)
    combinations = allowed_combinations(dim)
    aux = zeros(Float64, nE)
    for k in eachindex(combinations)
        m, n = combinations[k]
        # Perform the main computation
        @inbounds for i in 1:nE
            aux1 = real(G0less1)
            aux2 = 0.0
            aux3 = (aux1 - aux2)
            # Set result to 0 if invalid
            if isnan(aux3) || isinf(aux3)
                aux[i] = 0.0
            else
                aux[i] = aux3
            end
        end
        ans[k] = 0.5 * trapezoidal_2d(E1, E2, aux) / (2π)^2
    end
end
    
end