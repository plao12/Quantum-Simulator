module UniaxialAnisotropy

include(joinpath("..", "integration_methods", "integration.jl"))
using .Integration

export anisotropy!

function anisotropy!(
  E1, E2,
  nE1::Int, nE2::Int,
  G1less1, G1great1,
  G1less2, G1great2,
  ans
)::Nothing

  aux = zeros(Float64, nE1, nE2)

  @inbounds for i in 1:nE1, j in 1:nE2
    aux1 = real(
      (G1less1[j] * G1great2[i]) - (G1great1[j] * G1less2[i])
    ) / (E1[j] - E2[i] + 1e-10)

    if isnan(aux1) || isinf(aux1)
      aux[j, i] = 0.0
    else
      aux[j, i] = aux1
    end
  end
  ans[1] = trapezoidal_2d(E1, E2, aux) / (2π)^2
  return nothing
end

UniaxialAnisotropy

end  # module UniaxialAnisotropy
