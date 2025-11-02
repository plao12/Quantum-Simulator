module UniaxialAnisotropy

include(joinpath("..", "integration_methods", "integration.jl"))
using .Integration
using CUDA

export anisotropy!

function evaluation(ans, E1, E2, nE1, nE2, G1less1, G1great1, G1less2, G1great2)::Nothing
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= nE1 && j <= nE2
        aux1 = real(
            (G1less1[i] * G1great2[j]) - (G1great1[i] * G1less2[j])
        ) / (E1[i] - E2[j] + 1e-10)

        if isnan(aux1) || isinf(aux1)
            ans[i, j] = 0.0
        else
            ans[i, j] = aux1
        end
    end
    return nothing
end

function anisotropy!(
    E1, E2,
    nE1::Int, nE2::Int,
    G1less1, G1great1,
    G1less2, G1great2,
    ans
)::Nothing
    threads = (16, 16)
    blocks = (Int(ceil(nE1 / threads[1])), Int(ceil(nE2 / threads[2])))
    shared_memory_size = threads[1] * threads[2] * sizeof(Float64)

    aux_gpu = CUDA.zeros(Float64, nE1, nE2)
    ans_aux = CUDA.zeros(Float64, 1)

    G1less1_gpu = CUDA.CuArray(G1less1) 
    G1great1_gpu = CUDA.CuArray(G1great1)
    G1less2_gpu = CUDA.CuArray(G1less2)
    G1great2_gpu = CUDA.CuArray(G1great2)

    @cuda threads=threads blocks=blocks evaluation(aux_gpu, E1, E2, nE1, nE2, G1less1_gpu, G1great1_gpu, G1less2_gpu, G1great2_gpu)
    @cuda threads=threads blocks=blocks shmem=shared_memory_size trapezoidal2d!(ans_aux, E1, E2, aux_gpu, nE1, nE2)
    ans[1] = Array(ans_aux)[1] / (2π)^2
    return nothing
end

end  # module UniaxialAnisotropy