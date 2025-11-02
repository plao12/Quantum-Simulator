"""
# InteractionCaller Module : Call interaction functions based on device and level
"""
module InteractionCaller

export init_interactions

# Try to load CUDA and check functionality
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch err
    @warn "CUDA not available: $err"
    false
end

module CPUWrapperCommon
include(joinpath("cpu", "uniaxial_anisotropy", "uniaxial_anisotropy.jl"))
include(joinpath("cpu", "heisenberg", "heisenberg.jl"))
include(joinpath("cpu", "ising", "ising.jl"))

using .UniaxialAnisotropy
using .Heisenberg
using .Ising

end

# Now conditionally define GPUWrapperCommon using `eval` and a quoted expression
if HAS_CUDA
    @eval module GPUWrapperCommon
    #using CUDA

    include(joinpath("gpu", "uniaxial_anisotropy", "uniaxial_anisotropy.jl"))
    include(joinpath("gpu", "heisenberg", "heisenberg.jl"))
    include(joinpath("gpu", "ising", "ising.jl"))

    using .UniaxialAnisotropy
    using .Heisenberg
    using .Ising
    end  # end GPUWrapperCommon
else
    @info "CUDA not functional or not available. Skipping GPUWrapperCommon definition."
    const GPUWrapperCommon = nothing
end

# Make CPUWrapperCommon available
using .CPUWrapperCommon

# Conditionally use GPUWrapperCommon only if it was defined
if HAS_CUDA
    using .GPUWrapperCommon
end

function init_interactions(device::String, level::Int)::NamedTuple
    if device == "cpu"
        if level == 1
            return (anisotropy=CPUWrapperCommon.UniaxialAnisotropy.anisotropy!,)
        else
            return (
                ising=CPUWrapperCommon.Ising.ising!,
                heisenberg=CPUWrapperCommon.Heisenberg.heisenberg!
            )
        end
    elseif device == "gpu"
        if level == 1
            return (anisotropy=GPUWrapperCommon.UniaxialAnisotropy.anisotropy!,)
        else
            return (
                ising=GPUWrapperCommon.Ising.ising!,
                heisenberg=GPUWrapperCommon.Heisenberg.heisenberg!
            )
        end
    end
end

end # end InteractionCaller