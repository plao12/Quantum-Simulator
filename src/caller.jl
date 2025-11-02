"""
# GreensCaller Module : Call green's functions based on device, level and U interaction
"""
module GreensCaller

export init_greens_functions

# Try to load CUDA and check functionality
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch err
    @warn "CUDA not available: $err"
    false
end

module CPUWrapper

include(joinpath("cpu", "single_level", "green_function", "green_functions.jl"))
include(joinpath("cpu", "single_level_no_U", "green_function", "green_functions.jl"))
include(joinpath("cpu", "two_levels_no_U", "green_function", "green_functions.jl"))
include(joinpath("cpu", "three_levels_no_U", "green_function", "green_functions.jl"))

using .SingleLevelCPU
using .SingleLevelNoUCPU
using .TwoLevelsCPUNoU
using .ThreeLevelsCPUNoU

end  # end CPUWrapper

# Now conditionally define GPUWrapper using `eval` and a quoted expression
if HAS_CUDA
    @eval module GPUWrapper
    #using CUDA

    include(joinpath("gpu", "single_level_no_U", "green_function", "green_functions.jl"))

    using .SingleLevelNoUGPU
    end  # end GPUWrapper
else
    @info "CUDA not functional or not available. Skipping GPUWrapper definition."
    const GPUWrapper = nothing
end

# Make CPUWrapper available
using .CPUWrapper

# Conditionally use GPUWrapper only if it was defined
if HAS_CUDA
    using .GPUWrapper
end

function init_greens_functions(device::String, level::Int, U_interaction::Bool)::NamedTuple
    if device == "cpu"
        if level == 1
            if U_interaction
                return (
                    init_p=CPUWrapper.SingleLevelCPU.init_parameters,
                    main_c=CPUWrapper.SingleLevelCPU.main_computation
                )
            else
                return (
                    init_p=CPUWrapper.SingleLevelNoUCPU.init_parameters,
                    main_c=CPUWrapper.SingleLevelNoUCPU.main_computation
                )
            end
        elseif level == 2
            if U_interaction
            else
                return (
                    init_p=CPUWrapper.TwoLevelsCPUNoU.init_parameters,
                    main_c=CPUWrapper.TwoLevelsCPUNoU.main_computation
                )
            end
        elseif level == 3
            if U_interaction
            else
                return (
                    init_p=CPUWrapper.ThreeLevelsCPUNoU.init_parameters,
                    main_c=CPUWrapper.ThreeLevelsCPUNoU.main_computation
                )
            end
        end
    elseif device == "gpu"
        if level == 1
            if U_interaction
                return (
                    init_p=GPUWrapper.SingleLevelGPU.init_parameters,
                    main_c=GPUWrapper.SingleLevelGPU.main_computation
                )
            else
                return (
                    init_p=GPUWrapper.SingleLevelNoUGPU.init_parameters,
                    main_c=GPUWrapper.SingleLevelNoUGPU.main_computation
                )
            end
        end
    end
end

end # end GreensCaller