"""
Module for numerical integration methods using CUDA for GPU acceleration.
"""
module Integration

using CUDA

export trapezoidal2d!, trapezoidal1d!

"""
    trapezoidal2d!(ans, x, y, z, nx, ny)

Computes the 2D trapezoidal integral of `z` over a grid defined by `x` and `y` using CUDA.
This is a GPU kernel function designed to be launched with `CUDA.@cuda`.

# Arguments
- `ans`: A pointer or single-element array (e.g., `CuArray{Float64}(undef, 1)`) to store the accumulated integral result.
- `x`: A `CuDeviceArray` representing the x-coordinates of the grid.
- `y`: A `CuDeviceArray` representing the y-coordinates of the grid.
- `z`: A `CuDeviceArray` (2D) representing the function values on the grid, where `z[i, j]` corresponds to `f(x[i], y[j])`.
- `nx`: The number of points in the x-dimension.
- `ny`: The number of points in the y-dimension.

# Returns
`nothing`. The result is accumulated into `ans` using atomic operations.
"""
function trapezoidal2d!(ans, x, y, z, nx, ny)
    # Allocate static shared memory for intermediate results within a thread block.
    # The size is determined by the block dimensions at kernel launch.
    sharedaux = CUDA.CuDynamicSharedArray(Float64, (blockDim().x, blockDim().y))

    # Calculate global indices for the current thread within the 2D grid.
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    aux = 0.0
    # Check if the current thread's calculated indices are within the valid data range
    # (i.e., within the bounds of the actual grid elements for integration).
    if i <= nx - 1 && j <= ny - 1
        # Apply the 2D trapezoidal rule formula for the current rectangular element.
        # This calculates the contribution of one grid cell to the integral.
        aux = (x[i+1] - x[i]) * (y[j+1] - y[j]) * (z[i, j] + z[i+1, j] + z[i, j+1] + z[i+1, j+1])
    end
    # Store the calculated local auxiliary value in shared memory at the thread's position.
    sharedaux[threadIdx().x, threadIdx().y] = aux
    # Synchronize all threads within the block. This ensures that all threads have
    # completed their writes to `sharedaux` before any thread attempts to read from it.
    sync_threads()

    # The first thread (threadIdx().x == 1 and threadIdx().y == 1) in the block
    # is responsible for aggregating the results from shared memory for this block.
    if threadIdx().x == 1 && threadIdx().y == 1
        total = 0.0
        # Sum all auxiliary values from shared memory to get the block's total contribution.
        for tx in 1:blockDim().x, ty in 1:blockDim().y
            total += sharedaux[tx, ty]
        end
        # Atomically add the block's total contribution (scaled by 1/4) to the global answer.
        # `ans` is typically a `CuArray` of length 1, and `pointer(ans)` gets its device address.
        CUDA.atomic_add!(pointer(ans), total / 4)
    end
    return nothing
end

"""
    trapezoidal1d!(ans, x, y, n)

Computes the 1D trapezoidal integral of `y` over a range defined by `x` using CUDA.
This is a GPU kernel function designed to be launched with `CUDA.@cuda`.

# Arguments
- `ans`: A pointer or single-element array (e.g., `CuArray{Float64}(undef, 1)`) to store the accumulated integral result.
- `x`: A `CuDeviceArray` representing the x-coordinates of the intervals.
- `y`: A `CuDeviceArray` representing the function values.
- `n`: The number of points in the arrays `x` and `y`.

# Returns
`nothing`. The result is accumulated into `ans` using atomic operations.
"""
function trapezoidal1d!(ans, x, y, n)
    # Allocate dynamic shared memory for intermediate results within a thread block.
    # The size is determined at kernel launch via the `shmem` argument.
    sharedaux = CuDynamicSharedArray(Float64, blockDim().x)

    # Calculate the global index for the current thread within the 1D array.
    i = threadIdx().x + blockDim().x * (blockIdx().x - 1)

    aux = 0.0
    # Check if the current thread's calculated index is within the valid data range
    # (i.e., within the bounds of the actual array elements for integration).
    if i < n
        # Apply the 1D trapezoidal rule formula for the current segment.
        aux = (x[i+1] - x[i]) * (y[i] + y[i+1])
    end
    # Store the calculated local auxiliary value in shared memory at the thread's position.
    sharedaux[threadIdx().x] = aux
    # Synchronize all threads within the block. This ensures that all threads have
    # completed their writes to `sharedaux` before any thread attempts to read from it.
    sync_threads()

    # The first thread (threadIdx().x == 1) in the block is responsible for
    # aggregating the results from shared memory for this block.
    if threadIdx().x == 1
        total = 0.0
        # Sum all auxiliary values from shared memory to get the block's total contribution.
        for tx in 1:blockDim().x
            total += sharedaux[tx]
        end
        # Atomically add the block's total contribution (scaled by 1/2) to the global answer.
        # `ans` is typically a `CuArray` of length 1, and `pointer(ans)` gets its device address.
        CUDA.atomic_add!(pointer(ans), total / 2)
    end
    return nothing
end

end # module Integration