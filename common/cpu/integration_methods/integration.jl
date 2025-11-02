module Integration

export trapezoidal_1d, trapezoidal_2d

function trapezoidal_1d(x, z::Array{T})::Vector{T} where {T<:Float64}
    num_x = length(x)
    dimention = ndims(z)
    integral = zeros(T, dimention)
    if dimention > 1
        for j in 1:dimention
            for i in 1:num_x-1
                integral[j] += (x[i+1] - x[i]) * (z[i, j] + z[i+1, j])
            end
        end
    else
        for i in 1:num_x-1
            integral .+= (x[i+1] - x[i]) * (z[i] + z[i+1])
        end
    end
    return integral ./ T(2)
end

function trapezoidal_2d(x, y, z::Matrix{T})::T where {T<:Float64}
    num_x, num_y = size(z)
    integral = zero(T)
    for i in 1:num_x-1
        for j in 1:num_y-1
            delta_x = x[i+1] - x[i]
            delta_y = y[j+1] - y[j]
            sum_z_corners = z[i, j] + z[i+1, j] + z[i, j+1] + z[i+1, j+1]
            integral += delta_x * delta_y * sum_z_corners
        end
    end
    return integral / T(4)
end

end # module Integration