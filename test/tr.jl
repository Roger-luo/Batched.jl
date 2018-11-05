using Batched, LinearAlgebra, BenchmarkTools, Test

function test_trace!(out::AbstractVector{T}, A::AbstractArray{T, 3}) where T
    @inbounds for k in 1:size(A, 3)
        out[k] = LinearAlgebra.tr(selectdim(A, 3, k))
    end
    out
end

out = zeros(100)
A = rand(10, 10, 100)


@benchmark test_trace!(out, A)
@benchmark Batched.trace!(out, A)
