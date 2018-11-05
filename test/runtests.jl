using Batched, BenchmarkTools

A = rand(10, 10, 100)
C = zeros(10, 10, 100)
@benchmark Batched.gemm!('N', 'N', 1.0, A, A, 1.0, C)

using LinearAlgebra

@which tr(rand(10, 10))

function test_tr(A::Matrix{T}) where T
    n = BLAS.checksquare(A)
    t = zero(T)
    @inbounds for i=1:n
        t += A[i,i]
    end
    t
end


X = rand(10000, 10000)
@benchmark tr(X)



@benchmark test_tr(X)
