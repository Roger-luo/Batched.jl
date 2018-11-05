using Batched, LinearAlgebra, BenchmarkTools

A = rand(10, 10, 100)
C1 = zeros(10, 10, 100)
Batched.gemm!('N', 'N', 1.0, A, A, 1.0, C1)

function test_gemm!(tA::AbstractChar, tB::AbstractChar, alpha::T, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}, beta::T, C::AbstractArray{T, 3}) where T
    @assert size(A, 3) == size(B, 3) == size(C, 3) "Batch size mismatch"
    chunk_size = size(A, 1) * size(A, 2)
    for k in 1:size(A, 3)
        LinearAlgebra.BLAS.gemm!(tA, tB, alpha, selectdim(A, 3, k), selectdim(B, 3, k), beta, selectdim(C, 3, k))
    end
    C
end

C2 = zeros(10, 10, 100)
test_gemm!('N', 'N', 1.0, A, A, 1.0, C2)
