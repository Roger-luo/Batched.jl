batched_mul(A::AbstractVector{T}, B::AbstractArray{T, 3}) where T = batched_mul!(similar(B), A, B)

function batched_mul!(C::AbstractArray{T, 3}, A::AbstractVector{T}, B::AbstractArray{T, 3}) where T
    @boundscheck (size(C, 3) == length(A) == size(B, 3) || error("Batch size mismatch"))

    @inbounds for k in 1:size(C, 3)
        for j in 1:size(C, 2)
            for i in 1:size(C, 1)
                C[i, j, k] = A[k] * B[i, j, k]
            end
        end
    end
    C
end


batched_tr(A::AbstractArray) = batched_tr!(zero(A), A)

"""
    batched_tr!(B::AbstractVector{T}, A::AbstractArray{T, 3})

Perform batched matrix trace and add the trace to `B`.
"""
function batched_tr!(B::AbstractVector{T}, A::AbstractArray{T, 3}) where T
    @assert size(A, 1) == size(A, 2) "Expect a square matrix" # checksquare
    @boundscheck size(A, 3) == size(B, 1) || error("Batch size mismatch")

    nbatch = size(A, 3)
    n = size(A, 1)
    @inbounds for k in 1:nbatch
        for i in 1:n
            B[k] += A[i, i, k]
        end
    end
    B
end
