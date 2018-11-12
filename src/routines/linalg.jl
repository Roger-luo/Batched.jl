export batched_tr

batched_tr(A::AbstractArray{T, 3}) where T = batched_tr!(A, fill!(similar(A, (size(A, 3), )), 0))

function batched_tr!(A::AbstractArray{T, 3}, B::AbstractVector{T}) where T
    @boundscheck size(A, 3) == length(B) || error("Batch size mismatch")
    @inbounds for k in size(A, 3), i in size(A, 1)
        B[k] += A[i, i, k]
    end
    B
end
