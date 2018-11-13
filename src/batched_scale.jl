struct BatchedScaleMatrix{K, T, AT <: AbstractVector{T}} <: AbstractArray{T, 3}
    scalars::AT
end

BatchedScaleMatrix{K}(scalars::AT) where {K, T, AT <: AbstractVector{T}} = BatchedScaleMatrix{K, T, AT}(scalars)
Base.size(A::BatchedScaleMatrix{K}) where K = (K, K, length(A.scalars))
Base.eltype(A::BatchedScaleMatrix{K, T}) where {K, T} = T

function Base.getindex(A::BatchedScaleMatrix{K, T}, i, j, k) where {K, T}
    i == j ? A.scalars[k] : zero(T)
end

function batched_mul!(Y::AbstractArray{T, 3}, A::BatchedScaleMatrix{K, T}, B::AbstractArray{T, 3}) where {K, T}
    @boundscheck size(A, 3) == size(B, 3) == size(Y, 3) || error("Batch size mismatch")

    @inbounds for k in size(B, 3), j in size(B, 2), i in size(B, 1)
        Y[i, j, k] = A.scalars[k] * B[i, j, k]
    end
    Y
end

batched_mul!(Y::AbstractArray{T, 3}, A::AbstractArray{T, 3}, B::BatchedScaleMatrix{K, T}) where {K, T} =
    batched_mul!(Y, B, A)

function batched_mul!(Y::AbstractArray{T, 3}, A::BatchedScaleMatrix{K, T}, B::BatchedTransposeOrAdjoint{T, 3, <:AbstractArray{T, 3}}) where {T, K}
    @boundscheck size(A, 3) == size(B, 3) == size(Y, 3) || error("Batch size mismatch")

    # NOTE: we exchange the order of i,j since this is the contiguous order in memory
    @inbounds for k in size(B, 3), i in size(B, 1), j in size(B, 2)
        Y[i, j, k] = A.scalars[k] * B[i, j, k]
    end
    Y
end

batched_mul!(Y::AbstractArray{T, 3}, A::BatchedTransposeOrAdjoint{T, 3, <:AbstractArray{T, 3}}, B::BatchedScaleMatrix{K, T}) where {T, K} =
    batched_mul!(Y, B, A)
