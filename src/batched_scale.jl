export BatchedScaleMatrix

struct BatchedScaleMatrix{K, T, AT <: AbstractVector{T}} <: AbstractArray{T, 3}
    scalars::AT
end

BatchedScaleMatrix{K}(scalars::AT) where {K, T, AT <: AbstractVector{T}} = BatchedScaleMatrix{K, T, AT}(scalars)
Base.size(A::BatchedScaleMatrix{K}) where K = (K, K, length(A.scalars))
Base.eltype(A::BatchedScaleMatrix{K, T}) where {K, T} = T

function Base.getindex(A::BatchedScaleMatrix{K, T}, i, j, k) where {K, T}
    i == j ? A.scalars[k] : zero(T)
end

batched_lmul!(A::BatchedScaleMatrix{K, T}, B::AbstractArray{T, 3}) where {K, T} = batched_scal!(A.scalars, B)
batched_rmul!(B::AbstractArray{T, 3}, A::BatchedScaleMatrix{K, T}) where {K, T} = batched_scal!(A.scalars, B)

batched_mul(A::BatchedScaleMatrix{K, T}, B::AbstractArray{T, 3}) where {K, T} = batched_lmul!(A, copy(B))
batched_mul(A::BatchedScaleMatrix, B::BatchedTranspose) = batched_transpose(batched_mul(A, B.parent))
batched_mul(A::BatchedScaleMatrix, B::BatchedAdjoint) = batched_adjoint(batched_mul(A, B.parent))

batched_mul(B::AbstractArray{T, 3}, A::BatchedScaleMatrix{K, T}) where {K, T} = batched_rmul!(copy(B), A)
