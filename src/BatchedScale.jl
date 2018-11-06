"""
    BatchedScale{K, B, T, N, VT}

Scale a batch of arrays with a batch of scalars. The shape of batch is `BatchedScale.batchs`
of `K`. `N` = `K` + `B`.
"""
struct BatchedScale{K, T, N, ST <: AbstractArray{T}} <: AbstractBatchedMatrix{T, N}
    scalars::ST
    dims::Dims{K}

    BatchedScale(scalars::ST, dims::Dims{K}) where {K, T, ST <: AbstractArray{T}} =
        new{K, T, K + ndims(scalars), ST}(scalars, dims)
end

inner_size(x::BatchedScale) = x.dims
batch_size(x::BatchedScale) = size(x.scalars)
Base.size(x::BatchedScale{K, T, N}) where {K, T, N} = (inner_size(x)..., batch_size(x)...)

Base.getindex(m::BatchedScale, I...) = getindex(m, I)

function Base.@propagate_inbounds Base.getindex(m::BatchedScale{K, T, N}, I::NTuple{N, Int}) where {K, T, N}
    reduce(==, I[1:K]) ? m.scalars[I[K+1:N]...] : zero(T)
end

Base.IndexStyle(::Type{<:BatchedScale}) = IndexCartesian()
Base.transpose(A::BatchedScale) = A
