export BatchedUniformScaling

"""
    BatchedUniformScaling{T, N, ST <: AbstractArray{T, N}} <: AbstractBatchedArray{T, 0, N}

Scale a batch of arrays with a batch of scalars.

    BatchedUniformScaling(scalars)

The shape of batch can be multidimentional, which means member `BatchedScale.scalars`
can be a matrix or high dimentional array, the shape of this member is the shape of batch.
`dims` defines the dimmension of each sample in the batch. It can be multidimentional
as well.
"""
struct BatchedUniformScaling{T, N, ST <: AbstractArray{T, N}} <: AbstractBatchedArray{T, 0, N, ST}
    scalars::ST
end

inner_size(x::BatchedUniformScaling) = ()
batch_size(x::BatchedUniformScaling) = size(x.scalars)
Base.size(x::BatchedUniformScaling) = (inner_size(x)..., batch_size(x)...)

Base.getindex(m::BatchedUniformScaling, I...) = getindex(m.scalars, I...)

Base.IndexStyle(::Type{<:BatchedUniformScaling}) = IndexCartesian()
Base.transpose(A::BatchedUniformScaling) = A
Base.adjoint(A::BatchedUniformScaling{<:Real}) = A
Base.adjoint(A::BatchedUniformScaling{<:Complex}) = BatchedUniformScaling(conj.(A.scalars))

merge_batch_dim(x::BatchedUniformScaling) = vec(x.scalars)


struct BatchedMatrixScale{K, T, N, ST <: AbstractArray{T, N}} <: AbstractBatchedMatrix{T, N, ST}
    scalars::ST

    BatchedMatrixScale{K}(scalars::ST) where {K, T, N, ST <: AbstractArray{T, N}} = new{K, T, N, ST}(scalars)
end

inner_size(::BatchedMatrixScale{K}) where K = (K, K)
batch_size(x::BatchedMatrixScale) = size(x.scalars)
Base.size(x::BatchedMatrixScale) = (inner_size(x)..., batch_size(x)...)

# TODO: boundscheck?
Base.getindex(x::BatchedMatrixScale{K, T}, I...) where {K, T} = I[1] == I[2] ? x.scalars[I[3:end]...] : zero(T)
Base.IndexStyle(::Type{<:BatchedMatrixScale}) = IndexCartesian()

Base.transpose(A::BatchedMatrixScale) = A
Base.adjoint(A::BatchedMatrixScale{<:Real}) = A
Base.adjoint(A::BatchedMatrixScale{<:Complex}) = BatchedMatrixScale(conj.(A.scalars))

merge_batch_dim(x::BatchedMatrixScale) = vec(x.scalars)
