export BatchedArray, BatchedMatrix, BatchedVector

import LinearAlgebra
import LinearAlgebra: BLAS

"""
    AbstractBatchedArray{T, NI, N}

Abstract type batched array. A batched array use its last `N - NI` dimension as
batch dimension, it is a batch of array with dimension `NI`.
"""
abstract type AbstractBatchedArray{T, NI, N} <: AbstractArray{T, N} end


"""
    AbstractBatchedVector{T, N}

Batched vector.
"""
const AbstractBatchedVector{T, N} = AbstractBatchedArray{T, 1, N}

"""
    AbstractBatchedMatrix{T, N}

Batched matrix.
"""
const AbstractBatchedMatrix{T, N} = AbstractBatchedArray{T, 2, N}

"""
    inner_size(batched_array) -> Tuple

Returns a tuple of size of each inner dimension of the batched array.
"""
function inner_size end

"""
    batch_size(batched_array) -> Tuple

Returns a tuple of size of each batch dimension of the batched array.
"""
function batch_size end

"""
    merged_size(batched_array) -> Tuple

Returns the size of this batched array after merging all its batched dimension together.
"""
function merged_size end

Base.:(*)(lhs::AbstractBatchedMatrix, rhs::AbstractBatchedMatrix) = batched_gemm(lhs, rhs)


"""
    BatchedArray{T, NI, N, AT} <: AbstractBatchedArray{T, NI, N}

A concrete type for batched arrays. `T` is the element type, `NI` is the inner sample's
dimension, `N` is the total dimension and `AT` is the array type that actually holds the
value.
"""
struct BatchedArray{T, NI, N, AT <: AbstractArray{T, N}} <: AbstractBatchedArray{T, NI, N}
    parent::AT
end

BatchedArray(NI::Int, data::AT) where {T, N, AT <: AbstractArray{T, N}} = BatchedArray{T, NI, N, AT}(data)

Base.size(x::BatchedArray) = size(x.parent)
Base.strides(x::BatchedArray) = strides(x.parent)
Base.getindex(x::BatchedArray, I...) = getindex(x.parent, I...)
Base.setindex!(x::BatchedArray, v, I...) = setindex!(x.parent, v, I...)
Base.IndexStyle(x::BatchedArray) = IndexStyle(x.parent)

inner_size(x::BatchedArray{T, NI, N}) where {T, NI, N} = Tuple(size(x, i) for i in Base.OneTo(NI))
batch_size(x::BatchedArray{T, NI, N}) where {T, NI, N} = Tuple(size(x, i) for i in (NI+1):N)
batch_size(x::BatchedArray, i::Int) = batch_size(x, i)
merged_size(x::BatchedArray) = (inner_size(x)..., prod(batch_size(x)))

merge_batch_dim(x::BatchedArray{T, NI, N}) where {T, NI, N} = merge_batch_dim(Val(N-NI), x)
merge_batch_dim(::Val{1}, x::BatchedArray) = x.parent
merge_batch_dim(::Val, x::BatchedArray) = reshape(x.parent, merged_size(x)...)

function check_batch_dim_size(x, xs::BatchedArray...)
    first_batch_size = batch_size(x)
    for other in xs
        other != first_batch_size || error("Batch size mismatch expect $(first_batch_size) got $(batch_size(other))")
    end
    true
end

const BatchedVector{T, N, AT} = BatchedArray{T, 1, N, AT}
const BatchedMatrix{T, N, AT} = BatchedArray{T, 2, N, AT}

BatchedVector(data::AbstractArray) = BatchedArray(1, data)
BatchedMatrix(data::AbstractArray) = BatchedArray(2, data)

# Batched Trace

function LinearAlgebra.tr(A::BatchedMatrix)
    out = BatchedArray(0, similar(A.parent, batch_size(A)))
    batch_out = merge_batch_dim(out)
    trace!(batch_out, merge_batch_dim(A))
    out
end

"""
    trace!(B::AbstractVector{T}, A::AbstractArray{T, 3})

Perform batched matrix trace.
"""
function trace!(B::AbstractVector{T}, A::AbstractArray{T, 3}) where T
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
