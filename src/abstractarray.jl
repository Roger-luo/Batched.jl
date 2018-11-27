export AbstractBatchedArray, AbstractBatchedScalar, AbstractBatchedVector, AbstractBatchedMatrix
export datum_size, batch_size, merged_size, merged_batch_size

"""
    AbstractBatchedArray{T, NI, N}

Abstract type batched array. A batched array use its last `N - NI` dimension as
batch dimension, it is a batch of array with dimension `NI`.
"""
abstract type AbstractBatchedArray{T, NI, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N} end

"""
    AbstractBatchedScalar{T, N}

Batched scalars.
"""
const AbstractBatchedScalar{T, N, AT} = AbstractBatchedArray{T, 0, N, AT}

"""
    AbstractBatchedVector{T, N}

Batched vector.
"""
const AbstractBatchedVector{T, N, AT} = AbstractBatchedArray{T, 1, N, AT}

"""
    AbstractBatchedMatrix{T, N}

Batched matrix.
"""
const AbstractBatchedMatrix{T, N, AT} = AbstractBatchedArray{T, 2, N, AT}

"""
    datum_size(batched_array) -> Tuple

Returns a tuple of size of each inner dimension of the batched array.
"""
function datum_size end

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

merged_batch_size(A::AbstractBatchedArray) = prod(batch_size(A))
merged_size(A::AbstractBatchedArray) = (datum_size(A)..., merged_batch_size(A))

function check_batch_dim_size(x::AbstractBatchedArray, xs::AbstractBatchedArray...)
    first_batch_size = batch_size(x)
    for other in xs
        other != first_batch_size || error("Batch size mismatch expect $(first_batch_size) got $(batch_size(other))")
    end
    true
end
