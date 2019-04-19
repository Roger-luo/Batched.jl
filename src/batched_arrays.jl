export BatchedArray, BatchedMatrix, BatchedVector, BatchedScalar, Element,
    element_axes,
    element_size

using LinearAlgebra

struct Element{T, N, P} <: AbstractArray{T, N}
    parent::P
    offset::Int
end

Element{T, N}(A::P, offset::Int) where {T, N, P} = Element{T, N, P}(A, offset)

Base.parent(A::Element) = A.parent
Base.axes(A::Element{T, N}) where {T, N} = ntuple(i->axes(parent(A), i), N)
Base.size(A::Element{T, N}) where {T, N} = ntuple(i->size(parent(A), i), N)
Base.strides(A::Element{T, N}) where {T, N} = ntuple(i->stride(parent(A), i), N)

Base.getindex(A::Element, inds::Int...) = _getindex(A, inds)
Base.setindex!(A::Element, v, inds::Int...) = _setindex!(A, v, inds)

_setindex!(A::Element{T, N}, v, inds::NTuple{N, Int}) where {T, N} = setindex!(parent(A), v, _offset(A, inds...) + A.offset + 1)
_getindex(A::Element{T, N}, inds::NTuple{N, Int}) where {T, N} = A.parent[_offset(A, inds...) + A.offset + 1]

function _offset(A::AbstractArray{T, N}, inds::Int...) where {T, N}
    offset = 0
    for (k, idx) in enumerate(inds)
        offset += (idx - 1) * stride(A, k)
    end
    return offset
end

struct BatchedArray{T, N, B, P} <: AbstractArray{Element{T, N}, B}
    parent::P
end

BatchedArray(n::Int, A::AbstractArray) = BatchedArray{eltype(A), n, ndims(A) - n, typeof(A)}(A)
BatchedArray(A::AbstractArray{T, 3}) where T = BatchedArray{T, 2, 1, typeof(A)}(A)
BatchedArray(A::AbstractArray{T, 2}) where T = BatchedArray{T, 1, 1, typeof(A)}(A)
BatchedArray(A::AbstractArray{T, 1}) where T = BatchedArray{T, 0, 1, typeof(A)}(A)

const BatchedMatrix{T, B, P} = BatchedArray{T, 3, B, P}
const BatchedVector{T, B, P} = BatchedArray{T, 2, B, P}
const BatchedScalar{T, B, P} = BatchedArray{T, 1, B, P}

Base.parent(A::BatchedArray) = A.parent
Base.axes(A::BatchedArray{T, N, B}) where {T, N, B} = ntuple(i->axes(parent(A), i+N), B)
Base.size(A::BatchedArray{T, N, B}) where {T, N, B} = ntuple(i->size(parent(A), i+N), B)
Base.strides(A::BatchedArray{T, N, B}) where {T, N, B} = ntuple(i->stride(parent(A), i+N), B)
Base.stride(A::BatchedArray{T, N}, idx::Int) where {T, N} = stride(parent(A), idx+N)

Base.pointer(A::Element) = pointer(A.parent, 1)
Base.getindex(A::BatchedArray{T, N}, inds::Int...) where {T, N} = Element{T, N}(parent(A), _offset(A, inds...))

function Base.setindex!(A::BatchedArray, v, inds...)
    return A[inds...] .= v
end

# creates an Array by default
function Base.similar(::Type{<:BatchedArray{T}}, element_axes::NTuple{N, <:Base.OneTo}, batch_axes::NTuple{B, <:Base.OneTo}) where {T, N, B}
    P = similar(Array{T}, (element_axes..., batch_axes...))
    return BatchedArray{T, N, B, typeof(P)}(P)
end

element_axes(A::BatchedArray{T, N}) where {T, N} = ntuple(i->axes(parent(A), i), N)
element_size(A::BatchedArray{T, N}) where {T, N} = ntuple(i->size(parent(A), i), N)
element_axes(A::AbstractArray) = axes(A)
element_size(A::AbstractArray) = size(A)
