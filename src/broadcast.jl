import Base.Broadcast: AbstractArrayStyle, combine_styles, DefaultArrayStyle, Broadcasted, broadcast_shape, combine_axes

"""
    BatchedArrayStyle{N, B} <: AbstractArrayStyle{B}


"""
struct BatchedArrayStyle{N, B} <: AbstractArrayStyle{B} end

Broadcast.BroadcastStyle(::Type{<:BatchedArray{T, N, B}}) where {T, N, B} = BatchedArrayStyle{N, B}()
Broadcast.BroadcastStyle(::AbstractArrayStyle{N}, ::BatchedArrayStyle{N, B}) where {N, B} = BatchedArrayStyle{N, B}()
Broadcast.BroadcastStyle(::BatchedArrayStyle{N, B}, ::AbstractArrayStyle{N}) where {N, B} = BatchedArrayStyle{N, B}()
Broadcast.BroadcastStyle(::BatchedArrayStyle{N, B}, ::DefaultArrayStyle{N}) where {N, B} = BatchedArrayStyle{N, B}()

# this is for things like A * B
function Base.similar(bc::Broadcasted{BatchedArrayStyle{N, B}}, ::Type{ElType}) where {N, B, ElType <: AbstractArray}
    similar(BatchedArray{eltype(ElType), N, B}, combine_element_axes(bc.f, bc.args...), combine_batch_axes(bc.args...))
end

function Base.similar(bc::Broadcasted{BatchedArrayStyle{N, B}}, ::Type{ElType}) where {N, B, ElType}
    similar(BatchedArray{ElType, N, B}, combine_element_axes(bc.f, bc.args...), combine_batch_axes(bc.args...))
end

combine_batch_axes(A::AbstractArray, B...) = broadcast_shape((), combine_batch_axes(B...))
combine_batch_axes(A::BatchedArray, B...) = broadcast_shape(axes(A), combine_batch_axes(B...))
combine_batch_axes(A::AbstractArray) = ()
combine_batch_axes(A::BatchedArray) = axes(A)

combine_element_axes(::typeof(*), A, B, C...) = prod_shape(element_axes(A), combine_element_axes(*, B, C...))
combine_element_axes(::typeof(*), A, B) = prod_shape(element_axes(A), element_axes(B))

prod_shape(s) = s
const Axes = Union{Integer, Base.OneTo}
function prod_shape(s1::NTuple{2, <:Axes}, s2::NTuple{2, <:Axes})
    s1[2] == s2[1] && return s1[1], s2[2]
    sizeA = last(s1[1]), last(s1[2])
    sizeB = last(s2[1]), last(s2[2])
    throw(DimensionMismatch("A has dimensions $sizeA, but B has dimensions $sizeB"))
end

function prod_shape(s1::NTuple{2, <:Axes}, s2::Tuple{<:Axes})
    s1[2] == s2[1] && return (s1[1], )
end

combine_element_axes(::typeof(kron), A, B...) = kron_shape(element_axes(A), combine_element_axes(kron, B...))
combine_element_axes(::typeof(kron), A) = element_axes(A)

combine_element_axes(::typeof(tr), A) = ()

kron_shape(s::NTuple{2, <:Axes}) = s
function kron_shape(s1::NTuple{2, <:Axes}, s2::NTuple{2, <:Axes})
    Base.OneTo(last(s1[1]) * last(s2[1])), Base.OneTo(last(s1[2]) * last(s2[2]))
end

# return element axes if not specified
combine_element_axes(f::Function, A) = element_axes(A)
combine_element_axes(f::Function, A, B...) = same_element_axes(element_axes(A), combine_element_axes(f, B...))

same_element_axes(s1::NTuple{2, <:Axes}, s2::NTuple{2, <:Axes}) =
    s1 == s2 ? s1 : DimensionMismatch("dimensions must match")


@inline function copyto!(dest::BatchedArray{T, N, B}, bc::Broadcasted{Nothing}) where {T, N, B}
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    # TODO: optimization
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    bc′ = Broadcast.preprocess(dest, bc)

    @simd for I in eachindex(bc′)
        @inbounds dest[I] = bc′[I]
    end
    return dest
end
