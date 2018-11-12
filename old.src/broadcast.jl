# struct BatchedArrayStyle <: Broadcast.BroadcastStyle end
# Base.BroadcastStyle(::Type{<:AbstractBatchedArray}) = BatchedArrayStyle()
# Broadcast.BroadcastStyle(s::BatchedArrayStyle, x::Broadcast.BroadcastStyle) = s
#
# struct BatchedBroadcasted{BC}
#     bc::BC
# end
#
# @inline broadcast_unbatch(x) = x
# @inline broadcast_unbatch(x::AbstractBatchedArray) = broadcast_unbatch(x.parent)
# @inline broadcast_unbatch(x::BatchedUniformScaling) = error("cannot broadcast BatchedUniformScaling")
#
# @inline Base.ndims(::Type{BatchedBroadcasted{BC}}) where BC = ndims(BC)
# @inline Base.ndims(bc::BatchedBroadcasted) = ndims(bc.bc)
# @inline Base.getindex(bc::BatchedBroadcasted, I...) = getindex(bc.bc, I...)
#
# @inline Broadcast.broadcasted(::BatchedArrayStyle, f, args...) =
#     BatchedBroadcasted(Broadcast.broadcasted(f, map(broadcast_unbatch, args)...))
# @inline Broadcast.materialize(x::BatchedBroadcasted) = Broadcast.materialize(x.bc)

# Broadcast.BroadcastStyle(s::BatchedArrayStyle{N1}, x::BatchedArrayStyle{N2}) where {N1, N2} =
#     error("cannot broadcast on different batch")
# Broadcast.BroadcastStyle(s::BatchedArrayStyle{N}, x::BatchedArrayStyle{N}) where N = s
#
# Base.copy(bc::Broadcast.Broadcasted{BatchedArrayStyle{NI, AT}}) where {NI, AT} =
#     BatchedArray(NI, copy(convert(Broadcast.Broadcasted{Nothing}, bc)))

# Base.similar(bc::Broadcast.Broadcasted{BatchedArrayStyle{NI, AT}}, ::Type{ElType}) where {ElType, NI, T, N, AT <: AbstractArray{T, N}} =
#     similar(BatchedArray{ElType, NI, N, AT}, axes(bc))
#
# Base.similar(BatchedArray{Float64, 2, 3, Array{Float64, 3}}, (2, 2, 2))
