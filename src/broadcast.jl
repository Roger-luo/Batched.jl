# struct BatchedArrayStyle{NI, AT} <: Broadcast.BroadcastStyle end
# Base.BroadcastStyle(::Type{<:AbstractBatchedArray{T, NI, N, AT}}) where {T, NI, N, AT} = BatchedArrayStyle{NI, AT}()
# Broadcast.BroadcastStyle(s::BatchedArrayStyle, x::Broadcast.BroadcastStyle) = s
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
