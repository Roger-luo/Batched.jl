export ScalarIdentity

struct ScalarIdentity{B, K, T} <: AbstractArray{T, 3}
    scalars::Vector{T}
    ScalarIdentity{B, K}(scalars::Vector{T}) where {B, K, T} = new{B, K, T}(scalars)
end

Base.size(x::ScalarIdentity{B, K, T}) where {B, K, T} = (K, K, B)
Base.@propagate_inbounds Base.getindex(m::ScalarIdentity{B, K, T}, i::Int, j::Int, k::Int) where {B, K, T} =
    i == j ? getindex(m.scalars, k) : zero(T)
Base.IndexStyle(::Type{<:ScalarIdentity}) = IndexCartesian()
Base.transpose(A::ScalarIdentity) = A

function bgemm!(A::AbstractArray{T, 3}, B::ScalarIdentity{NBatch, K, T}, C::AbstractArray{T, 3}) where {T, NBatch, K}
    @inbounds for i in 1:NBatch
        view(C, :, :, i) .+= B.scalars[i] * view(A, :, :, i)
    end
    C
end

function bgemm!(A::ScalarIdentity{NBatch, K, T}, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where {T, NBatch, K}
    @inbounds for i in 1:NBatch
        view(C, :, :, i) .+= A.scalars[i] * view(B, :, :, i)
    end
    C
end
