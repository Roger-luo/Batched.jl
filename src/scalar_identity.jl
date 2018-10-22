export ScalarIdentity

"""
    ScalarIdentity{B, K, T} <: AbstractArray{T, 3}

A batch of scalar multiplies a batch of identities, where batch size is
`B`, each identity's size is `K`.
"""
struct ScalarIdentity{B, K, T, VT <: AbstractVector{T}} <: AbstractArray{T, 3}
    scalars::VT
    ScalarIdentity{B, K}(scalars::VT) where {B, K, T, VT <: AbstractVector{T}} = new{B, K, T, VT}(scalars)
end

Base.size(x::ScalarIdentity{B, K, T}) where {B, K, T} = (K, K, B)
Base.@propagate_inbounds Base.getindex(m::ScalarIdentity{B, K, T}, i::Int, j::Int, k::Int) where {B, K, T} =
    i == j ? getindex(m.scalars, k) : zero(T)
Base.IndexStyle(::Type{<:ScalarIdentity}) = IndexCartesian()
Base.transpose(A::ScalarIdentity) = A

function bgemm!(A::ScalarIdentity{NBatch, K, T}, B::Transpose{NBatch, T}, C::AbstractArray{T, 3}) where {NBatch, K, T}
    @inbounds for i in 1:NBatch
        C[:, :, i] .+= A.scalars[i] * view(B, :, :, i)
    end
    C
end

function bgemm!(A::ScalarIdentity{NBatch, K, T}, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where {NBatch, K, T}
    @inbounds for i in 1:NBatch
        C[:, :, i] .+= A.scalars[i] * view(B, :, :, i)
    end
    C
end

bgemm!(A::AbstractArray{T, 3}, B::ScalarIdentity{NBatch, K, T}, C::AbstractArray{T, 3}) where {NBatch, K, T} = bgemm!(B, A, C)
bgemm!(A::Transpose{NBatch, T}, B::ScalarIdentity{NBatch, K, T}, C::AbstractArray{T, 3}) where {NBatch, K, T} = bgemm!(B, A, C)
