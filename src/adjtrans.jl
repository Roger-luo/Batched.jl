export Transpose

struct Transpose{B, T, AT <: AbstractArray{T, 3}} <: AbstractArray{T, 3}
    parent::AT

    Transpose(A::AT) where {T, AT <: AbstractArray{T, 3}} = new{size(A, 3), T, AT}(A)
end

Base.size(m::Transpose) = (size(m.parent, 2), size(m.parent, 1), size(m.parent, 3))
Base.axes(m::Transpose) = (axes(m.parent, 2), axes(m.parent, 1), axes(m.parent, 3))
Base.@propagate_inbounds Base.getindex(m::Transpose, i::Int, j::Int, k::Int) = getindex(m.parent, j, i, k)
Base.@propagate_inbounds Base.setindex!(m::Transpose, v, i::Int, j::Int, k::Int) = setindex!(m.parent, v, j, i, k)
Base.IndexStyle(::Type{<:Transpose}) = IndexCartesian()
Base.transpose(A::AbstractArray{T, 3}) where T = Transpose(A)

bgemm!(A::AbstractArray{T, 3}, B::Batched.Transpose, C::AbstractArray{T, 3}) where T =
    bgemm!('N', 'T', one(T), A, B.parent, one(T), C)
bgemm!(A::Batched.Transpose, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where T =
    bgemm!('T', 'N', one(T), A.parent, B, one(T), C)
