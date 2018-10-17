export bgemm!, bgemm, btr!, btr

# last dim is batch dim
function bgemm!(tA::Char, tB::Char, alpha::T, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}, beta::T, C::AbstractArray{T, 3}) where T
    nbatch = size(C, 3)
    @inbounds for i = 1:nbatch
        BLAS.gemm!(tA, tB, alpha, view(A, :, :, i), view(B, :, :, i), beta, view(C, :, :, i))
    end
    C
end

bgemm!(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}, C::AbstractArray{T, 3}) where T =
    bgemm!('N', 'N', one(T), A, B, one(T), C)

function bgemm(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    @boundscheck size(A, 3) == size(B, 3) || throw(DimensionMismatch("Batch dimension mismatch, got $(size(A, 3)) and $(size(B, 3))."))
    bgemm!(A, B, zeros(T, (size(A, 1), size(B, 2), size(A, 3))))
end

function btr!(A::AbstractArray{T, 3}, out::AbstractArray{T, 1}) where T
    @inbounds for i in eachindex(out)
        out[i] = tr(view(A, :, :, i))
    end
    out
end

btr(A::AbstractArray{T, 3}) where T = btr!(A, Vector{T}(undef, size(A, 3)))
