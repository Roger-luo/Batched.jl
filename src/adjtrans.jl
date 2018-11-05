export Transpose

"""
    BatchedTranspose{T, NI, N, AT} <: AbstractBatchedArray{T, NI, N}

Batched transpose. Transpose a batch of matrix.
"""
struct BatchedTranspose{T, N, AT} <: AbstractBatchedMatrix{T, N}
    parent::BatchedMatrix{T, N, AT}
end

Base.size(m::BatchedTranspose) = (size(m.parent, 2), size(m.parent, 1), size(m.parent, 3))
Base.axes(m::BatchedTranspose) = (axes(m.parent, 2), axes(m.parent, 1), axes(m.parent, 3))
Base.@propagate_inbounds Base.getindex(m::BatchedTranspose, i::Int, j::Int, k::Int...) = getindex(m.parent, j, i, k...)
Base.@propagate_inbounds Base.setindex!(m::BatchedTranspose, v, i::Int, j::Int, k::Int...) = setindex!(m.parent, v, j, i, k...)
Base.IndexStyle(::Type{<:BatchedTranspose}) = IndexCartesian()
Base.transpose(A::AbstractBatchedMatrix) = BatchedTranspose(A)

batched_gemm(tA::AbstractChar, tB::AbstractChar, alpha::T, A::BatchedTranspose{T}, B::AbstractBatchedMatrix{T}) where T =
    batched_gemm(tA == 'N' ? 'T' : 'N', tB, A.parent, B)
batched_gemm(tA::AbstractChar, tB::AbstractChar, alpha::T, A::AbstractBatchedMatrix{T}, B::BatchedTranspose{T}) where T =
    batched_gemm(tA, tB == 'N' ? 'T' : 'N', A.parent, B)

batched_gemm(A::BatchedMatrix, B::BatchedTranspose) = batched_gemm('N', 'T', A, B.parent)
batched_gemm(A::BatchedTranspose, B::BatchedTranspose) = batched_gemm('T', 'T', A.parent, B.parent)
