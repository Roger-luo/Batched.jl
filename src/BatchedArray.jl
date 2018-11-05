export BatchedArray, BatchedMatrix, BatchedVector

import LinearAlgebra

struct BatchedArray{T, NI, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::AT
end

BatchedArray(NI::Int, data::AT) where {T, N, AT <: AbstractArray{T, N}} = BatchedArray{T, NI, N, AT}(data)

Base.size(x::BatchedArray) = size(x.data)
Base.strides(x::BatchedArray) = strides(x.data)
Base.getindex(x::BatchedArray, I...) = getindex(x.data, I...)
Base.setindex!(x::BatchedArray, v, I...) = setindex!(x.data, v, I...)
Base.IndexStyle(x::BatchedArray) = IndexStyle(x.data)

inner_size(x::BatchedArray{T, NI, N}) where {T, NI, N} = Tuple(size(x, i) for i in Base.OneTo(NI))
batch_size(x::BatchedArray{T, NI, N}) where {T, NI, N} = Tuple(size(x, i) for i in (NI+1):N)
batch_size(x::BatchedArray, i::Int) = batch_size(x, i)
merged_size(x::BatchedArray) = (inner_size(x)..., prod(batch_size(x)))

merge_batch_dim(x::BatchedArray{T, NI, N}) where {T, NI, N} = merge_batch_dim(Val(N-NI), x)
merge_batch_dim(::Val{1}, x::BatchedArray) = x.data
merge_batch_dim(::Val, x::BatchedArray) = reshape(x.data, merged_size(x)...)

function check_batch_dim_size(x, xs::BatchedArray...)
    first_batch_size = batch_size(x)
    for other in xs
        other != first_batch_size || error("Batch size mismatch expect $(first_batch_size) got $(batch_size(other))")
    end
    true
end

const BatchedVector{T, N, AT} = BatchedArray{T, 1, N, AT}
const BatchedMatrix{T, N, AT} = BatchedArray{T, 2, N, AT}

BatchedVector(data::AbstractArray) = BatchedArray(1, data)
BatchedMatrix(data::AbstractArray) = BatchedArray(2, data)

Base.:(*)(lhs::BatchedMatrix, rhs::BatchedMatrix) = LinearAlgebra.BLAS.gemm(lhs, rhs)

LinearAlgebra.BLAS.gemm(A::BatchedMatrix, B::BatchedMatrix) = LinearAlgebra.BLAS.gemm('N', 'N', A, B)
LinearAlgebra.BLAS.gemm(tA::Char, tB::Char, A::BatchedMatrix{T}, B::BatchedMatrix{T}) where T =
    LinearAlgebra.BLAS.gemm(tA, tB, one(T), A, B)

function LinearAlgebra.BLAS.gemm(tA::Char, tB::Char, alpha::T, A::BatchedMatrix{T}, B::BatchedMatrix{T}) where T
    data = similar(A.data, (size(A, 1), size(B, 2), batch_size(A)...))
    fill!(data, zero(T))
    output = BatchedMatrix(data)
    LinearAlgebra.BLAS.gemm!(tA, tB, alpha, A, B, one(T), output)
end

function LinearAlgebra.BLAS.gemm!(tA::Char, tB::Char, alpha, A::BatchedMatrix, B::BatchedMatrix, beta, C::BatchedMatrix)
    @boundscheck check_batch_dim_size(A, B, C)
    batchA, batchB, batchC = merge_batch_dim(A), merge_batch_dim(B), merge_batch_dim(C)
    gemm!(tA, tB, alpha, batchA, batchB, beta, batchC)
    C
end

function LinearAlgebra.tr(A::BatchedMatrix)
    out = BatchedArray(0, similar(A.data, batch_size(A)))
    batch_out = merge_batch_dim(out)
    tr!(batch_out, merge_batch_dim(A))
    out
end

function gemm!(tA::Char, tB::Char, alpha::T, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}, beta::T, C::AbstractArray{T, 3}) where T
    @assert size(A, 3) == size(B, 3) == size(C, 3) "Batch size mismatch"
    for k in 1:size(A, 3)
        LinearAlgebra.BLAS.gemm!(tA, tB, alpha, selectdim(A, 3, k), selectdim(B, 3, k), beta, selectdim(C, 3, k))
    end
    C
end

function tr!(out::AbstractVector{T}, A::AbstractArray{T, 3}) where T
    @inbounds for k in 1:size(A, 3)
        out[k] = LinearAlgebra.tr(selectdim(A, 3, k))
    end
    out
end

struct BatchedTranspose{T, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::BatchedMatrix
end
