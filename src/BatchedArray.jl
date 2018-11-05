export BatchedArray, BatchedMatrix, BatchedVector

import LinearAlgebra
import LinearAlgebra: BLAS

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
LinearAlgebra.BLAS.gemm(tA::AbstractChar, tB::AbstractChar, A::BatchedMatrix{T}, B::BatchedMatrix{T}) where T =
    LinearAlgebra.BLAS.gemm(tA, tB, one(T), A, B)

function LinearAlgebra.BLAS.gemm(tA::AbstractChar, tB::AbstractChar, alpha::T, A::BatchedMatrix{T}, B::BatchedMatrix{T}) where T
    data = similar(A.data, (size(A, 1), size(B, 2), batch_size(A)...))
    fill!(data, zero(T))
    output = BatchedMatrix(data)
    LinearAlgebra.BLAS.gemm!(tA, tB, alpha, A, B, one(T), output)
end

function LinearAlgebra.BLAS.gemm!(tA::AbstractChar, tB::AbstractChar, alpha, A::BatchedMatrix, B::BatchedMatrix, beta, C::BatchedMatrix)
    @boundscheck check_batch_dim_size(A, B, C)
    batchA, batchB, batchC = merge_batch_dim(A), merge_batch_dim(B), merge_batch_dim(C)
    gemm!(tA, tB, alpha, batchA, batchB, beta, batchC)
    C
end

for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))
    @eval begin
        function gemm!(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3}, beta::($elty), C::AbstractArray{$elty, 3})
            @assert !BLAS.has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end
            BLAS.chkstride1(A)
            BLAS.chkstride1(B)
            BLAS.chkstride1(C)

            ptrA = Base.unsafe_convert(Ptr{$elty}, A)
            ptrB = Base.unsafe_convert(Ptr{$elty}, B)
            ptrC = Base.unsafe_convert(Ptr{$elty}, C)

            for k in 1:size(A, 3)
                ccall((LinearAlgebra.BLAS.@blasfunc($gemm), BLAS.libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt},
                     Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BLAS.BlasInt},
                     Ptr{$elty}, Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty},
                     Ref{BLAS.BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,stride(A,2)),
                     ptrB, max(1,stride(B,2)), beta, ptrC,
                     max(1,stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * 8
                ptrB += size(B, 1) * size(B, 2) * 8
                ptrC += size(C, 1) * size(C, 2) * 8
            end
            C
        end
        function gemm(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            gemm!(transA, transB, alpha, A, B, zero($elty), similar(B, $elty, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), size(B, 3))))
        end
        function gemm(transA::AbstractChar, transB::AbstractChar, A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            gemm(transA, transB, one($elty), A, B)
        end
    end
end

# Batched Trace

function LinearAlgebra.tr(A::BatchedMatrix)
    out = BatchedArray(0, similar(A.data, batch_size(A)))
    batch_out = merge_batch_dim(out)
    trace!(batch_out, merge_batch_dim(A))
    out
end

"""
    trace!(B::AbstractVector{T}, A::AbstractArray{T, 3})

Perform batched matrix trace.
"""
function trace!(B::AbstractVector{T}, A::AbstractArray{T, 3}) where T
    @assert size(A, 1) == size(A, 2) "Expect a square matrix" # checksquare
    @boundscheck size(A, 3) == size(B, 1) || error("Batch size mismatch")

    nbatch = size(A, 3)
    n = size(A, 1)
    @inbounds for k in 1:nbatch
        for i in 1:n
            B[k] += A[i, i, k]
        end
    end
    B
end

struct BatchedTranspose{T, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::BatchedMatrix
end
