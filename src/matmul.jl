const _BATCHED_MATRIX_LIST = [
        (:BatchedMatrix, 'N'),
        (:(BatchedTranspose{T, N, <:BatchedMatrix} where N), 'T'),
        (:(BatchedAdjoint{T, N, <:BatchedMatrix} where N), 'C')
]

for (TA, transA) in _BATCHED_MATRIX_LIST, (TB, transB) in _BATCHED_MATRIX_LIST
    @eval function LinearAlgebra.mul!(C::BatchedMatrix{T}, A::$TA, B::$TB) where T
        @boundscheck check_batch_dim_size(A, B, C)
        batchA, batchB, batchC = merge_batch_dim(A), merge_batch_dim(B), merge_batch_dim(C)
        batched_gemm!($transA, $transB, one(T), batchA, batchB, one(T), batchC)
        C
    end
end

# Base.:(*)(A::BatchedUniformScaling, B::AbstractArray{T, 3}) where T =
#     batched_mul!(similar(B), A.scalars, B)
# Base.:(*)(A::AbstractArray{T, 3}, B::BatchedUniformScaling) where T = B * A
#
# Base.:(*)(A::BatchedUniformScaling, B::AbstractBatchedArray) =
#     LinearAlgebra.mul!(similar(B), A, B)
# Base.:(*)(A::AbstractBatchedArray, B::BatchedUniformScaling) = B * A
#
# function LinearAlgebra.mul!(C::BatchedArray{T, NI}, A::BatchedUniformScaling{T}, B::AbstractBatchedArray{T, NI}) where {T, NI}
#     @boundscheck check_batch_dim_size(A, B, C)
#     batchA = merge_batch_dim(A)
#     batchB = merge_batch_dim(B)
#     batchC = merge_batch_dim(C)
#
#     @inbounds for k in 1:size(batchC, 3)
#         for j in 1:size(batchC, 2)
#             for i in 1:size(batchC, 1)
#                 C[i, j, k] = A[k] * B[i, j, k]
#             end
#         end
#     end
#     C
# end
#
# function LinearAlgebra.mul!(C::BatchedMatrix{T}, A::BatchedUniformScaling{T}, B::BatchedTransposeOrAdjoint) where T
#     @boundscheck check_batch_dim_size(A, B, C)
#     batchA = merge_batch_dim(A)
#     batchB = merge_batch_dim(B)
#     batchC = merge_batch_dim(C)
#
#     @inbounds for k in 1:size(batchC, 3)
#         for j in 1:size(batchC, 2)
#             for i in 1:size(batchC, 1)
#                 C[i, j, k] = A[k] * B[i, j, k]
#             end
#         end
#     end
#     C
# end
#
#
# LinearAlgebra.mul!(C::BatchedArray{T, NI}, A::AbstractBatchedArray{T, NI}, B::BatchedUniformScaling{T}) where {T, NI} =
#     LinearAlgebra.mul!(C, B, A)
