export batched_mul!, batched_mul

batched_mul(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T =
        batched_mul!(similar(A, (size(A, 1), size(B, 2), size(A, 3))), A, B)

"""
    batched_mul!(C, A, B) -> C

batched `mul!`.
"""
function batched_mul! end

_unbatch(A) = A
_unbatch(A::BatchedTransposeOrAdjoint) = A.parent

# bmm
const _BATCHED_MATRIX_LIST = [
        (:(AbstractArray{T, 3}), 'N'),
        (:(BatchedTranspose{T, N, <:AbstractArray{T, 3}} where N), 'T'),
        (:(BatchedAdjoint{T, N, <:AbstractArray{T, 3}} where N), 'C')
]

for (TA, transA) in _BATCHED_MATRIX_LIST, (TB, transB) in _BATCHED_MATRIX_LIST
    @eval function batched_mul!(C::AbstractArray{T, 3}, A::$TA, B::$TB) where T
        batched_gemm!($transA, $transB, one(T), _unbatch(A), _unbatch(B), zero(T), C)
        C
    end
end

# bmv
function batched_mul!(C::AbstractArray{T, 2}, A::AbstractArray{T, 3}, B::AbstractArray{T, 2}) where T
end
