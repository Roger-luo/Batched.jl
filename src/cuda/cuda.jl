using CuArrays, CUDAdrv

const CuBatchedArray{T, NI, N} = BatchedArray{T, NI, N, CuArray{T, N}}
const CuBatchedMatrix{T, N} = CuBatchedArray{T, 2, N}
const CuBatchedVector{T, N} = CuBatchedArray{T, 1, N}

include("routines/blas.jl")
