using CuArrays.CUBLAS: @check, cublasop

function cublasSgemmStridedBatched(
               handle, transA, transB,
               m, n, k,
               alpha,
               A, lda, strideA,
               B, ldb, strideB,
               beta, C, ldc, strideC)

  @check ccall((:cublasSgemmStridedBatched, libcublas),
               cublasStatus_t,
               (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint,
                Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Ptr{Cfloat}, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat},
                Cint, Cint),
               handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC)
end

function batched_gemm!(transA::Char,
               transB::Char,
               alpha::Float32,
               A::CuBatchedVecOrMat{Float32},
               B::CuBatchedVecOrMat{Float32},
               beta::Float32,
               C::CuBatchedVecOrMat{Float32})
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)
    if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch(""))
    end
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    ldc = max(1,stride(C,2))

    strideA = stride(A, 3)
    strideB = stride(B, 3)
    strideC = stride(C, 3)
    @check ccall((:cublasSgemmStridedBatched, libcublas), cublasStatus_t,
                 (cublasHandle_t, cublasOperation_t,
                  cublasOperation_t, Cint, Cint, Cint, Ptr{Float32},
                  Ptr{Float32}, Cint, Cint, Ptr{Float32}, Cint, Cint, Ptr{Float32},
                  Ptr{Float32}, Cint, Cint),
                 handle(), cutransA,
                 cutransB, m, n, k, [alpha], A, lda, strideA, B, ldb, strideB, [beta],
                 C, ldc, strideC)
    C
end
