using Test, Batched, LinearAlgebra

@testset "Testing batched_gemm! with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]

    alpha, beta = rand(elty), rand(elty)
    A = rand(elty, 10, 10, 100)
    B = rand(elty, 10, 10, 100)
    C = rand(elty, 10, 10, 100)
    test_C = copy(C)

    Batched.batched_gemm!('N', 'N', alpha, A, B, beta, C)

    for k in 1:100
        test_C[:, :, k] = alpha * (A[:, :, k]) * B[:, :, k] + beta * test_C[:, :, k]
    end

    @test C ≈ test_C
end

@testset "Testing batched_gemm with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    alpha = rand(elty)
    A = rand(elty, 10, 10, 100)
    B = rand(elty, 10, 10, 100)

    C = Batched.batched_gemm('N', 'N', alpha, A, B)

    test_C = zeros(elty, 10, 10, 100)
    for k in 1:100
        test_C[:, :, k] = alpha * (A[:, :, k]) * B[:, :, k]
    end

    @test test_C ≈ C
end

@testset "Testing batched_gemm (uniform scaling) with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = BatchedUniformScaling(rand(elty, 100))
    B = rand(elty, 10, 10, 100)

    C = Batched.batched_gemm(A, B)

    test_C = zeros(elty, 10, 10, 100)
    for k in 1:100
        test_C[:, :, k] = (A[k]) * B[:, :, k]
    end

    @test test_C ≈ C
end

@testset "Testing batched_gemm (uniform scaling & transpose) with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = BatchedUniformScaling(rand(elty, 100))
    vB = rand(elty, 10, 8, 100)
    B = batched_transpose(vB)

    C = Batched.batched_gemm(A, B)

    test_C = zeros(elty, 8, 10, 100)
    for k in 1:100
        test_C[:, :, k] = (A[k]) * transpose(vB[:, :, k])
    end

    @test test_C ≈ C
end
