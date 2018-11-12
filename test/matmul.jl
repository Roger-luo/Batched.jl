using Test, Batched

@testset "checking BatchedMatrix matmul" begin

    A = BatchedMatrix(rand(3, 4, 2, 3))
    B = BatchedMatrix(rand(4, 3, 2, 3))

    C = A * B

    vA = A.parent
    vB = B.parent
    vC = zeros(3, 3, 2, 3)

    for i in 1:2
        for j in 1:3
            vC[:, :, i, j] = vA[:, :, i, j] * vB[:, :, i, j]
        end
    end

    @test vC ≈ C.parent

end

@testset "checking BatchedMatrix matmul (transposed)" begin

    A = BatchedMatrix(rand(3, 4, 2, 3))
    B = BatchedMatrix(rand(3, 4, 2, 3))

    C = A * transpose(B)

    vA = A.parent
    vB = B.parent
    vC = zeros(3, 3, 2, 3)

    for i in 1:2
        for j in 1:3
            vC[:, :, i, j] = vA[:, :, i, j] * transpose(vB[:, :, i, j])
        end
    end

    @test vC ≈ C.parent

end


@testset "checking BatchedMatrix matmul (scale)" begin

    A = BatchedMatrix(rand(3, 4, 2, 3))
    B = BatchedUniformScaling(rand(2, 3))

    C = B * A

    vA = A.parent
    vB = B.scalars
    vC = zeros(3, 4, 2, 3)

    for i in 1:2
        for j in 1:3
            vC[:, :, i, j] = vB[i, j] * vA[:, :, i, j]
        end
    end

    @test vC ≈ C.parent

end

@testset "checking BatchedMatrix matmul (scale & transpose)" begin

    A = BatchedMatrix(rand(3, 4, 2, 3))
    B = BatchedUniformScaling(rand(2, 3))

    C = B * transpose(A)

    vA = A.parent
    vB = B.scalars
    vC = zeros(4, 3, 2, 3)

    for i in 1:2
        for j in 1:3
            vC[:, :, i, j] = vB[i, j] * transpose(vA[:, :, i, j])
        end
    end

    @test vC ≈ C.parent
end
