using Batched, BenchmarkTools

A = BatchedMatrix(rand(10, 10, 30, 5))
@benchmark 0.1 * A

@profiler for i in 1:4000
    0.1 * A
end

B = rand(10, 10, 30, 5)

@benchmark 0.1 * B

@profiler for i in 1:4000
    0.1 * B
end

t = Broadcast.broadcasted(*, 0.1, B)

@which copy(t)

@which copyto!(similar(B), t)

convert(Broadcast.Broadcast)

Base.isconcretetype(Float64)
