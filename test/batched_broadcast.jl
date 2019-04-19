using Test, Batched
import Base.Broadcast: BroadcastStyle, combine_styles, broadcasted

A = BatchedArray(rand(2, 3, 10));
B = BatchedArray(rand(3, 2, 10));
C = rand(2, 2)

@test combine_styles(A, B) == Batched.BatchedArrayStyle{2, 1}()
@test combine_styles(A, B, C) == Batched.BatchedArrayStyle{2, 1}()

bc = broadcasted(*, A, B)

import Batched: combine_batch_axes

@test combine_batch_axes(A, B) == (Base.OneTo(10), ) # A * B
@test combine_batch_axes(A, B, C) == (Base.OneTo(10), ) # A * B * C

import Batched: combine_element_axes

combine_element_axes(*, A, B) == (Base.OneTo(2), Base.OneTo(2))

copy(bc)

A .* B

# using LinearAlgebra
# tr.(A)