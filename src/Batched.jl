module Batched

# using LinearAlgebra

include("BatchedArray.jl")
include("BatchedScale.jl")
include("adjtrans.jl")

include("routines/blas.jl")
include("routines/linalg.jl")

include("matmul.jl")

end # module
