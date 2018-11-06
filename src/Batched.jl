module Batched

using Requires
# using LinearAlgebra

include("BatchedArray.jl")
include("BatchedScale.jl")
include("adjtrans.jl")

include("routines/blas.jl")
include("routines/linalg.jl")

include("matmul.jl")

function __init__()
    @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda/cuda.jl")
end

end # module
