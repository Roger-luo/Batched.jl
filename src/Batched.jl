module Batched

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("Batched.jl has not been built, please run Pkg.build(\"Batched\").")
include(ext)

include("abstractarray.jl")
include("adjtrans.jl")
include("batched_scale.jl")

include("routines/linalg.jl")
include("routines/blas.jl")
include("matmul.jl")

# @static if USE_CUDA
#     include("cuda/cuda.jl")
# end

end # module
