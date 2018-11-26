module Batched

using BatchedRoutines

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("Batched.jl has not been built, please run Pkg.build(\"Batched\").")
include(ext)

include("abstractarray.jl")
include("adjtrans.jl")
include("batched_scale.jl")
include("matmul.jl")

end # module
