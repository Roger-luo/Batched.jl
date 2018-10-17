module Batched

using LinearAlgebra

include("gemm.jl")
include("adjtrans.jl")
include("scalar_identity.jl")

end # module
