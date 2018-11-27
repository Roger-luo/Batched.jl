module Batched

using BatchedRoutines

export batched_tr

include("abstractarray.jl")
include("adjtrans.jl")
include("batched_scale.jl")
include("matmul.jl")

end # module
