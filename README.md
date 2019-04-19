# Batched.jl

[![Build Status](https://travis-ci.org/Roger-luo/Batched.jl.svg?branch=master)](https://travis-ci.org/Roger-luo/Batched.jl)

BatchedArrays in Julia.

**Warning**: this is still under its early stage, use at your own risk.

## Batched Arrays

`BatchedArray` is like an `Array` of `Array`s, but with contiguous storage in memory, so we could do some optimization based on that.

```julia
julia> A = BatchedArray(rand(2, 3, 10));

julia> B = BatchedArray(rand(3, 2, 10));

julia> eltype(A)
Element{Float64,2,P} where P
```

And with Julia's broadcast, this will just work:

```julia
julia> A .* B
10-element BatchedArray{Float64,2,1,Array{Float64,3}}:
 [0.947403 1.40625; 0.869711 0.848727] 
 [0.555413 0.699582; 0.465844 0.871226]
 [0.38841 0.381471; 0.551932 0.309496] 
 [0.93626 0.408086; 0.880168 0.311356] 
 [1.49227 0.941464; 1.26334 0.804171]  
 [1.08533 0.536161; 0.468115 0.188942] 
 [0.434965 0.813795; 0.367398 0.264233]
 [0.302816 0.990396; 0.674112 0.966444]
 [0.626944 0.89332; 1.30737 1.82809]   
 [0.582451 1.23067; 0.889868 1.36178]
```
