# Batched.jl

Batched operations in Julia.


## Supported Operations

**(CPU)**: CPU implementations are just wrappers of for-loops for convenience.

- [x] batched `gemm`: `bgemm`
- [x] batched `tr`: `btr`
- [x] batched `transpose`: `transpose(::AbstractArray{T, 3})`
- [ ] batched `adjoint`
