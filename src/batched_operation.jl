struct BatchedOperation{FT, ArgsT, KwargsT}
    f::FT
    args::ArgsT
    kwargs::KwargsT
end

merge_batch_dim(x::AbstractArray{T, 1}) = x
merge_batch_dim(x::AbstractArray{T, 2}) = x
merge_batch_dim(x::AbstractArray{T, 3}) = x

function merge_batch_dim(x::AbstractArray{T, N}; batch_dims=) where {T, N}
    reshape(x, size(x, 1), size(x, 2))
end
