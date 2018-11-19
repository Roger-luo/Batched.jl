
# FIXME: implement this with native cuda kernel
function batched_tr(A::CuArray{T, 3}) where
    sum(view(A, i, i, :) for i in 1:size(A, 1))
end
