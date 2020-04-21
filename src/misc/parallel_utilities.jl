
function parallelevals( f::Function,
                        X,
                        dummy::T) where T <: Real
    #
    N = length(X)

    a = SharedArray{T}(N)
    @sync begin
        @distributed for i = 1:N
            a[i] = f(X[i])
        end
    end

    return convert(Vector{T}, a)
end

function parallelevals( f::Function,
                        X::Vector{Vector{T}}) where T <: Real
    #
    N = length(X)

    a = SharedArray{T}(N)
    @sync begin
        @distributed for i = 1:N
            a[i] = f(X[i])
        end
    end

    return convert(Vector{T}, a)
end
