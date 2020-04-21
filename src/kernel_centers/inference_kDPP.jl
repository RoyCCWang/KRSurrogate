
# Algorithm 7 of Determinantal Point Processes for Machine Learning.

function getelementarysympolynomials(k::Int, x::Vector{T})::Matrix{T} where T <: Real
    N = length(x)

    J = Matrix{T}(undef, k+1, N+1)
    getelementarysympolynomials!(J, k, x)

    return J
end

# modifies the intermediate buffer J.
function getelementarysympolynomials!(J::Matrix{T}, k::Int, x::Vector{T})::Nothing where T <: Real
    N = length(x)

    @assert size(J) == (k+1,N+1)

    for n = 0:N
        J[1, n+1] = one(T)
    end

    for l = 1:k
        J[l+1, 1] = zero(T)
    end

    for l = 1:k

        for n = 1:N

            J[l+1, n+1] = J[l+1, n-1+1] + x[n]*J[l-1+1, n-1+1]
        end
    end

    return nothing
end

function samplekDPP(k::Int, λ::Vector{T})::Tuple{Vector{Int},Matrix{T}} where T <: Real
    N = length(λ)

    e_mat = getelementarysympolynomials(k, λ)

    J::Vector{Int} = Vector{Int}(undef,0)
    l = k
    for n = N:-1:1
        if l == 0
            return J, e_mat
        end

        if rand() < λ[n]*e_mat[l-1+1,n-1+1]/e_mat[l+1,n+1]
            push!(J,n)
            l -= 1
        end
    end

    return J, e_mat
end
