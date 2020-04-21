
# assume y is within the radius of convergence of y0.
function evalTaylorinverse(  y::T,
                            y0::T,
                            x0::T,
                            dF_array::Vector{Function},
                            N::Int)::T where T <: Real

    # prepare derivatives functions.
    #N = length(dF_array)

    # compute derivatives for the forward function.
    #dⁿy_array = [ dF(x0); d2F(x0); d3F(x0); d4F(x0); d5F(x0)]
    dⁿy_array = collect( dF_array[i](x0) for i = 1:N )
    #@time collect( dF_array[i](x0) for i = 1:N )

    return evalTaylorinverse(y, y0, x0, dⁿy_array, N)
end

function evalTaylorinverse(  y::T,
                            y0::T,
                            x0::T,
                            dⁿy_array::Vector{T},
                            N::Int)::T where T <: Real

    # compute derivatives of the inverse.
    dⁿx_array, P = computeTaylorinversederivatives(dⁿy_array, x0, N)
    #@time computeTaylorinversederivatives(dⁿy_array, x0, N)

    ξ = y-y0

    x_eval = sum( dⁿx_array[n]*ξ^n/factorial(n) for n = 1:N ) +x0
    #@time (sum( dⁿx_array[n]*ξ^n/factorial(n) for n = 1:N ) +x0)

    return x_eval
end

# x is a scalar.
function computeTaylorinversederivatives(   dⁿy_array::Vector{T},
                                            x,
                                            N::Int)::Tuple{Vector{T},Matrix{T}} where T <: Real
    #


    P = computerTaylorP(N, dⁿy_array)

    dⁿx_array = computealldⁿx(N, P, dⁿy_array[1])

    return dⁿx_array, P
end


function computealldⁿx( N::Int,
                        P::Matrix{T},
                        dy::T)::Vector{T} where T

    # allocate.
    dⁿx_array = Vector{T}(undef,N)

    # fill in values.
    dⁿx_array[1] = one(T)/dy

    for n = 2:N

        multiplier = -factorial(n)/dy^n

        running_sum = zero(T)
        for j = 1:n-1
            running_sum += dⁿx_array[j]/factorial(j)*P[j,n]
        end

        dⁿx_array[n] = multiplier*running_sum
    end

    return dⁿx_array
end

# P[j,k] is P_{j,k}.
function computerTaylorP(N::Int, dᵐy_array::Vector{T})::Matrix{T} where T <: Real

    #@assert k >= j
    P = Matrix{T}(undef,N-1,N)
    #fill!(P,-Inf)

    for j = 1:N-1

        # k == 1
        P[j,j] = dᵐy_array[1]^j

        # k > 1.
        for k = j+1:N

            multiplier = 1/( (k-j)*dᵐy_array[1] )

            running_sum = zero(T)
            for l = 1:k-j
                tmp = l*j -k +j +l

                running_sum += tmp*dᵐy_array[l+1]*P[j,k-l]/factorial(l+1)
            end


            P[j,k] = multiplier*running_sum

            #println("P[j,k] = ", P[j,k])
            #@assert 55==4
        end
    end

    return P
end

#### factorial included.
function evalTaylorinversewfactorial(   y::T,
                                        y0::T,
                                        x0::T,
                                        dF_array::Vector{Function},
                                        N::Int)::T where T <: Real


    c_y_array = collect( dF_array[i](x0) for i = 1:N )

    return evalTaylorinversewfactorial(y, y0, x0, c_y_array, N)
end

# c_y_array assumed to have been divided by factorial(n) already.
function evalTaylorinversewfactorial(  y::T,
                            y0::T,
                            x0::T,
                            c_y_array::Vector{T},
                            N::Int)::T where T <: Real

    # compute derivatives of the inverse.
    c_x_array, P = computeTaylorinversederivativeswfactorial(c_y_array, x0, N)
    #@time computeTaylorinversederivatives(dⁿy_array, x0, N)

    ξ = y-y0

    x_eval = sum( c_x_array[n]*ξ^n for n = 1:N ) +x0

    return x_eval
end

# x is a scalar.
function computeTaylorinversederivativeswfactorial(   c_y_array::Vector{T},
                                            x,
                                            N::Int)::Tuple{Vector{T},Matrix{T}} where T <: Real
    #


    P = computerTaylorPwfactorial(N, c_y_array)

    c_x_array = computealldⁿxwfactorial(N, P, c_y_array[1])

    return c_x_array, P
end


function computealldⁿxwfactorial( N::Int,
                        P::Matrix{T},
                        dy::T)::Vector{T} where T

    # allocate.
    c_x_array = Vector{T}(undef,N)

    # fill in values.
    c_x_array[1] = one(T)/dy

    for n = 2:N

        multiplier = -one(T)/dy^n

        running_sum = zero(T)
        for j = 1:n-1
            running_sum += c_x_array[j]*P[j,n]
        end

        c_x_array[n] = multiplier*running_sum
    end

    return c_x_array
end

# P[j,k] is P_{j,k}.
function computerTaylorPwfactorial(N::Int, c_y_array::Vector{T})::Matrix{T} where T <: Real

    #@assert k >= j
    P = Matrix{T}(undef,N-1,N)
    #fill!(P,-Inf)

    for j = 1:N-1

        # k == 1
        P[j,j] = c_y_array[1]^j

        # k > 1.
        for k = j+1:N

            multiplier = 1/( (k-j)*c_y_array[1] )

            running_sum = zero(T)
            for l = 1:k-j
                tmp = l*j -k +j +l

                running_sum += tmp*c_y_array[l+1]*P[j,k-l]
            end


            P[j,k] = multiplier*running_sum

            #println("P[j,k] = ", P[j,k])
            #@assert 55==4
        end
    end

    return P
end


## economic versions.

function computeTaylorinversederivativeswfactorial!(c_x_array::Vector{T},
                                                    P::Matrix{T},
                                                    c_y_array::Vector{T},
                                                    N::Int)::Nothing where T <: Real
    #
    computerTaylorPwfactorial!(P, N, c_y_array)

    computealldⁿxwfactorial!(c_x_array, N, P, c_y_array[1])

    return nothing
end


function computealldⁿxwfactorial!( c_x_array::Vector{T},
                        N::Int,
                        P::Matrix{T},
                        dy::T)::Nothing where T

    # allocate.
    #c_x_array = Vector{T}(undef,N)

    # fill in values.
    c_x_array[1] = one(T)/dy

    for n = 2:N

        multiplier = -one(T)/dy^n

        running_sum = zero(T)
        for j = 1:n-1
            running_sum += c_x_array[j]*P[j,n]
        end

        c_x_array[n] = multiplier*running_sum
    end

    return nothing
end

# P[j,k] is P_{j,k}.
function computerTaylorPwfactorial!(P::Matrix{T}, N::Int, c_y_array::Vector{T})::Nothing where T <: Real
    @assert size(P) == (N-1,N)

    #P = Matrix{T}(undef,N-1,N)

    # pre-allocate.
    multiplier::T = zero(T)
    running_sum::T = zero(T)

    # update P.
    for j = 1:N-1

        # k == 1
        P[j,j] = c_y_array[1]^j

        # k > 1.
        for k = j+1:N

            multiplier = one(T)/( (k-j)*c_y_array[1] )

            running_sum = zero(T)
            for l = 1:k-j
                tmp = l*j -k +j +l

                running_sum += tmp*c_y_array[l+1]*P[j,k-l]
            end


            P[j,k] = multiplier*running_sum
        end
    end

    return nothing
end

#### estimate radius of convergence.

# the linear regression is taken from the least-squares method here:
#   https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
# 𝑐 is the array of Taylor coefficients.
# returned solution is for the line β*x + β0.
# the regression is fitted against {𝑛_reciprocal, 𝑏}
function estimateROCMRLS(𝑐::Vector{T}, st::Int = 3) where T <: Real
    @assert 3 <= st < length(𝑐)

    fin = length(𝑐)-1
    𝑏_sq = collect( abs( (𝑐[n+1]*𝑐[n-1]-𝑐[n]^2)/(𝑐[n]*𝑐[n-2]-𝑐[n-1]^2) ) for n = st:fin )
    𝑏 = sqrt.(𝑏_sq)
    𝑛_reciprocal = collect( one(Float64)/n for n = st:fin )

    M = length(𝑏)

    𝑛_reciprocal_bar = sum(𝑛_reciprocal)/M
    𝑏_bar = sum(𝑏)/M

    β = sum( (𝑛_reciprocal[i]-𝑛_reciprocal_bar)*(𝑏[i]-𝑏_bar) for i = 1:M )/sum( (𝑛_reciprocal[i]-𝑛_reciprocal_bar)^2 for i = 1:M )
    β0 = 𝑏_bar -β*𝑛_reciprocal_bar

    return 𝑛_reciprocal, 𝑏, β, β0
end

# D-dimensional regression from N samples.
# Model: f(x) = sum( β[j]*ϕ[j](x) for j = 1:D ).
# Set A[i,j] = ϕ[j](X[i]), where X[i] is the input to the i-th sample, i ∈ [N].
function fitLS(y::Vector{T}, X, ϕ_array::Vector{Function})::Vector{T} where T <: Real
    N = length(y)
    M = length(ϕ_array)

    A = Matrix{T}(undef, N, M)
    for j = 1:M
        for i = 1:N
            A[i,j] = ϕ_array[j](X[i])
        end
    end

    β = (A'*A)\A'*y
    return β
end

function fitNNLS(y::Vector{T}, X, ϕ_array::Vector{Function})::Vector{T} where T <: Real
    N = length(y)
    M = length(ϕ_array)

    A = Matrix{T}(undef, N,M)
    for j = 1:M
        for i = 1:N
            A[i,j] = ϕ_array[j](X[i])
        end
    end

    β = NNLS.nnls(A, y)
    return β
end

function identityfunc(x::T) where T
    return x::T
end

# no affine offset.
function fit1Dlinearregression(y::Vector{T}, X)::T where T <: Real
    ϕ_array = Vector{Function}(undef,1)
    ϕ_array[1] = identityfunc

    return fitLS(y, X, ϕ_array)[1]
end

function estimateROCCHregression(c::Vector{T}, st::Int = 5) where T <: Real
    N = length(c)
    @assert st < length(c)

    y = collect( log(abs(c[n])) for n = st:N )
    x = collect( convert(T,n) for n = st:N )

    β = fit1Dlinearregression(y,x)

    return exp(-β)
end

function estimateROCMR(c::Vector{T}) where T <: Real
    @assert length(c) > 3

    r = Vector{T}(undef, length(c)-3)
    i = 1
    for k = 2:length(c)-2

        numerator = c[k+1]*c[k-1] - c[k]^2
        denominator = c[k+2]*c[k] - c[k+1]^2

        r[i] = sqrt(abs(numerator/denominator))
        i += 1
    end

    return r
end


function estimateROCCH(c::Vector{T}) where T <: Real
    return collect( one(T)/(abs(c[n])^(1/n)) for n = 1:length(c))
end

#### fit quantile.

function evalTaylorinverse1storder( y::T,
                                    y0::T,
                                    x0::T,
                                    df_x0::T) where T <: Real

    # prepare derivatives functions.
    N = 1
    dⁿy_array = [ df_x0 ]

    # compute derivatives of the inverse.
    dⁿx_array, P = computeTaylorinversederivatives(dⁿy_array, x0, N)

    ξ = y-y0

    x_eval = sum( dⁿx_array[n]*ξ^n/factorial(n) for n = 1:N ) +x0

    return x_eval
end

### Taylor inverse.

function getTaylorapproxofinverserefinement(    y::T,
                                                v,
                                                y0,
                                                x0,
                                                d_f1_RKHS2_AN::Function,
                                                Z1_v) where T <: Real

    # prepare derivatives functions.
    N = 1
    dF_x0 = evalCDFviaHCubature(x0, d_f1_RKHS2_AN)/Z1_v
    dⁿy_array = [ dF_x0 ]

    # compute derivatives of the inverse.
    dⁿx_array, P = computeTaylorinversederivatives(dⁿy_array, x0, N)

    ξ = y-y0

    x_eval = sum( dⁿx_array[n]*ξ^n/factorial(n) for n = 1:N ) +x0

    return x_eval
end
