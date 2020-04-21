function getTaylorcoeffsofdensity(𝑐::Vector{T})::Vector{T} where T <: Real
    c_𝑓 = Vector{T}(undef, length(𝑐)-1)
    getTaylorcoeffsofdensity!(c_𝑓, 𝑐)

    return c_𝑓
end

function getTaylorcoeffsofdensity!(c_𝑓::Vector{T}, 𝑐::Vector{T})::Nothing where T <: Real
    resize!(c_𝑓, length(𝑐)-1)
    for i = 2:length(𝑐)
        c_𝑓[i-1] = 𝑐[i]*i
    end

    return nothing
end

# assumes ξ is in the operation range of (x0,c,f), where f is the function corresponding to c.
function evalTaylorapprox(ξ::T, c::Vector{T}, y0::T)::T where T <: Real
    return sum( c[n]*(ξ)^n for n = 1:length(c) ) + y0
end

# assumes x is in the operation range of (x0, c, f), where f is the function corresponding to c.
function evalTaylorapprox(x::T, x0::T, c::Vector{T}, y0::T)::T where T <: Real
    ξ = x-x0
    return evalTaylorapprox(ξ, c, y0)
end


##### CDF.

function ξtoerrorcdf(ξ::T,
                            c_𝑓::Vector{T},
                            x0::T,
                            𝑓_x0::T,
                            𝑓::Function)::T where T <: Real

    f_x_ξ = evalTaylorapprox(ξ, c_𝑓, 𝑓_x0)
    𝑓_x_ξ = 𝑓(x0 + ξ)

    return (abs(𝑓_x_ξ - f_x_ξ)/𝑓_x_ξ)
end

function getsafeTaylorstep( 𝑓::Function,
                            𝑐::Vector{T},
                            x0::T,
                            𝑓_x0::T,
                            err_tol::T,
                            sign_multiple::T,
                            n_limit::Int = 10,
                            n0::Int = 5)::Tuple{Bool,T} where T <: Real
    #
    c_𝑓 = getTaylorcoeffsofdensity(𝑐)

    return getsafeTaylorstepcf(𝑓, c_𝑓, x0, 𝑓_x0, err_tol, sign_multiple, n_limit, n0)
end

# we don't check with 𝑓(x0 +ξ) for speed reasons.
# sign_multiple == 1.0 for forward, == -1.0 for backward.
function getsafeTaylorstepcf( 𝑓::Function,
                            c_𝑓::Vector{T},
                            x0::T,
                            𝑓_x0::T,
                            err_tol::T,
                            sign_multiple::T,
                            n_limit::Int = 10,
                            n0::Int = 5)::Tuple{Bool,T} where T <: Real

    @assert n_limit >= n0
    ξ_limit = sign_multiple*minimum(estimateROCMR(c_𝑓))

    # ROC multipliers to try: collect(0.01*1.55^n for n = 1:10)
    # collect(0.01*1.55^n for n = 1:10) .* ξ_limit to see all ξ candidates.
    n = n0
    ξ = (0.01*1.55^n)*ξ_limit
    relative_err = ξtoerrorcdf(ξ, c_𝑓, x0, 𝑓_x0, 𝑓)

    if relative_err > err_tol
        # decrease ξ until abs discrepancy becomes lower than err_tol.

        while relative_err > err_tol && n > 1
            n -= 1

            ξ = (0.01*1.55^n)*ξ_limit
            relative_err = ξtoerrorcdf(ξ, c_𝑓, x0, 𝑓_x0, 𝑓)
        end

        if relative_err > err_tol
            return false, ξ
        else
            return true, ξ
        end

    else
        #println("small")
        # increase ξ until abs discrepancy becomes larger than err_tol.
        while relative_err < err_tol && n < n_limit
            n += 1

            ξ = (0.01*1.55^n)*ξ_limit
            relative_err = ξtoerrorcdf(ξ, c_𝑓, x0, 𝑓_x0, 𝑓)
        end

        return true, (0.01*1.55^(n-1))*ξ_limit
    end

    return true, ξ
end


####### for inverse of CDF.

function ξtorelativeerrquantile(ξ::T,
                                𝑐::Vector{T},
                                c_𝑔::Vector{T},
                                𝑔_y0::T,
                                x0::T,
                                y0::T,
                                𝑓::Function)::T where T <: Real

    x = x0 + ξ
    y = evalTaylorapprox(ξ, 𝑐, y0)
    ϵ = y - y0

    g_y_ϵ = evalTaylorapprox(ϵ, c_𝑔, 𝑔_y0)
    𝑔_y_ϵ = one(T)/𝑓(x)

    return (abs(𝑔_y_ϵ - g_y_ϵ)/𝑔_y_ϵ)
end

# assumes 𝑐 and 𝑑 were updated.
# g is Taylor approx of derivative of F_inverse.
#   𝑔 is its analytic evaluation using Taylor approx of F.
function getsafeTaylorstepquantile( 𝑓::Function,
                                            𝑐::Vector{T},
                                            𝑑::Vector{T},
                                            x0::T,
                                            𝑓_x0::T,
                                            y0::T,
                                            err_tol::T,
                                            sign_multiple::T,
                                            n_limit::Int = 10,
                                            n0::Int = 5)::Tuple{Bool,T} where T <: Real
    #
    c_𝑓 = getTaylorcoeffsofdensity(𝑐)
    status_flag, ξ_limit::T = getsafeTaylorstepcf(𝑓, c_𝑓, x0, 𝑓_x0, err_tol, sign_multiple, n_limit, n0) #
    #ξ_limit = sign_multiple*ξ_limit

    c_𝑔 = getTaylorcoeffsofdensity(𝑑)
    𝑔_y0 = one(T)/𝑓_x0

    # ROC multipliers to try: collect(0.01*1.55^n for n = 1:10)
    # collect(0.01*1.55^n for n = 1:10) .* ξ_limit to see all ξ candidates.
    n = n0
    ξ = (0.01*1.55^n)*ξ_limit
    relative_err = ξtorelativeerrquantile(ξ, 𝑐, c_𝑔, 𝑔_y0, x0, y0, 𝑓)

    if relative_err > err_tol
        # decrease ξ until abs discrepancy becomes lower than err_tol.

        while relative_err > err_tol && n > 1
            n -= 1

            ξ = (0.01*1.55^n)*ξ_limit
            relative_err = ξtorelativeerrquantile(ξ, 𝑐, c_𝑔, 𝑔_y0, x0, y0, 𝑓)
        end

        if relative_err > err_tol
            return false, ξ
        else
            return true, ξ
        end

    else
        #println("small")
        # increase ξ until abs discrepancy becomes larger than err_tol.
        while relative_err < err_tol && n < n_limit
            n += 1

            ξ = (0.01*1.55^n)*ξ_limit
            relative_err = ξtorelativeerrquantile(ξ, 𝑐, c_𝑔, 𝑔_y0, x0, y0, 𝑓)
        end

        return true, (0.01*1.55^(n-1))*ξ_limit
    end

    return true, ξ
end
