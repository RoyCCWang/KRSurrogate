# integrates out the last D-length(x) number of variables.
# integration domain is -∞ to ∞.
# f: ℝ^{D} → ℝ,
function evaljointpdf(  x::Vector{T},
                        f::Function,
                        D::Int;
                        max_integral_evals = 1000000,
                        initial_divisions = 1) where T <: Real
    # check.
    d = length(x)
    @assert 0 < length(x)

    # pre-allocate.
    if length(x) < D
        x_min = -ones(T, D-d)
        x_max = ones(T, D-d)

        h = getMDZintegrand(f, x, D)
        (val, err) = HCubature.hcubature(h, x_min, x_max;
                        norm = LinearAlgebra.norm, rtol = sqrt(eps(T)), atol = 0,
                        maxevals = max_integral_evals,
                        initdiv = initial_divisions)

        return val, err
    end

    return f(x), zero(T)
end




# compute the normalizing constant on the last D-length(X) dimensions.
# for use with HCubature, so vector-valued input.
function getMDZintegrand(f::Function, x::Vector{T}, D::Int) where T <: Real
    @assert length(x) < D
    d = length(x)
    M = D-d

    return tt->f( [x; collect(tt[d]/(1-tt[d]^2) for d = 1:M)] )*
                    prod( (1+tt[d]^2)/(1-tt[d]^2)^2 for d = 1:M )
end

function evalcdfviaHCubature(f::Function, max_integral_evals::Int, limits_a::T, limits_b::T) where T <: Real

    val_NI, err_NI = HCubature.hcubature(f, [limits_a], [limits_b];
                                        norm=norm, rtol=sqrt(eps(T)),
                                        atol=0, maxevals=max_integral_evals, initdiv=1)

    return val_NI, err_NI
end


function evaljointpdfcompact(  x::Vector{T},
                                f::Function,
                                limits_a::Vector{T},
                                limits_b::Vector{T};
                                max_integral_evals = 1000000,
                                initial_divisions = 1) where T <: Real

    # check.
    d = length(x)
    D = length(limits_a)
    @assert 0 < d <= D

    if length(x) < D

        x_min = limits_a[d+1:D]
        x_max = limits_b[d+1:D]

        h = xx->f( [x; xx] )
        (val, err) = HCubature.hcubature(h, x_min, x_max;
                        norm = LinearAlgebra.norm, rtol = sqrt(eps(T)),
                        atol = 0,
                        maxevals = max_integral_evals,
                        initdiv = initial_divisions)

        return val, err
    end

    return f(x), zero(T)
end

# # f is a normalized density function.
# function evalCDFviaHCubature(  x::T,
#                                 f::Function, max_integral_evals::Int = 10000, ϵ = 1e-17)::T where T <: Real
#     #
#     integrand_func = get1DCDFintegrand(x, f)
#
#     #out = computetsq(x_tanh, w_tanh, integrand_func, m_tsq)
#     out, err_out = HCubature.hcubature(integrand_func, [-one(T)+ϵ], [one(T)-ϵ];
#                                         norm=norm, rtol=sqrt(eps(T)),
#                                         atol=0, maxevals=max_integral_evals, initdiv=1)
#
#     return out
# end

function evalintegral(  f_v::Function,
                    a::T,
                    b::T;
                    initial_divisions::Int = 1,
                    max_integral_evals::Int = 100000 )::T where T
    #
    sol_val, sol_err = HCubature.hquadrature( f_v, a, b;
                                        norm = norm, rtol = sqrt(eps(T)),
                                        atol = 0,
                                        maxevals = max_integral_evals,
                                        initdiv = initial_divisions)
    #
    return sol_val
end

function evalCDFv(  f::Function,
                    x_full::Vector{T},
                    lower_bound::T,
                    upper_bound::T;
                    zero_tol::T = eps(T)*2,
                    max_tol::T = 1e100,
                    max_integral_evals::Int = 1000000,
                    initial_divisions::Int = 1)::T where T
    #
    v = x_full[1:end-1]
    x = x_full[end]

    f_v = xx->f([v; xx])
    Z = clamp( evalintegral(f_v, lower_bound, upper_bound;
                            max_integral_evals = max_integral_evals,
                            initial_divisions = initial_divisions),
               zero_tol, max_tol )
    # to do: error checking here.

    numerator = evalintegral( f_v, lower_bound, x;
                    max_integral_evals = max_integral_evals,
                    initial_divisions = initial_divisions)

    return clamp(numerator/Z, zero(T), one(T))
end
