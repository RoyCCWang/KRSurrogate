# dir_var is positive for forward traverse.
function computeTaylorinverse(  x0::T,
                                y0::T,
                                update𝑐::Function,
                                update𝑑::Function,
                                𝑓::Function,
                                F::Function,
                                𝑐::Vector{T},
                                𝑑::Vector{T},
                                𝑓_x0::T,
                                err_tol::T,
                                dir_var::T,
                                ϵ::T,
                                n_limit::Int,
                                n0::Int)::Tuple{T,T} where T <: Real
    update𝑐(x0, dir_var)
    update𝑑(x0)

    ## Get x_next: Determine how large of a step to take.
    status_flag, ξ_limit = getsafeTaylorstepquantile(𝑓, 𝑐, 𝑑, x0, 𝑓_x0, y0,
                                err_tol, dir_var, n_limit, n0)

    # find out if y_target is within the operating range.
    y_limit = evalTaylorapprox(ξ_limit, 𝑐, y0)
    ϵ_limit = y_limit - y0 #ϵ_limit = y - y_limit
    @assert dir_var*sign(ϵ_limit) >= 0 # potential problem.

    # allocate for scope reasons.
    x_next = NaN
    y_next = NaN

    if abs(ϵ) < abs(ϵ_limit)
        x_next = evalTaylorapprox(ϵ, 𝑑, x0)
    else
        x_next = evalTaylorapprox(ϵ_limit, 𝑑, x0)
    end

    ## Get y_next: if not confident about inverse Taylor's estimate, use numerical integration.
    if status_flag
        y_next = evalTaylorapprox(x_next - x0, 𝑐, y0)
    else
        y_next = F(x_next)
    end

    return x_next, y_next
end
