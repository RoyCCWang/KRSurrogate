
"""
g_array[d] contains the function g(x_{1:d}).
"""
function evalKRdensityeval!( Z::Vector{T},
                            g_array::Vector{FT},
                            x::Vector{T},
                            limit_a::Vector{T},
                            limit_b::Vector{T};
                            initial_divisions::Int = 1,
                            max_integral_evals::Int = 100000)::T where {T,FT}
    #
    D = length(x)
    @assert length(g_array) == D

    for d = 2:D
        h = tt->g_array[d]([x[1:d-1]; tt])
        Z[d] = evalintegral(h, limit_a[d], limit_b[d];
                        initial_divisions = initial_divisions,
                        max_integral_evals = max_integral_evals)
    end

    ln_out = sum( log(g_array[d](x[1:d])) - log(Z[d]) for d = 1:D )
    out = exp(ln_out)

    return out
end

function setupKRdensityeval(  g_array::Vector{FT},
                                limit_a::Vector{T},
                                limit_b::Vector{T};
                                initial_divisions::Int = 1,
                                max_integral_evals::Int = 100000)::Function where {T<: Real,FT}
    #
    Z_persist = Vector{T}(undef, length(g_array))

    h = tt->g_array[1]([tt])
    Z_persist[1] = evalintegral(h, limit_a[1], limit_b[1];
                    initial_divisions = initial_divisions,
                    max_integral_evals = max_integral_evals)
    #
    pdf = xx->evalKRdensityeval!( Z_persist, g_array, xx, limit_a, limit_b;
                               initial_divisions = initial_divisions,
                               max_integral_evals = max_integral_evals)
    #
    return pdf
end
