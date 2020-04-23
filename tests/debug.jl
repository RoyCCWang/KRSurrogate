import Random
import PyPlot
import BSON

using LinearAlgebra

PyPlot.close("all")
fig_num = 1

Random.seed!(25)


# data = BSON.load("debug.bson")
# L = data[:L_full]
#
# println("typeof(L) = ", typeof(L))
# println()
#
# isposdef(L) # takes a while if in @everywhere?
# eigen(L) # takes a while

include("./example_helpers/test_densities.jl")


τ = 1e-2
N_array = [100; 100]
D = length(N_array)
h, x_range = generaterandombetacopula( τ, N_array)

x0 = rand(D)
h(x0)

# TODO develop visualize pair-wse marginals.

z_min = copy(limit_a)
deleteat!(z_min, sort([i_select; j_select]))

z_max = copy(limit_b)
deleteat!(z_max, sort([i_select; j_select]))


evalpairwisemarginal(h, z_min, z_max, i_select, j_select, xi, xj, D)

# evaluate the marginal for dimensions i and j.
function evalpairwisemarginal(  f::Function,
                                z_min::Vector{T},
                                z_max::Vector{T},
                                i::Int,
                                j::Int,
                                xi,
                                xj,
                                D::Int;
                                initial_divisions::Int = 1,
                                max_integral_evals::Int = 100000 )::T where T
    #
    @assert length(z_min) == D-2 == length(z_max)

    h = zz->f( substitutedim(zz, xi, xj, i, j) )
    (val, err) = HCubature.hcubature(h, z_min, z_max;
                    norm = LinearAlgebra.norm, rtol = sqrt(eps(T)),
                    atol = 0,
                    maxevals = max_integral_evals,
                    initdiv = initial_divisions)
    return val
end

function substitutedim(x::AbstractVector{T}, xi, xj, i, j)::Vector{T} where T
    out = copy(x)

    out[i] = xi
    out[j] = xj

    return out
end
