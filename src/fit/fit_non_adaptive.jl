
# make all routines here non-adaptive kernel. including kernel selection (later).

function fitskiplastnonadaptive( f_X_array::Vector{Vector{T}},
                      X_array::Vector{Vector{Vector{T}}},
                      max_iters_RKHS::Int,
                      a_array::Vector{T},
                      σ_array::Vector{T},
                      fit_optim_config,
                      prune_tol::T )::Tuple{Vector{Vector{T}},
                      Vector{Vector{Vector{T}}},
                      Vector{RKHSRegularization.RationalQuadraticKernelType{T}} } where T <: Real

    #
    D = length(f_X_array)
    @assert length(σ_array) == length(a_array) == D

    # allocate output.
    c_array = Vector{Vector{T}}(undef,D)
    θ_array = Vector{RKHSRegularization.RationalQuadraticKernelType{T}}(undef,D)
    𝓧_array = Vector{Vector{Vector{T}}}(undef,D)

    X = X_array[D]
    f_X = f_X_array[D]
    θ_array[D] = RKHSRegularization.RationalQuadraticKernelType(a_array[D])
    σ² = σ_array[D]^2

    c_array[D], 𝓧_array[D], unused = fitRKHSdensity(  f_X,
                                                    X,
                                                    σ²,
                                                    θ_array[D],
                                                    fit_optim_config,
                                                    prune_tol )

    for d = (D-1):-1:1

        println("Working on d = ", d) #  debug.

        X = X_array[d]
        f_X = f_X_array[d]
        θ_array[d] = RKHSRegularization.RationalQuadraticKernelType(a_array[d])
        σ² = σ_array[d]^2

        c_array[d], 𝓧_array[d], unused = fitRKHSdensity(  f_X,
                                                    X, σ²,
                                                    θ_array[d],
                                                    fit_optim_config,
                                                    prune_tol )
    end

    return c_array, 𝓧_array, θ_array
end
