


### x_ranges is for all dimensions.
function getYcurrent(   x_ranges,
                        d_select::Int,
                        Y_prev::Vector{T})::Vector{T} where T
    #
    @assert d_select < length(x_ranges)

    x_ranges_current = x_ranges[1:d_select]

    x_ranges_previous = x_ranges[1:d_select+1]
    sz_prev = collect( length(x_ranges_previous[i]) for i = 1:length(x_ranges_previous) )
    Y_nD_prev = reshape(Y_prev, sz_prev...)

    # get warp map sampling positions.
    h_ranges::Vector{LinRange{T}} = Vector{LinRange{T}}(undef, d_select+1)
    for i = 1:d_select
     h_ranges[i] = x_ranges_current[i]
    end
    h_ranges[end] = x_ranges_previous[end]
    Xq_nD = Utilities.ranges2collection(h_ranges, Val(d_select+1))

    # get warp map values.
    h_itp, d_h_itp_unused,
     d2_h_itp_unused = Utilities.setupcubicitp(Y_nD_prev, x_ranges_previous, 1.0)

    H_nD = h_itp.(Xq_nD)
    tmp = sum(H_nD, dims = d_select+1)
    y = vec(tmp ./ maximum(tmp))

    return y
end

# f is the unnormalized marginal density function.
# Y_X is an approximation of f_X via discrete marginalization. Here, f is f_joint.
# for the d ∈ [D-1] dimensions.
function fitproxydensity(  fit_optim_config,
                            f_X::Vector{T},
                            X::Vector{Vector{T}},
                            xd_ranges,
                            d_select::Int,
                            y_prev::Vector{T},
                            max_iters_RKHS::Int,
                            a::T,
                            σ²::T,
                            amplification_factor::T,
                            N_bands::Int,
                            attenuation_factor_at_cut_off::T,
                            zero_tol_RKHS::T,
                            prune_tol::T,
                            max_integral_evals::Int)::Tuple{Vector{T},
                                Vector{Vector{T}},
                                RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}},
                                Function,
                                Function,
                                Function,
                                Function} where T <: Real

    println("Working on d = ", d_select) #  debug.


    sz_current = collect( length(xd_ranges[i]) for i = 1:length(xd_ranges) )
    y_nD = reshape(y_prev, sz_current...)

    # fit density.
    c_out, 𝓧, θ_a, dϕ, d2ϕ,
        keep_indicators = fitRKHSRegularizationdensity(  fit_optim_config, f_X,
                                                    X,
                                                    xd_ranges,
                                                    y_nD,
                                                    max_iters_RKHS,
                                                    a,
                                                    σ²,
                                                    amplification_factor,
                                                    N_bands,
                                                    attenuation_factor_at_cut_off,
                                                    zero_tol_RKHS,
                                                    prune_tol)

    # in case dϕ or d2ϕ has discontinuities. We defer this feature for future.
    dϕ_w_dir = (xx, dir_varr, knot_zero_toll)->dϕ(xx)[end]
    d2ϕ_w_dir = (xx, dir_varr, knot_zero_toll)->d2ϕ(xx)[end]


    return c_out, 𝓧, θ_a, dϕ_w_dir, d2ϕ_w_dir, dϕ, d2ϕ
end

# for the D-th dimension.
function fitproxydensity(  fit_optim_config,
                            f_X::Vector{T},
                            X::Vector{Vector{T}},
                            x_ranges,
                            f::Function,
                            max_iters_RKHS::Int,
                            a::T,
                            σ²::T,
                            amplification_factor::T,
                            N_bands::Int,
                            attenuation_factor_at_cut_off::T,
                            zero_tol_RKHS::T,
                            prune_tol::T,
                            max_integral_evals::Int)::Tuple{Vector{T},
                                Vector{Vector{T}},
                                RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}},
                                Function, Function, Function, Function, Vector{T},
                                BitArray{1}} where T <: Real

    D = length(x_ranges)
    println("Working on d = ", D) #  debug.

    # get warp map sampling positions.
    X_nD = Utilities.ranges2collection(x_ranges, Val(D))

    # get warp map values.
    Y_nD = f.(X_nD)
    Y_nD = Y_nD ./ maximum(Y_nD)

    # fit density.
    c_out, 𝓧, θ_a, dϕ, d2ϕ,
        keep_indicators = fitRKHSRegularizationdensity(  fit_optim_config, f_X,
                                                    X,
                                                    x_ranges,
                                                    Y_nD,
                                                    max_iters_RKHS,
                                                    a,
                                                    σ²,
                                                    amplification_factor,
                                                    N_bands,
                                                    attenuation_factor_at_cut_off,
                                                    zero_tol_RKHS,
                                                    prune_tol)

    # in case dϕ or d2ϕ has discontinuities. We defer this feature for future.
    dϕ_w_dir = (xx, dir_varr, knot_zero_toll)->dϕ(xx)[end]
    d2ϕ_w_dir = (xx, dir_varr, knot_zero_toll)->d2ϕ(xx)[end]

    return c_out, 𝓧, θ_a, dϕ_w_dir, d2ϕ_w_dir,
            dϕ, d2ϕ, vec(Y_nD), keep_indicators
end

function fitmarginalsadaptive( fit_optim_config::FitDensityConfigType,
                            f_X_array::Vector{Vector{T}},
                            X_array::Vector{Vector{Vector{T}}},
                            x_ranges,
                            f::Function,
                            max_iters_RKHS::Int,
                            a_array::Vector{T},
                            σ_array::Vector{T},
                            amplification_factor::T,
                            N_bands::Int,
                            attenuation_factor_at_cut_off::T,
                            zero_tol_RKHS::T,
                            prune_tol::T,
                            max_integral_evals::Int )::Tuple{Vector{Vector{T}},
                            Vector{Vector{Vector{T}}},
                            Vector{RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}}},
                            Vector{Function},
                            Vector{Function},
                            Vector{Function},
                            Vector{Function},
                            Vector{Vector{T}}} where T <: Real

    #
    D = length(x_ranges)

    for d = 1:D
        @assert length(X_array[d]) > 1
        # if assert fails, it means not enough fitting locations for this dimension.
        # to do: handle this gracefully.
    end

    # allocate output.
    c_array = Vector{Vector{T}}(undef,D)
    θ_array = Vector{RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}}}(undef,D)
    𝓧_array = Vector{Vector{Vector{T}}}(undef,D)
    dφ_array = Vector{Function}(undef,D)
    d2φ_array = Vector{Function}(undef,D)
    dϕ_array = Vector{Function}(undef,D)
    d2ϕ_array = Vector{Function}(undef,D)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))
    y_array = Vector{Vector{T}}(undef,D)

    𝑋 = X_array[D]
    f_𝑋 = f_X_array[D]
    a = a_array[D]
    σ² = σ_array[D]^2

    c_array[D], 𝓧_array[D], θ_array[D],
        dφ_array[D], d2φ_array[D],
        dϕ_array[D], d2ϕ_array[D],
        y_array[D],
        keep_indicators_unused = fitproxydensity(  fit_optim_config, f_𝑋,
                                𝑋, x_ranges,
                                f, max_iters_RKHS, a, σ²,
                                amplification_factor, N_bands, attenuation_factor_at_cut_off,
                                zero_tol_RKHS,
                                prune_tol,
                                max_integral_evals)

    for d = (D-1):-1:1
        𝑋 = X_array[d]
        f_𝑋 = f_X_array[d]
        a = a_array[d]
        σ² = σ_array[d]^2

        y_array[d] = getYcurrent(x_ranges, d, y_array[d+1])

        c_array[d], 𝓧_array[d], θ_array[d],
            dφ_array[d], d2φ_array[d],
            dϕ_array[d], d2ϕ_array[d] = fitproxydensity(  fit_optim_config, f_𝑋,
                                    𝑋, x_ranges[1:d], d,
                                    y_array[d], max_iters_RKHS, a, σ²,
                                    amplification_factor, N_bands, attenuation_factor_at_cut_off,
                                    zero_tol_RKHS,
                                    prune_tol,
                                    max_integral_evals)

    end

    return c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array, dϕ_array, d2ϕ_array,
            y_array
end
