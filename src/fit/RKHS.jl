# used by kernel centers selection algorithms.
function fitRKHSdensity(  f_X::Vector{T},
                            X::Vector{Vector{T}},
                            σ²::T,
                            θ::KT,
                            config::SDPConfigType{T},
                            prune_tol::T)::Tuple{Vector{T},
                                Vector{Vector{T}},
                                BitArray{1}} where {T,KT}


    η, α_solution = RKHSRegularization.fitnDdensity(vec(f_X), vec(X), σ², θ,
                                        config.zero_tol, config.max_iters)


    #
    # ## debug: use brute-force optim.
    # K = RKHSRegularization.constructkernelmatrix(X, θ)
    # s, Q = eigen(K)
    # println("s = ")
    # display(s)
    # println()
    #
    # println("K = ")
    # display(K)
    #
    # #
    # η, α_solution, _ = RKHSRegularization.fitdensityoptim( vec(f_X), vec(X), σ², θ;
    #                                     barrier_factor = 20000.0,
    #                                     barrier_scale = 0.01,
    #                                     max_iters = 1000,
    #                                     initial_solve_iters = 1000,
    #                                     show_trace = false,
    #                                     store_trace = false,
    #                                     zero_tol = prune_tol )

    # prune kernels that have a negligible coefficient magnitude.
    keep_indicators = η.c .> prune_tol

    c_out = Vector{T}(undef,0)
    𝓧 = Vector{Vector{T}}(undef,0)
    if sum(keep_indicators) == 0
        c_out = copy(η.c)
        𝓧 = copy(η.X)
    else
        c_out = η.c[keep_indicators]
        𝓧 = η.X[keep_indicators]
    end

    return c_out, 𝓧, keep_indicators
end

function fitRKHSdensity(  f_X::Vector{T},
                            X::Vector{Vector{T}},
                            σ²::T,
                            θ::KT,
                            config::ROptimConfigType{T},
                            prune_tol::T)::Tuple{Vector{T},
                                Vector{Vector{T}},
                                BitArray{1}} where {T,KT}

    #
    #α_initial = ones(T, length(y)),     ## initial guess.

    η, α_solution = RKHSRegularization.fitnDdensityRiemannian( vec(f_X),
                            vec(X), σ², θ, config.zero_tol, config.max_iters;
                            verbose_flag = config.verbose_flag,
                            max_iter_tCG = config.max_iter_tCG,
                            ρ_lower_acceptance = config.ρ_lower_acceptance,
                            ρ_upper_acceptance = config.ρ_upper_acceptance,
                            minimum_TR_radius = config.minimum_TR_radius,
                            maximum_TR_radius = config.maximum_TR_radius,
                            norm_df_tol = config.norm_df_tol,
                            objective_tol = config.objective_tol,
                            avg_Δf_tol = config.avg_Δf_tol,
                            avg_Δf_window = config.avg_Δf_window,
                            max_idle_update_count = config.max_idle_update_count,
                            g = config.g,
                            𝑟 = config.𝑟 )

    # prune kernels that have a negligible coefficient magnitude.
    keep_indicators = η.c .> prune_tol

    c_out = Vector{T}(undef,0)
    𝓧 = Vector{Vector{T}}(undef,0)
    if sum(keep_indicators) == 0
        c_out = copy(η.c)
        𝓧 = copy(η.X)
    else
        c_out = η.c[keep_indicators]
        𝓧 = η.X[keep_indicators]
    end

    return c_out, 𝓧, keep_indicators
end

function fitRKHSRegularizationdensity(  fit_optim_config::FitDensityConfigType,
                                f_X::Vector{T},
                                X::Vector{Vector{T}},
                                x_ranges,
                                Y_nD,
                                max_iters_RKHS::Int,
                                a::T,
                                σ²::T,
                                amplification_factor::T,
                                N_bands::Int,
                                attenuation_factor_at_cut_off::T,
                                zero_tol_RKHS::T,
                                prune_tol::T) where T <: Real

    # construct warp map.
    ϕ, dϕ, d2ϕ = preparewarpmap( N_bands, attenuation_factor_at_cut_off, Y_nD, x_ranges, amplification_factor)

    # make adaptive kernel.
    θ_canonical = RKHSRegularization.RationalQuadraticKernelType(a)
    θ_a = RKHSRegularization.AdaptiveKernelType(θ_canonical, ϕ)

    # fit.
    c, 𝓧, keep_indicators = fitRKHSdensity( f_X,
                                            X,
                                            σ²,
                                            θ_a,
                                            fit_optim_config,
                                            prune_tol )

    return c, 𝓧, θ_a, dϕ, d2ϕ, keep_indicators
end

function preparewarpmap(N_bands::Int,
                        attenuation_factor_at_cut_off::T,
                        Y_nD::Array{T,D},
                        x_ranges,
                        amplification_factor::T) where {T,D}

    # warp map parameter set up.
    reciprocal_cut_off_percentages = ones(N_bands) ./collect(LinRange(1.0,0.2,N_bands))
    ω_set = collect( π/(reciprocal_cut_off_percentages[i]*sqrt(2*log(attenuation_factor_at_cut_off))) for i = 1:length(reciprocal_cut_off_percentages) )
    pass_band_factor = abs(ω_set[1]-ω_set[2])*0.2

    # construct warp map.
    ϕ_nD = RKHSRegularization.getRieszwarpmapsamples(Y_nD, Val(:simple), Val(:uniform), ω_set, pass_band_factor)
    ϕ, dϕ, d2_ϕ = Utilities.setupcubicitp(ϕ_nD, x_ranges, amplification_factor)

    return ϕ, dϕ, d2_ϕ
end
