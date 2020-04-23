

function getkernelcenters(θ_canonical::KT,
                            Y_nD::Array{T,D},
                            x_ranges,
                            X_ref::Vector{Vector{T}},
                            f::Function,
                            limit_a::Vector{T},
                            limit_b::Vector{T},
                            config::KernelCenterConfigType{T},
                            fit_optim_config) where {T,D,KT}

    θ_a, ϕ_map_func, d_ϕ_map_func,
        d2_ϕ_map_func = setupadaptivekernel(θ_canonical,
                                            Y_nD,
                                            f,
                                            x_ranges;
                                            amplification_factor = config.amplification_factor,
                                            attenuation_factor_at_cut_off = config.attenuation_factor_at_cut_off,
                                            N_bands = config.N_bands)
    #

    println("config.N_preliminary_candidates = ", config.N_preliminary_candidates)


    # get candidates.
    X_candidates, y_unused = getinitialcandidatesviauniform(   config.N_preliminary_candidates,
                                limit_a,
                                limit_b,
                                f;
                                zero_tol = config.candidate_zero_tol,
                                truncation_factor = config.candidate_truncation_factor)

    # debug. store initial pool of fit locations.
    X_pool = copy(X_candidates)

    # first round of fit location selection.
    f_scale_factor::T = 1/maximum(Y_nD)
    DPP_base = θ_a
    additive_function = xx->f_scale_factor*f(xx)

    c1, X1,
        X_fit1 = selectkernelcenters!(  X_candidates,
                            additive_function,
                            f,
                            DPP_base,
                            θ_a,
                            fit_optim_config;
                            #max_iters_RKHS = config.max_iters_RKHS,
                            base_gain = config.base_gain,
                            kDPP_zero_tol = config.kDPP_zero_tol,
                            N_kDPP_draws = config.N_kDPP_draws,
                            N_kDPP_per_draw  = config.N_kDPP_per_draw ,
                            #zero_tol_RKHS = config.zero_tol_RKHS,
                            prune_tol = config.prune_tol,
                            σ² = config.σ²)
#@assert 4==45
    #
    fq1 = xx->RKHSRegularization.evalquery(xx, c1, X1, θ_a)
    f_X_ref = f.(X_ref)
    err_1 = norm(fq1.(X_ref)-f_X_ref)

    ## subsequent refinement.
    ##### refine via sequential.

    #X0 = copy(X1)
    X0 = copy(X_fit1)

    # println("length(X_candidates) = ", length(X_candidates))
    # println("length(X1) = ", length(X1))
    # println("length(X_pool) = ", length(X_pool))
    # @assert 1==2

    println("refining.")
    @time c_history, X_history,
        error_history,
        X_fit_history = refinecenters!( X_candidates,
                                X0,
                                config.N_refinements,
                                f,
                                fq1,
                                θ_a,
                                X_ref,
                                fit_optim_config;
                                #zero_tol_RKHS = config.zero_tol_RKHS,
                                prune_tol = config.prune_tol,
                                close_radius_tol = config.close_radius_tol,
                                #max_iters_RKHS = config.max_iters_RKHS,
                                σ² = config.σ²)
    println()

    # add initial results.

    insert!(c_history, 1, c1)
    insert!(X_history, 1, X1)
    insert!(error_history, 1, err_1)
    insert!(X_fit_history, 1, X_fit1)

    return θ_a, c_history, X_history, error_history, X_fit_history, X_pool
end



function getkernelcentersviaadaptivekernel(  canonical_a::T, # larger means larger bandwidth.)
                                            x_ranges,
                                            limit_a::Vector{T},
                                            limit_b::Vector{T};
                                            N_preliminary_candidates = 100,
                                            N_kDPP_draws = 1,
                                            N_kDPP_per_draw = 20,
                                            close_radius_tol = 1e-5,
                                            base_gain = 1.0, #1/100 # the higher, the more selection among edges of f
                                            N_refinements = 10,
                                            amplification_factor = 5.0) where T
    # set up.
    D = length(limit_a)
    @assert length(limit_b) == length(limit_a) == D

    ### get fit locations.

    zero_tol_RKHS = 1e-13
    prune_tol = 1.1*zero_tol_RKHS
    max_iters_RKHS = 10000
    σ_array = sqrt(1e-1) .* ones(Float64, D)
    max_integral_evals = 10000 #500 #typemax(Int)

    attenuation_factor_at_cut_off = 2.0
    N_bands = 5

    # this setting is for [0,1]^D  copula.
    #N_preliminary_candidates = 100^D #10000 enough for 2D, not enough for 3D.
    candidate_truncation_factor = 0.01 #0.005
    candidate_zero_tol = 1e-12

    kDPP_zero_tol = 1e-12

    initial_divisions = 1

    center_config = KernelCenterConfigType( amplification_factor,
                    attenuation_factor_at_cut_off,
                    N_bands,
                    N_preliminary_candidates,
                    candidate_truncation_factor,
                    candidate_zero_tol,
                    base_gain,
                    kDPP_zero_tol,
                    N_kDPP_draws,
                    N_kDPP_per_draw,
                    prune_tol,
                    close_radius_tol,
                    N_refinements,
                    #σ²,
                    σ_array[end]^2,
                    initial_divisions)

    #
    θ_canonical = RKHSRegularization.RationalQuadraticKernelType(canonical_a)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))
    X_ref = vec(X_nD)

    Y_nD = f.(X_nD)

    fit_optim_config = ROptimConfigType(zero_tol_RKHS, max_iters_RKHS)


    θ_a, c_history, X_history, error_history,
        X_fit_history, X_pool = getkernelcenters(   θ_canonical,
                            Y_nD,
                            x_ranges,
                            X_ref,
                            f,
                            limit_a,
                            limit_b,
                            center_config,
                            fit_optim_config)
    #
    min_error, min_ind = findmin(error_history)
    𝑋 = X_history[min_ind]

    return 𝑋
end

## TODO I am here. do this for non-adaptive.
function getkernelcentersnonadaptive(  canonical_a::T, # larger means larger bandwidth.)
                                            x_ranges,
                                            limit_a::Vector{T},
                                            limit_b::Vector{T};
                                            N_preliminary_candidates = 100,
                                            N_kDPP_draws = 1,
                                            N_kDPP_per_draw = 20,
                                            close_radius_tol = 1e-5,
                                            base_gain = 1.0, #1/100 # the higher, the more selection among edges of f
                                            N_refinements = 10,
                                            amplification_factor = 5.0) where T
    # set up.
    D = length(limit_a)
    @assert length(limit_b) == length(limit_a) == D

    ### get fit locations.

    zero_tol_RKHS = 1e-13
    prune_tol = 1.1*zero_tol_RKHS
    max_iters_RKHS = 10000
    σ_array = sqrt(1e-1) .* ones(Float64, D)
    max_integral_evals = 10000 #500 #typemax(Int)

    attenuation_factor_at_cut_off = 2.0
    N_bands = 5

    # this setting is for [0,1]^D  copula.
    #N_preliminary_candidates = 100^D #10000 enough for 2D, not enough for 3D.
    candidate_truncation_factor = 0.01 #0.005
    candidate_zero_tol = 1e-12

    kDPP_zero_tol = 1e-12

    initial_divisions = 1

    center_config = KernelCenterConfigType( amplification_factor,
                    attenuation_factor_at_cut_off,
                    N_bands,
                    N_preliminary_candidates,
                    candidate_truncation_factor,
                    candidate_zero_tol,
                    base_gain,
                    kDPP_zero_tol,
                    N_kDPP_draws,
                    N_kDPP_per_draw,
                    prune_tol,
                    close_radius_tol,
                    N_refinements,
                    #σ²,
                    σ_array[end]^2,
                    initial_divisions)

    #
    θ_canonical = RKHSRegularization.RationalQuadraticKernelType(canonical_a)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))
    X_ref = vec(X_nD)

    Y_nD = f.(X_nD)

    fit_optim_config = ROptimConfigType(zero_tol_RKHS, max_iters_RKHS)


    θ_a, c_history, X_history, error_history,
        X_fit_history, X_pool = getkernelcenters(   θ_canonical,
                            Y_nD,
                            x_ranges,
                            X_ref,
                            f,
                            limit_a,
                            limit_b,
                            center_config,
                            fit_optim_config)
    #
    min_error, min_ind = findmin(error_history)
    𝑋 = X_history[min_ind]

    return 𝑋
end
