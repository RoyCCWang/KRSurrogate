
"""
Fits a copula on [0,1]^D.
D = length(N_array).
N_array is the number of samples for the warp map, per dimension.
τ is the tolerance offset from 0 and 1.
"""
function fitexamplebetacopuladensity(   ::Val{D},
                                        N_array::Vector{Int};
                                        τ::T = convert(Float64, 1e-2) ) where {T,D}
    ## set up.
     @assert D == length(N_array)

    limit_a = ones(T, D) .* τ
    limit_b = ones(T, D) - limit_a

    # oracle probability density.
    𝑝, x_ranges = getmixture2Dbetacopula1(τ, N_array)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))



    # unnormalized probability density.
    𝑝_X_nD = 𝑝.(X_nD)
    f_scale_factor = maximum(𝑝_X_nD)
    f = xx->𝑝(xx)/f_scale_factor

    ### get fit locations.

    zero_tol_RKHS = 1e-13
    prune_tol = 1.1*zero_tol_RKHS
    max_iters_RKHS = 10000
    σ_array = sqrt(1e-3) .* ones(Float64, D)
    initial_divisions = 1
    max_integral_evals = 10000 #500 #typemax(Int)
    amplification_factor = 5.0 #1.0 #50.0
    attenuation_factor_at_cut_off = 2.0
    N_bands = 5

    N_preliminary_candidates = 100^D #10000 enough for 2D, not enough for 3D.
    candidate_truncation_factor = 0.01 #0.005
    candidate_zero_tol = 1e-12

    base_gain = 1.0 #1/100 # the higher, the more selection among edges of f.
    kDPP_zero_tol = 1e-12
    N_kDPP_draws = 1
    N_kDPP_per_draw = 50

    close_radius_tol = 1e-5
    N_refinements = 10 #50

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
                    zero_tol_RKHS,
                    prune_tol,
                    close_radius_tol,
                    N_refinements,
                    max_iters_RKHS,
                    #σ²,
                    σ_array[end]^2,
                    initial_divisions)

    #
    canonical_a = 0.03
    θ_canonical = RKHSRegularization.RationalQuadraticKernelType(canonical_a)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))
    X_ref = vec(X_nD)

    Y_nD = f.(X_nD)

    θ_a, c_history, X_history, error_history,
        X_fit_history, X_pool = getkernelcenters(   θ_canonical,
                            Y_nD,
                            x_ranges,
                            X_ref,
                            f,
                            limit_a,
                            limit_b,
                            center_config)
    #
    min_error, min_ind = findmin(error_history)
    𝑋 = X_history[min_ind]

    #𝑋 = collect( vec(rand(Y_dist, 1)) for n = 1:N_realizations)
    #

    # prepare all marginal joint densities.
    #f_joint = xx->evaljointpdf(xx,f,D)[1]
    # f_joint = xx->evaljointpdf(xx,f,D,
    #                 max_integral_evals = typemax(Int),
    #                 initial_divisions = 3)[1] # higher accuracy.
    f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

    ###### fit KR, adaptive kernels with RQ as canonical kernel.


    a_array = canonical_a .* ones(Float64, D) #[0.7; 0.7] # for each dimension.
    N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.

    X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D )
    removeclosepositionsarray!(X_array, close_radius_tol)
    # to do: auto fuse points with self-adapting close_radius_tol in fit.jl
    #       this is to avoid semidef error for the QP solver.

    f_X_array = collect( f_joint.(X_array[d]) for d = 1:D )


    println("Timing: fitproxydensities")
    @time c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array,
            Y_array = fitmarginalsadaptive(f_X_array, X_array,
                                        x_ranges, f,
                                        max_iters_RKHS, a_array, σ_array,
                                        amplification_factor, N_bands,
                                        attenuation_factor_at_cut_off,
                                        zero_tol_RKHS, prune_tol, max_integral_evals)

    println("Number of kernel centers kept, per dim:")
    println(collect( length(c_array[d]) for d = 1:D))


    ### set up KR config.

    # quantile-related parameters.
    max_integral_evals = 10000
    quantile_iters = 1000
    N_nodes_tanh = 200
    m_tsq = 7
    quantile_err_tol = 1e-3 #0.01  both gave l-2 norm( abs(u-u_rec) ), summed over all dimensions: 0.024788099072235982
    max_traversals = 50
    N_predictive_traversals = 40
    correction_epoch = 10
    quantile_max_iters = 500
    quantile_convergence_zero_tol = 1e-6 #1e-4
    n_limit = 10
    n0 = 5

    KR_config = KRTaylorConfigType( x_ranges,
                                    quantile_iters,
                                    max_integral_evals,
                                    N_nodes_tanh,
                                    m_tsq,
                                    quantile_err_tol,
                                    max_traversals,
                                    N_predictive_traversals,
                                    correction_epoch,
                                    quantile_max_iters,
                                    quantile_convergence_zero_tol,
                                    n_limit,
                                    n0 )

    return c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array,
            Y_array, KR_config, limit_a, limit_b, f
end

function fitexamplebetacopuladensityecon1(   ::Val{D},
                                                N_array::Vector{Int};
                                                τ::T = convert(Float64, 1e-2) ) where {T,D}
    ## set up.
     @assert D == length(N_array)

    limit_a = ones(T, D) .* τ
    limit_b = ones(T, D) - limit_a

    # oracle probability density.
    𝑝, x_ranges = getmixture2Dbetacopula1(τ, N_array)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))



    # unnormalized probability density.
    𝑝_X_nD = 𝑝.(X_nD)
    f_scale_factor = maximum(𝑝_X_nD)
    f = xx->𝑝(xx)/f_scale_factor

    ### get fit locations.

    zero_tol_RKHS = 1e-13
    prune_tol = 1.1*zero_tol_RKHS
    max_iters_RKHS = 10000
    σ_array = sqrt(1e-1) .* ones(Float64, D)
    max_integral_evals = 10000 #500 #typemax(Int)
    amplification_factor = 5.0 #1.0 #50.0
    attenuation_factor_at_cut_off = 2.0
    N_bands = 5

    N_preliminary_candidates = 100^D #10000 enough for 2D, not enough for 3D.
    candidate_truncation_factor = 0.01 #0.005
    candidate_zero_tol = 1e-12

    base_gain = 1.0 #1/100 # the higher, the more selection among edges of f.
    kDPP_zero_tol = 1e-12
    N_kDPP_draws = 1
    N_kDPP_per_draw = 20

    close_radius_tol = 1e-5
    N_refinements = 10 #10
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
                    zero_tol_RKHS,
                    prune_tol,
                    close_radius_tol,
                    N_refinements,
                    max_iters_RKHS,
                    #σ²,
                    σ_array[end]^2,
                    initial_divisions)

    #
    canonical_a = 0.03 # larger means larger bandwidth.
    θ_canonical = RKHSRegularization.RationalQuadraticKernelType(canonical_a)

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))
    X_ref = vec(X_nD)

    Y_nD = f.(X_nD)

    θ_a, c_history, X_history, error_history,
        X_fit_history, X_pool = getkernelcenters(   θ_canonical,
                            Y_nD,
                            x_ranges,
                            X_ref,
                            f,
                            limit_a,
                            limit_b,
                            center_config)
    #
    min_error, min_ind = findmin(error_history)
    𝑋 = X_history[min_ind]

    #𝑋 = collect( vec(rand(Y_dist, 1)) for n = 1:N_realizations)
    #

    # prepare all marginal joint densities.
    #f_joint = xx->evaljointpdf(xx,f,D)[1]
    # f_joint = xx->evaljointpdf(xx,f,D,
    #                 max_integral_evals = typemax(Int),
    #                 initial_divisions = 3)[1] # higher accuracy.
    f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

    ###### fit KR, adaptive kernels with RQ as canonical kernel.


    a_array = canonical_a .* ones(Float64, D) #[0.7; 0.7] # for each dimension.
    N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.

    X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D )
    removeclosepositionsarray!(X_array, close_radius_tol)
    # to do: auto fuse points with self-adapting close_radius_tol in fit.jl
    #       this is to avoid semidef error for the QP solver.

    f_X_array = collect( f_joint.(X_array[d]) for d = 1:D )


    println("Timing: fitproxydensities")
    @time c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array,
            Y_array = fitmarginalsadaptive(f_X_array, X_array,
                                        x_ranges, f,
                                        max_iters_RKHS, a_array, σ_array,
                                        amplification_factor, N_bands,
                                        attenuation_factor_at_cut_off,
                                        zero_tol_RKHS, prune_tol, max_integral_evals)

    println("Number of kernel centers kept, per dim:")
    println(collect( length(c_array[d]) for d = 1:D))


    ### set up KR config.

    # quantile-related parameters.
    max_integral_evals = 10000
    quantile_iters = 1000
    N_nodes_tanh = 200
    m_tsq = 7
    quantile_err_tol = 1e-3 #0.01  both gave l-2 norm( abs(u-u_rec) ), summed over all dimensions: 0.024788099072235982
    max_traversals = 50
    N_predictive_traversals = 40
    correction_epoch = 10
    quantile_max_iters = 500
    quantile_convergence_zero_tol = 1e-6 #1e-4
    n_limit = 10
    n0 = 5

    KR_config = KRTaylorConfigType( x_ranges,
                                    quantile_iters,
                                    max_integral_evals,
                                    N_nodes_tanh,
                                    m_tsq,
                                    quantile_err_tol,
                                    max_traversals,
                                    N_predictive_traversals,
                                    correction_epoch,
                                    quantile_max_iters,
                                    quantile_convergence_zero_tol,
                                    n_limit,
                                    n0 )

    return c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array,
            Y_array, KR_config, limit_a, limit_b, f
end
