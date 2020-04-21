# returns an unnoramlized density.
function getcauchygmmdensity(   dummy_val::T;
                                D = 3,
                                N_array = [100; 100; 100],
                                N_components = 3 ) where T <: Real
    ## set up.
     @assert D == length(N_array)

    limit_a = ones(T, D) .* τ
    limit_b = ones(T, D) - limit_a

    # oracle probability density.
    Y_dist, f_syn, x_ranges = setupsyntheticdataset(N_components,
                                    N_array, limit_a, limit_b,
                                    getnDGMMrealizations, Val(D))

    # add cauchy component to avoid numerical ill-conditioning.
    μ_cauchy = randn(D) .* 3
    σ_cauchy = rand(D) .+ 2.5
    cauchy_dists = collect( Distributions.Cauchy(μ_cauchy[d], σ_cauchy[d]) for d = 1:D )
    cauchy_pdf = xx->exp(sum( Distributions.logpdf(cauchy_dists[d], xx[d]) for d = 1:D))
    f = xx->(f_syn(xx)+cauchy_pdf(xx))

    # check if cauchy is enough.
    f_ratio_tol = 1e-8
    status_flag, min_f_x, max_f_x = dynamicrangecheck(f, x_ranges,
                                            f_ratio_tol)
    @assert status_flag # to do: handle this failure gracefully.

    #f_joint = xx->evaljointpdf(xx,f,D)[1]
    f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

    𝑋 = collect( rand(Y_dist, 1) for n = 1:N_realizations)
    # I am here. figure out how to mixture sample from Cauchy correctly.

    return f, f_joint
end


# in progress.
function fitcauchygmm()
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
