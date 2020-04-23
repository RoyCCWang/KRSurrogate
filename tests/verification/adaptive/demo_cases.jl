
function democgmm3Dadaptivebp(;    limit_a = [-14.0; -16.0; -11.0],
                        limit_b = [16.0; 15.0; 5.0],
                        N_array = [100; 100; 100],
                        N_realizations = 40,
                        fig_num = 1,
                        max_integral_evals = 100000,
                        initial_divisions = 1,
                        skip_flag = false)

    # set up target.
    D = length(limit_a)

    Y_pdf, Y_dist, x_ranges = setupskipDgmm3D(limit_a, limit_b, N_array)

    f, f_joint = preparesyntheticdist(Y_pdf, Y_dist, x_ranges)

    f_ratio_tol = 1e-8
    status_flag, min_f_x, max_f_x = dynamicrangecheck(f, x_ranges, f_ratio_tol)
    @assert status_flag

    # set up kernel centers.
    𝑋 = collect( rand(Y_dist) for n = 1:N_realizations)

    N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.
    N_X[1] = 25

    # fit density via Bayesian optimization.

    amplification_factor = 2.0
    a_array = [0.7; 0.1; 0.1]
    if skip_flag
        a_array = [0.3; 0.3]
    end

    c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array,
            Y_array = fitdensityadaptivebp(f, f_joint, 𝑋, N_X, x_ranges,
                                        amplification_factor = amplification_factor,
                                        a_array = a_array,
                                        skip_flag = skip_flag)


    # source distribution parameters.
    src_μ = [0.6782809912688208; 0.29860446771736016; 0.7943982298972518]
    src_σ_array = [5.719457852631949; 3.55395529739239; 2.558762578342345]

    return c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array, f, limit_a, limit_b, x_ranges,
            src_μ, src_σ_array, Y_array, f_joint
end


function democgmm3Dnonadaptive(;    limit_a = [-14.0; -16.0; -11.0],
                        limit_b = [16.0; 15.0; 5.0],
                        N_array = [100; 100; 100],
                        N_realizations = 40,
                        fig_num = 1,
                        max_integral_evals = 100000,
                        initial_divisions = 1,
                        skip_flag = false)

    # set up target.
    D = length(limit_a)

    Y_pdf, Y_dist, x_ranges = setupskipDgmm3D(limit_a, limit_b, N_array)

    f, f_joint = preparesyntheticdist(Y_pdf, Y_dist, x_ranges)

    f_ratio_tol = 1e-8
    status_flag, min_f_x, max_f_x = dynamicrangecheck(f, x_ranges, f_ratio_tol)
    @assert status_flag

    # set up kernel centers.
    𝑋 = collect( rand(Y_dist) for n = 1:N_realizations)

    N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.
    N_X[1] = 25

    # fit density via Bayesian optimization.

    a_array = [0.1; 0.1; 0.1]

    c_array, 𝓧_array, θ_array,
        X_array_debug, f_X_array_debug = fitdensitynonadaptive(f, f_joint, 𝑋, N_X,
                                a_array = a_array,
                                skip_flag = skip_flag)


    # source distribution parameters.
    src_μ = [0.6782809912688208; 0.29860446771736016; 0.7943982298972518]
    src_σ_array = [5.719457852631949; 3.55395529739239; 2.558762578342345]

    return c_array, 𝓧_array, θ_array, f, limit_a, limit_b, x_ranges,
            src_μ, src_σ_array, f_joint
end


### helmet.

function demohelmetadaptivebp(  a_array::Vector{T},
                                σ_array::Vector{T};
                                τ = 2, # offset from border of samples.
                                #N_realizations = 40,
                                fig_num = 1,
                                max_integral_evals = 100000,
                                initial_divisions = 1,
                                skip_flag = false,
                                amplification_factor = 1.0,
                                zero_tol_RKHS::T = 1e-10,
                                max_iters::Int = 2000,
                                scene_tag = "squiggle",
                                downsample_factor = 1.0,
                                kernel_center_option = "grid") where T <: Real

    # set up target.
    D = 2

    #Y_pdf, Y_dist, x_ranges = setupskipDgmm3D(limit_a, limit_b, N_array)

    f, x_ranges, limit_a, limit_b, fig_num = setuporaclefromimage(scene = scene_tag)
    N_array = collect( length(x_ranges[d]) for d = 1:D )

    #f_joint = xx->evaljointpdf(xx,f,D)[1]
    f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

    f_ratio_tol = 1e-8
    status_flag, min_f_x, max_f_x = dynamicrangecheck(f, x_ranges, f_ratio_tol)
    @assert status_flag



    ### set up kernel centers.
    𝑋 = Vector{Vector{T}}(undef, 0)

    if kernel_center_option == "grid"
        ## option 1: grid sample the sample space of the target distribution.
        Np_array::Vector{Int} = collect( div(N_array[d], downsample_factor) for d = 1:2 )
        Xp_ranges = collect( LinRange(limit_a[d], limit_b[d], Np_array[d] ) for d = 1:D )
        Xp = Utilities.ranges2collection(Xp_ranges, Val(D))
        𝑋 = vec(Xp)

    elseif kernel_center_option == "draw"

        ## option 2: samples of the target distributions, if we know how to sample from it.
        𝑋 = collect( rand(Y_dist) for n = 1:N_realizations)

    elseif kernel_center_option == "kDPP"
        # ## option 3: kDPP.
        canonical_a = a_array[end]
        𝑋 = getkernelcentersviaadaptivekernel(canonical_a, x_ranges, limit_a, limit_b;
                    amplification_factor = amplification_factor,
                    N_preliminary_candidates = 2000,
                    N_kDPP_per_draw = 100,
                    N_refinements = 10)
    end

    # fine-tune candidate kernel centers.
    N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.
    #N_X[1] = 25
    println("N_X = ", N_X)
    # @assert 1==2

    # fit density via Riemannian optimization.
    fit_optim_config = ROptimConfigType(zero_tol_RKHS, max_iters)

    #
    c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array,
            Y_array = fitdensityadaptivebp(f, f_joint, 𝑋, N_X, x_ranges,
                                        fit_optim_config;
                                        amplification_factor = amplification_factor,
                                        a_array = a_array,
                                        skip_flag = skip_flag,
                                        σ_array = σ_array)


    # source distribution parameters.
    src_μ::Vector{T} = (limit_a + limit_b) ./2
    src_σ_array::Vector{T} = [13.926789325846464; 12.348448115367624]

    return c_array, 𝓧_array, θ_array,
            dφ_array, d2φ_array,
            dϕ_array, d2ϕ_array, f, limit_a, limit_b, x_ranges,
            src_μ, src_σ_array, Y_array, f_joint
end

function demohelmetnonadaptive(  a_array::Vector{T},
                                σ_array::Vector{T};
                                τ = 2, # offset from border of samples.
                                #N_realizations = 40,
                                fig_num = 1,
                                max_integral_evals = 100000,
                                initial_divisions = 1,
                                skip_flag = false,
                                zero_tol_RKHS::T = 1e-10,
                                max_iters::Int = 2000,
                                scene_tag = "squiggle",
                                downsample_factor = 1.0,
                                kernel_center_option = "grid") where T <: Real

    # set up target.
    D = 2

    #Y_pdf, Y_dist, x_ranges = setupskipDgmm3D(limit_a, limit_b, N_array)

    f, x_ranges, limit_a, limit_b, fig_num = setuporaclefromimage(scene = scene_tag)
    N_array = collect( length(x_ranges[d]) for d = 1:D )

    #f_joint = xx->evaljointpdf(xx,f,D)[1]
    f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

    f_ratio_tol = 1e-8
    status_flag, min_f_x, max_f_x = dynamicrangecheck(f, x_ranges, f_ratio_tol)
    @assert status_flag



    ### set up kernel centers.
    𝑋 = Vector{Vector{T}}(undef, 0)

    if kernel_center_option == "grid"
        ## option 1: grid sample the sample space of the target distribution.
        Np_array::Vector{Int} = collect( div(N_array[d], downsample_factor) for d = 1:2 )
        Xp_ranges = collect( LinRange(limit_a[d], limit_b[d], Np_array[d] ) for d = 1:D )
        Xp = Utilities.ranges2collection(Xp_ranges, Val(D))
        𝑋 = vec(Xp)

    elseif kernel_center_option == "draw"

        ## option 2: samples of the target distributions, if we know how to sample from it.
        𝑋 = collect( rand(Y_dist) for n = 1:N_realizations)

    elseif kernel_center_option == "kDPP"
        # ## option 3: kDPP.
        canonical_a = a_array[end]
        𝑋 = getkernelcentersviaadaptivekernel(canonical_a, x_ranges, limit_a, limit_b;
                    amplification_factor = amplification_factor,
                    N_preliminary_candidates = 2000,
                    N_kDPP_per_draw = 100,
                    N_refinements = 10)
    end

    N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.
    #N_X[1] = 25
    println("N_X = ", N_X)
    # @assert 1==2

    # fit density via Riemannian optimization.
    fit_optim_config = ROptimConfigType(zero_tol_RKHS, max_iters)

    # fit.
    c_array, 𝓧_array, θ_array,
        X_array_debug, f_X_array_debug = fitdensitynonadaptive(f, f_joint,
                                𝑋, N_X, fit_optim_config;
                                a_array = a_array,
                                skip_flag = skip_flag)

    # source distribution parameters.
    src_μ::Vector{T} = (limit_a + limit_b) ./2
    src_σ_array::Vector{T} = [13.926789325846464; 12.348448115367624]

    return c_array, 𝓧_array, θ_array, f, limit_a, limit_b, x_ranges,
            src_μ, src_σ_array, f_joint
end
