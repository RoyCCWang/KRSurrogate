


# f is the CDF of f(x0[1:D], x[end]).
# This function is for the case when f isn't an adaptive kernel nor RQ kernel.
function setupquantilegenericproxy( f::Function,
                                a::T,
                                b::T,
                                d::Int;
                                max_numerical_inverse_iters::Int = 1000000,
                                max_integral_evals::Int = 10000,
                                initial_divisions::Int = 1,
                                N_nodes_tanh::Int = 100,
                                m_tsq::Int = 7) where T <: Real
    # conditional PDF.
    v_persist::Vector{T} = ones(T,d-1)
    Z_persist::Vector{T} = ones(T,1)
    f_v = xx->f([v_persist; xx]) # unnormalized pdf.

    𝑓 = xx->f_v(xx)/Z_persist[1] # normalized pdf.

    # update for the conditioning variables.
    updatev = vv->updatefvnoTaylor!(v_persist,
                    Z_persist,
                    vv,
                    f_v, # unnormalized pdf.
                    a,
                    b;
                    max_integral_evals = max_integral_evals,
                    initial_divisions = initial_divisions)

    # coniditional CDF.
    # to do: pass initial_divisions to evalcdfviaHCubature.
    𝐹 = xx->clamp(evalcdfviaHCubature(𝑓,
                    max_integral_evals,
                    a, xx)[1], zero(T), one(T))

    # initial search.
    q_v_initial = setupquantileintialsearch(a, b, 𝑓,
                                    N_nodes_tanh, m_tsq)

    q_v = uu->evalquantilewithoutTaylor(q_v_initial, 𝐹, uu, a, b,
                            𝑓, f_v;
                            max_iters = max_numerical_inverse_iters)

    # to do: add checks and speed this up.
    q = (vv,uu)->evalconditionalquantile(uu, vv, updatev, q_v)

    fetchZ = xx->fetchscalarpersist(Z_persist)

    return q, f_v, updatev, fetchZ
end

function evalconditionalquantile(y::T,
                                 v::Vector{T},
                                 updatev::Function,
                                 q_v::Function)::Tuple{T,T,Int} where T <: Real
    updatev(v)

    return q_v(y)
end

function updatefvnoTaylor!( v::Vector{T},
                            Z::Vector{T},
                            v_input::Vector{T},
                            fq_v0::Function,
                            lower_limit::T,
                            upper_limit::T;
                            max_integral_evals::Int = 10000,
                            initial_divisions::Int = 1) where T

    @assert length(v) == length(v_input)

    # update conditioning variables.
    v[:] = v_input
    # println("k = ", v)
    # println("fq_v0([1.23]) = ", fq_v0([1.23]))

    # update normalizing constant.
    val_Z, err_Z = HCubature.hcubature(fq_v0,
                        [lower_limit], [upper_limit];
                        norm = norm, rtol = sqrt(eps(T)),
                        atol = 0,
                        maxevals = max_integral_evals,
                        initdiv = initial_divisions)
    #println("val_Z = ", val_Z)
    Z[1] = val_Z

    return nothing
end

# problem: 𝑓 is sufficiently too close to zero. The normalizing constant is too small.
# This causes f_v(a) and f_v(b) to be zero, which
#   cases 𝑓(a) or 𝑓(b) to be NaN since the normalizing constant is Z.
function evalquantilewithoutTaylor(q_initial::Function,
                                    F::Function,
                                    y_target::T,
                                    a::T,
                                    b::T,
                                    𝑓::Function,
                                    f_v::Function;
                                    max_iters = 1000) where T <: Real
    # initial search.
    x0 = clamp(q_initial(y_target), a, b)

    # println("x0 = ", x0)
    # println("F(x0) = ", F(x0))
    # println("y_target = ", y_target)

    # refine.
    x_star, results = eval1Dnumericalinverse(F, y_target,
                                x0, a, b, max_iters)
    y_star = F(x_star)
    n_iters = results.iterations

    if !isfinite(y_star) || !isfinite(x_star)
        println("𝑓(a) = ", 𝑓(a))
        println("𝑓(b) = ", 𝑓(b))
        println("f_v(a) = ", f_v(a))
        println("f_v(b) = ", f_v(b))
        println("x0 = ", x0)
        println("results = ", results)
        println("y_target = ", y_target)
        println("F(x0) = ", F(x0))

        @assert 1==2
    end

    # println("x_star = ", x_star)
    # println("F(x_star) = ", F(x_star))


    return x_star, y_star, n_iters
end

# set up the quantile for the current dimension, d_select.
function setupquantilefordimW2( c::Vector{T},
                                θ::RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}},
                                X::Vector{Vector{T}},
                                max_integral_evals::Int,
                                limit_a::T,
                                limit_b::T,
                                d_ϕ_map::Function,
                                d2_ϕ_map::Function,
                                d_select::Int,
                                N_nodes_tanh::Int,
                                m_tsq::Int,
                                quantile_err_tol::T,
                                max_traversals::Int,
                                N_predictive_traversals::Int,
                                correction_epoch::Int,
                                quantile_max_iters::Int,
                                quantile_convergence_zero_tol::T,
                                n_limit::Int,
                                n0::Int)::Function where T <: Real

    # set up given current RKHS fit of pdf for the d_select-th dimension.
    K_array, sqrt_K_array, B_array,
    C_array, A_array, w2_t_array, b_array, w_X, P, 𝑐,
        𝑑 = setupTaylorquantilebuffers(length(X), one(T)) # specifically for the RQ adaptive kernel.

    updatewarpfuncevals!(w_X, θ.warpfunc, X, d_select)

    ### quantities that need updating once the content of v0 changes.
    v0, Z, Taylor_multiplier, F, F_no_clamp_with_err, update𝑐, update𝑑, updatev, 𝑓,
        ∂𝑓_∂x, q_v_initial, q_v, q, fq_v0 = setupTaylorquantilemethods(  𝑐, 𝑑, w_X,
                                            b_array,
                                            K_array,
                                            sqrt_K_array,
                                            B_array,
                                            C_array,
                                            A_array,
                                            w2_t_array,
                                            P,
                                        θ,
                                        c,
                                        X,
                                        max_integral_evals,
                                        limit_a,
                                        limit_b,
                                        d_ϕ_map,
                                        d2_ϕ_map,
                                        d_select,
                                            N_nodes_tanh,
                                            m_tsq,
                                            quantile_err_tol,
                                            max_traversals,
                                            N_predictive_traversals,
                                            correction_epoch,
                                            quantile_max_iters,
                                            quantile_convergence_zero_tol,
                                            n_limit,
                                            n0)


    return q
end


function parallelapplyadaptiveKRviaTaylorW2(  c_array::Vector{Vector{T}},
                            θ_array::Vector{RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}}},
                            𝓧_array::Vector{Vector{Vector{T}}},
                            d_ϕ_array::Vector{Function},
                            d2_ϕ_array::Vector{Function},
                            M::Int,
                            N_batches::Int,
                            config::KRTaylorConfigType{T,RT}) where {T <: Real, RT}

    # work on intervals.
    M_for_each_batch = Vector{Int}(undef, N_batches)
    𝑀::Int = round(Int, M/N_batches)
    fill!(M_for_each_batch, 𝑀)
    M_for_each_batch[end] = abs(M - (N_batches-1)*𝑀)

    @assert M == sum(M_for_each_batch) # sanity check.

    # prepare worker function.
    workerfunc = xx->applyadaptiveKRviaTaylorW2(   xx,
                                c_array,
                                θ_array,
                                𝓧_array,
                                config.max_integral_evals,
                                config.x_ranges,
                                d_ϕ_array,
                                d2_ϕ_array,
                                config.N_nodes_tanh,
                                config.m_tsq,
                                config.quantile_err_tol,
                                config.max_traversals,
                                config.N_predictive_traversals,
                                config.correction_epoch,
                                config.quantile_max_iters,
                                config.quantile_convergence_zero_tol,
                                config.n_limit,
                                config.n0)

    # compute solution.
    sol = pmap(workerfunc, M_for_each_batch )

    # unpack solution.
    x_array, discrepancy_array = unpackpmap(sol, M)

    #return x_array, discrepancy_array, sol
    return x_array, discrepancy_array
end

function unpackpmap(sol::Array{Tuple{Array{Array{T,1},1},Array{Array{T,1},1}},1},
                    M::Int)::Tuple{Vector{Vector{T}},Vector{Vector{T}}} where T <: Real

    N_batches = length(sol)

    x_array = Vector{Vector{T}}(undef,M)
    discrepancy_array = Vector{Vector{T}}(undef,M)

    st::Int = 0
    fin::Int = 0
    for j = 1:N_batches

        st = fin + 1
        fin = st + length(sol[j][1]) - 1

        x_array[st:fin] = sol[j][1]
        discrepancy_array[st:fin] = sol[j][2]
    end

    return x_array, discrepancy_array
end
