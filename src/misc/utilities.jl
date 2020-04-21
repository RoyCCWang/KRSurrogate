
# based on the central-difference in:
# https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
function numericalhessian(f::Function, x0::Vector{T}; order::Int = 5)::Matrix{T} where T
    D = length(x0)

    fdm = FiniteDifferences.central_fdm(order, 1)
    df = xx->FiniteDifferences.grad(fdm, f, xx)
    df_array = collect( xx->FiniteDifferences.grad(fdm, f, xx)[1][i] for i = 1:D )

    d2f_x0 = zeros(T, D, D)
    for i = 1:D
        d2f_i = xx->FiniteDifferences.grad(fdm, df_array[i], xx)

        for j = 1:D
            d2f_j = xx->FiniteDifferences.grad(fdm, df_array[j], xx)

            t1 = d2f_i(x0)[1][j]
            t2 = d2f_j(x0)[1][i]

            # # debug.
            # v1 = df(x0)
            # println("v1 = ", v1)
            # u1 = df_array[i](x0)
            # println("u1 = ", u1)
            # println("t1 = ", t1)
            # println("t2 = ", t2)

            d2f_x0[i,j] = 0.5*( t1 + t2 )
        end
    end

    return d2f_x0
end
### to do: package this into Utilities.
# # test code.
# import ForwardDiff
# import FiniteDifferences
# import Utilities
#
# D = 2
# A = Utilities.generaterandomposdefmat(D)
# v = randn(D)
# f = xx->exp(-dot(xx,A*xx))*sin(dot(xx-v,xx-v))
#
# x0 = randn(D)
# H_ND = numericalhessian(f, x0)
# H_AD = ForwardDiff.hessian(f,x0)
#
# println("discrepancy = ", norm(H_ND-H_AD))



function dynamicrangecheck(f::Function,
                            x_ranges::Vector{LinRange{T}},
                            ratio_tol::T) where T <: Real
    #
    # sanity-check:
    max_val = -Inf
    min_val = Inf

    r = x_ranges[end]

    for x_tuple in Iterators.product(x_ranges...)

        f_x = f(collect(x_tuple))

        if max_val < f_x
            max_val = f_x
        end

        if min_val > f_x
            min_val = f_x
        end
    end


    status_flag = min_val/max_val > ratio_tol
    println("ratio is = ", min_val/max_val)
    println("max f(x) = ", max_val)
    println("min f(x) = ", min_val)
    println()

    return status_flag, min_val, max_val
end

# for a lower-triangular dT.
function evalKRd2Tratiodiscrepancy(X::Vector{Matrix{T}}, Y) where T
    @assert length(X) == length(Y)
    return collect( X[d][1:d,1:d] ./ Y[d][1:d,1:d] for d = 1:length(Y) )
end

function ratiodiscrepancywithintol( disc::Vector{Matrix{T}};
                                    tol = 1e-4) where T
    #
    D = length(disc)

    out = falses(D)
    for d = 1:D

        Nd = length(disc[d])
        flags_d = falses(Nd)
        for n = 1:Nd
            flags_d[n] = Utilities.isnumericallyclose(disc[d][n], one(T), tol)
        end

        out[d] = all(flags_d)
    end

    return out
end

function fetchscalarpersist(Z::Vector{T})::T where T
    return Z[1]
end

function updatevy1d!(   b_array::Vector{T},
                        v::Vector{T},
                        y1d::Vector{T},
                        Z_v::Vector{T},
                        f_v::Function,
                        x_full::Vector{T},
                        y_in::Vector{T},
                        d::Int,
                        θ::KT,
                        X::Vector{Vector{T}},
                        lower_limit::T,
                        upper_limit::T;
                        zero_tol::T = 1e-8,
                        initial_divisions::Int = 1,
                        max_integral_evals::Int = 100000) where {T <: Real, KT}
    d = length(y_in)
    @assert length(v) == d-1
    @assert length(y1d) == d

    y1d[:] = y_in[1:d]
    v[:] = y_in[1:d-1]
    x_full[1:end-1] = y_in[1:d-1]
    #println("y1d = ", y1d)

    D_m_1 = length(v)
    for n = 1:length(b_array)
        b_array[n] = θ.canonical_params.a + norm(v - X[n][1:D_m_1])^2
    end

    val_Z, err_Z = HCubature.hquadrature(f_v,
                        lower_limit, upper_limit;
                        norm = norm, rtol = sqrt(eps(T)),
                        atol = 0,
                        maxevals = max_integral_evals,
                        initdiv = initial_divisions)

    Z_v[1] = clamp(val_Z, zero_tol, one(T))

    return nothing
end

function updatevy1d!(   b_array::Vector{T},
                        v::Vector{T},
                        y1d::Vector{T},
                        Z_v::Vector{T},
                        f_v::Function,
                        x_full::Vector{T},
                        y_in::Vector{T},
                        d::Int,
                        θ::KT,
                        X::Vector{Vector{T}},
                        lower_limit::T,
                        upper_limit::T;
                        zero_tol::T = 1e-8,
                        initial_divisions::Int = 1,
                        max_integral_evals::Int = 100000) where {T <: Real, KT}
    d = length(y_in)
    @assert length(v) == d-1
    @assert length(y1d) == d

    y1d[:] = y_in[1:d]
    v[:] = y_in[1:d-1]
    x_full[1:end-1] = y_in[1:d-1]
    #println("y1d = ", y1d)

    D_m_1 = length(v)
    for n = 1:length(b_array)
        b_array[n] = θ.canonical_params.a + norm(v - X[n][1:D_m_1])^2
    end

    val_Z, err_Z = HCubature.hquadrature(f_v,
                        lower_limit, upper_limit;
                        norm = norm, rtol = sqrt(eps(T)),
                        atol = 0,
                        maxevals = max_integral_evals,
                        initdiv = initial_divisions)

    Z_v[1] = clamp(val_Z, zero_tol, one(T))

    return nothing
end

"""

"""
function evaldg99( updatev_array::Vector{Function},
                    dG_array::Vector{Function},
                    y::Vector{T})::Vector{Vector{T}} where T <: Real

    D = length(y)
    @assert length(dG_array) == D == length(updatev_array)

    dg_y = Vector{Vector{T}}(undef, D)
    for d = 1:D

        updatev_array[d](y, d)

        dg_y[d] = dG_array[d](y)
    end

    return dg_y

    #return invertlowertriangularrowwise(dg_y)
end

"""
L[i] is the i-th row of the lower triangular matrix L_mat.
This function returns inv(L_mat) with a dense matrix data type.
"""
function invertlowertriangularrowwise(L::Vector{Vector{T}})::Matrix{T} where T <: Real
    D = length(L)

    X = zeros(T, D, D)

    for k = 1:D
        X[k,k] = one(T)/L[k][k]

        for i = k+1:D
            X[i,k] = -L[i][k:i-1]*X[k:i-1,k]/L[i][i]
        end
    end

    return X
end

function removeclosepositionsarray!(X_array::Vector{Vector{Vector{T}}}, ϵ::T) where T

    for i = 1:length(X_array)
        X_array[i] = removeclosepositions!(X_array[i], ϵ)
    end

    return nothing
end

function removeclosepositions!(X::Vector{Vector{T}}, ϵ::T) where T <: Real
    @assert ϵ > zero(T)

    N = length(X)
    D = length(X[1])

    #ϵ = 0.05
    #point = rand(3)

    X_pool = copy(X)

    out = Vector{Vector{T}}(undef, N)
    out_ind = 0

    while !isempty(X_pool)
        #println("X_pool = ", X_pool)

        point = X_pool[1]

        # find point.
        X_pool_mat = Utilities.array2matrix(X_pool)

        balltree = NearestNeighbors.BallTree(X_pool_mat)
        merge_indices = NearestNeighbors.inrange(balltree, point, ϵ, true) # guaranteed to have 1 hit, at index 1.

        #println("merge_indices = ", merge_indices)

        # store merged point.
        out_ind += 1
        out[out_ind] = sum( X_pool[i] for i in merge_indices ) ./ length(merge_indices)

        # delete points that make up the merged point.
        deleteat!(X_pool, merge_indices)

    end

    resize!(out, out_ind)
    #@assert 555==5
    return out
end


# y assumed to have been sorted in descending order.
function findfirstsorteddescending(y::Vector{T}, threshold::T)::Int where T
    for i = 1:length(y)
        if y[i] < threshold
            return i
        end
    end

    return 0
end



function setupadaptivekernel(   θ_canonical::KT,
                                Y_nD::Array{T,D},
                                f::Function,
                                x_ranges;
                                amplification_factor::T = 1.0,
                                attenuation_factor_at_cut_off::T = 2.0,
                                N_bands::Int = 5)::Tuple{RKHSRegularization.AdaptiveKernelType{KT}, Function, Function, Function} where {T,D,KT}

    # warp map parameter set up.
    reciprocal_cut_off_percentages = ones(N_bands) ./collect(LinRange(1.0,0.2,N_bands))
    ω_set = collect( π/(reciprocal_cut_off_percentages[i]*sqrt(2*log(attenuation_factor_at_cut_off))) for i = 1:length(reciprocal_cut_off_percentages) )
    pass_band_factor = abs(ω_set[1]-ω_set[2])*0.2

    # get bandpass samples on x_ranges.
    ϕ = RKHSRegularization.getRieszwarpmapsamples(Y_nD, Val(:simple), Val(:uniform),
                        ω_set, pass_band_factor)

    # get warpmap from samples.
    ϕ_map_func, d_ϕ_map_func,
      d2_ϕ_map_func = Utilities.setupcubicitp(ϕ, x_ranges,
                            amplification_factor)
    # #### df2 experiment:
    # ϕ_map_func = xx->norm(diag(ForwardDiff.hessian(f,xx)))
    # #  see df2_warpmap.jl in RKHSRegularization.
    #
    # #### end experiment.

    # assemble adaptive kernel.
    θ_a = RKHSRegularization.AdaptiveKernelType(θ_canonical, ϕ_map_func)

    return θ_a, ϕ_map_func, d_ϕ_map_func, d2_ϕ_map_func
end


# compute df_u*dg_x.
# Q*diagm(a) is Q with the j-th column per-element-multiplied with a[j].
function lowertriangularpostmultiplydiagonalmatrix!(A::Matrix{T}, c::Vector{T}) where T <: Real

    N = length(c)
    @assert size(A,1) == N == size(A,2)

    for j = 1:N
        for i = j:N
            A[i,j] = A[i,j] *c[j]
        end
    end

    return nothing
end

#### fast querying of RKHS solutions.

# v0 is taken into account with b_array.
function evalqueryRQ!(  x_full::Vector{T},
                        x::T,
                        c::Vector{T},
                        X::Vector{Vector{T}},
                        ϕ::Function,
                        b_array::Vector{T},
                        w_X::Vector{T},
                        multiplier::T)::T where T <: Real
    #
    x_full[end] = x

    w_x = ϕ(x_full)
    out = sum( c[n]/sqrt( b_array[n] + (x-X[n][end])^2 + (w_x-w_X[n])^2 )^3 for n = 1:length(c) )

    out = out*multiplier
    return out
end

function evalqueryRQ!(  x_full::Vector{T},
                        x,
                        c::Vector{T},
                        X::Vector{Vector{T}},
                        ϕ::Function,
                        b_array::Vector{T},
                        w_X::Vector{T},
                        multiplier::T)::T where T <: Real
    #
    return evalqueryRQ!(  x_full,
                            x[1],
                            c,
                            X,
                            ϕ,
                            b_array,
                            w_X,
                            multiplier)
end
