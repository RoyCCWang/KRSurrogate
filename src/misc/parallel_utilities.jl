



function parallelevals( f::Function,
                        X,
                        dummy::T) where T <: Real
    #
    N = length(X)

    a = SharedArray{T}(N)
    @sync begin
        @distributed for i = 1:N
            a[i] = f(X[i])
        end
    end

    return convert(Vector{T}, a)
end

function parallelevals( f::Function,
                        X::Vector{Vector{T}}) where T <: Real
    #
    N = length(X)

    a = SharedArray{T}(N)
    @sync begin
        @distributed for i = 1:N
            a[i] = f(X[i])
        end
    end

    return convert(Vector{T}, a)
end



##### evaluate pair-wise marginals via numerical integration. Slow.
function evalgridpairwisemarginal(  f::Function,
                                    x_ranges,
                                    d_i::Int,
                                    d_j::Int,
                                    z_min::Vector{T},
                                    z_max::Vector{T};
                                    initial_divisions::Int = 1,
                                    max_integral_evals::Int = 100000) where T
    #
    D = length(x_ranges)
    @assert D > 2

    #
    x_i_ranges = x_ranges[d_i]
    x_j_ranges = x_ranges[d_j]
    N_i = length(x_i_ranges)
    N_j = length(x_j_ranges)

    f_X = SharedArray{T}(N_i, N_j)

    @sync begin
        @distributed for j = 1:N_j
            for i = 1:N_i
                xi = x_i_ranges[i]
                xj = x_j_ranges[j]

                f_X[i,j] = evalpairwisemarginal(h, z_min, z_max,
                                        d_i, d_j,
                                        xi, xj, D;
                                        initial_divisions = initial_divisions,
                                        max_integral_evals = max_integral_evals)
            end
        end
    end

    return convert(Matrix{T}, f_X)
end

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

function substitutedim(z::AbstractVector{T}, xi, xj, i, j)::Vector{T} where T
    D = length(z) + 2
    out = Vector{T}(undef, D)

    a = 1
    for k = 1:length(out)
        if k != i && k !=j
            out[k] = z[a]
            a += 1
        end
    end

    out[i] = xi
    out[j] = xj

    return out
end
