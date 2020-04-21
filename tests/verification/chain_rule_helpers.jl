######## for verification.

function d2fkwrtdx(   x0::Vector{T},
                    f::Function,
                    g::Function,
                    df::Function,
                    dg::Function,
                    d2f::Function,
                    d2g::Function,
                    k::Int) where T <: Real

    dg_x0::Vector{Vector{T}} = dg(x0)
    g_x0 = g(x0)
    f_g0 = f(g_x0)
    df_g0::Vector{Vector{T}} = df(g_x0)

    d2f_g0::Vector{Matrix{T}} = d2f(g_x0)
    d2g_x0::Vector{Matrix{T}} = d2g(x0)

    D = length(x0)

    out = zeros(T, D, D)
    for j = 1:D
        for i = 1:D

            for l = 1:D

                ∂zlg_∂xj = computedzlg(d2f_g0, dg_x0, k, l, j)
                term1 = ∂zlg_∂xj*dg_x0[l][i]

                term2 = df_g0[k][l]*d2g_x0[l][j,i]

                out[j,i] += term1 + term2
            end

        end
    end

    return out
end

"""
Computes ∂(z_l ∘ g)/∂x_j (x)
"""
function computedzlg(   d2f_g0::Vector{Matrix{T}},
                        dg_x0::Vector{Vector{T}},
                        k::Int,
                        l::Int,
                        j::Int) where T <: Real
    D = size(d2f_g0, 1)

    out = zero(T)
    for b = 1:D
        out += d2f_g0[k][l,b]*dg_x0[b][j]
    end

    return out
end


function d2fkwrtdxterms(   x0::Vector{T},
                    f::Function,
                    g::Function,
                    df::Function,
                    dg::Function,
                    d2f::Function,
                    d2g::Function,
                    k::Int) where T <: Real

    dg_x0::Vector{Vector{T}} = dg(x0)
    g_x0 = g(x0)
    f_g0 = f(g_x0)
    df_g0::Vector{Vector{T}} = df(g_x0)

    d2f_g0::Vector{Matrix{T}} = d2f(g_x0)
    d2g_x0::Vector{Matrix{T}} = d2g(x0)

    D = length(x0)

    LHS = zeros(T, D, D)
    RHS = zeros(T, D, D)
    for j = 1:D
        for i = 1:D

            for l = 1:D

                ∂zlg_∂xj = computedzlg(d2f_g0, dg_x0, k, l, j)
                term1 = ∂zlg_∂xj*dg_x0[l][i]

                term2 = df_g0[k][l]*d2g_x0[l][j,i]

                #out[j,i] += term1 + term2
                LHS[j,i] += term1
                RHS[j,i] += term2
            end

        end
    end

    return LHS, RHS
end
