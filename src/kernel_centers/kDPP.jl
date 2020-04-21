

function selectelementviaunionofkDPP(   𝓨::Vector{Vector{T}},
                                        N_unions::Int,
                                        M::Int,
                                        θ::KT,
                                        warning_tol::T = 1e-8)::Tuple{Vector{Vector{T}}, Vector{Int}} where {T,KT}
    #
    ind_kDPP = Vector{Int}(undef, M*N_unions)

    fin::Int = 0
    st::Int = 0

    for i = 1:N_unions
        st = fin + 1
        fin = st + M -1


        Y_unused, ln_pdf_eval_unused, x = selectelementsviakDPP(𝓨,
                            M, θ, false, warning_tol)

        if length(x) == 0
            return Vector{Vector{T}}(undef,0), Vector{Int}(undef,0)
        end
        ind_kDPP[st:fin] = x
    end

    # keep unique ones.
    ind_kDPP_unique = unique(ind_kDPP)
    Y = 𝓨[ind_kDPP_unique]

    return Y, ind_kDPP_unique
end


# selct M elements from the set 𝓨, using the kernel θ.
function selectelementsviakDPP( 𝓨::Vector{Vector{T}},
                                M::Int,
                                θ::KT,
                                compute_ln_prob_flag::Bool = false,
                                warning_tol::T = 1e-8)::Tuple{Vector{Vector{T}}, T, Vector{Int}} where {T,KT}
    @assert length(𝓨) > M

    # prepare the similarity matrix, L. Denoted K here.
    K_full = RKHSRegularization.constructkernelmatrix(𝓨, θ)

    L_full = K_full
    #id_mat = Matrix{T}(LinearAlgebra.I, size(K_full))
    #L_full = Utilities.forcesymmetric(inv(id_mat-K_full)-id_mat)

    s0, Q = eigen(L_full)

    # force eigen values to be positive if semi-definite matrix.
    𝑠0 = abs.(s0)


    L_kDPP = L_full ./ maximum(𝑠0)
    s, Q = eigen(L_kDPP)

    # force eigen values to be positive if semi-definite matrix.
    𝑠 = abs.(s)
    if norm(s-𝑠) > warning_tol
        println("selectelementsviakDPP encountered negative or zero eigenvalues.")
    end

    # draw.
    ind_kDPP, e_mat = samplekDPP(M, 𝑠)
    Y = 𝓨[ind_kDPP]

    # compute probability of the draw Y.
    ln_pdf_eval = NaN
    if compute_ln_prob_flag
        K_Y = RKHSRegularization.constructkernelmatrix(Y, θ)

        Z = e_mat[end]
        ln_pdf_eval = logdet(K_Y) - log(Z)
    end

    return Y, ln_pdf_eval, ind_kDPP
end
