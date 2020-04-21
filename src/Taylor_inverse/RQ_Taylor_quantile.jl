
# for production, with use with RKHS query's cdf.



# f is the normalized conditional density function.
# if dir_var > 0, we want the Taylor approximation for values
#   larger than t (forward of t). Vice versa.
function evalRQTaylorcoeffsquery!(  eval_array::Vector{T},
                                    K_array::Vector{Vector{T}},
                                    sqrt_K_array::Vector{Vector{T}},
                                    B_array::Vector{Vector{T}},
                                    C_array::Vector{Vector{T}},
                                    A_array::Vector{Vector{T}},
                                    w2_t_array::Vector{T},
                    t::T,
                     v::Vector{T},
                     X::Vector{Vector{T}},
                     c::Vector{T},
                     w::Function,
                     w1::Function,
                     w2::Function,
                     b_array::Vector{T},
                     w_X::Vector{T},
                     multiplier_persist::Vector{T},
                     f::Function,
                     dir_var::T)::Nothing where T <: Real
    #
    N = length(b_array)
    @assert length(w_X) == N

    multiplier = multiplier_persist[1]

    # compute the common quantities.
    #x = [v; t]
    #w_t::T = w(x)
    w_t::T = w(t)
    w1_t::T = w1(t, dir_var)
    w2_t::T = w2(t, dir_var)

    # update quantities.
    for i = 1:length(w2_t_array)
        w2_t_array[i] = w2_t^i
    end

    for n = 1:N

        # alias for readability.
        z = X[n][end]
        w_z = w_X[n]
        b = b_array[n]

        K = K_array[n]
        sqrt_K = sqrt_K_array[n]
        B = B_array[n]
        C = C_array[n]
        A = A_array[n]

        # updates.
        K[1] = b + (t-z)^2 + (w_t-w_z)^2
        for i = 2:length(K)
            K[i] = K[1]^i
        end

        for i = 1:length(sqrt_K)
            sqrt_K[i] = sqrt(K[i])
        end

        B[1] = t-z + w1_t * (w_t-w_z)
        for i = 2:length(B)
            B[i] = B[1]^i
        end

        C[1] = 1 + w1_t^2 + w2_t * (w_t-w_z)
        for i = 2:length(C)
            C[i] = C[1]^i
        end

        A[1] = w1_t * w2_t
        for i = 2:length(A)
            A[i] = A[1]^i
        end
    end

    # eval derivatives.
    resize!(eval_array,11)
    fill!(eval_array,zero(T))

    eval_array[1] = f(t)

    for n = 1:N

        eval_array[2] += c[n]/2*evalRQTaylor1adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[3] += c[n]/3*evalRQTaylor2adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[4] += c[n]/4*evalRQTaylor3adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[5] += c[n]/5*evalRQTaylor4adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[6] += c[n]/6*evalRQTaylor5adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[7] += c[n]/7*evalRQTaylor6adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[8] += c[n]/8*evalRQTaylor7adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[9] += c[n]/9*evalRQTaylor8adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[10] += c[n]/10*evalRQTaylor9adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])
        #
        eval_array[11] += c[n]/11*evalRQTaylor10adaptivekernel(w1_t,
                                                        w2_t_array,
                                                        K_array[n],
                                                        sqrt_K_array[n],
                                                        B_array[n],
                                                        C_array[n],
                                                        A_array[n])

    end

    # Adjust by multiplier for the kernel derivative entries.
    # The normalizing constant is included here.
    for i = 2:length(eval_array)
        eval_array[i] *= multiplier
    end

    return nothing
end


function evalRQTaylor1adaptivekernel(  w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real

    return -3*B[1]/sqrt_K[5]
end


function evalRQTaylor2adaptivekernel(  w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real
    #
    denominator = 2 * sqrt_K[7]

    term1 = 15 * B[2]
    term2 = -3 * C[1] * K[1]

    return (term1 + term2)/denominator
end

function evalRQTaylor3adaptivekernel(  w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real

    denominator = sqrt_K[9]

    term1 = -35 * B[3]
    term2 = -3 * A[1] *K[2]
    term3 = C[1] * B[1] * 15*K[1]

    return (term1 + term2 + term3)/(2*denominator)
end

function evalRQTaylor4adaptivekernel(  w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real

    denominator = sqrt_K[11]

    term1 = 315/8 * B[4]
    term2 = -3/8 * w2_t[2] * K[3]
    term3 = 15/8 * C[2] * K[2]
    term4 = -3/8 * B[2] * C[1] * 70 * K[1]
    term5 = 15/2 * A[1] * B[1] * K[2]

    return (term1 + term2 + term3 + term4 + term5)/denominator
end


function evalRQTaylor5adaptivekernel( w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real

    term1 = -693/8 * B[5] /sqrt_K[13]
    term2 = -105/4 * A[1] * B[2] / sqrt_K[9]
    term3 = -105/8 * B[1] * C[2] / sqrt_K[9]
    term4 = 15/8 * B[1] / sqrt_K[7] * w2_t[2]
    term5 = 15/4 * A[1] * C[1] / sqrt_K[7]
    term6 = 315/4 * C[1] * B[3] / sqrt_K[11]

    return term1 + term2 + term3 + term4 + term5 + term6
end

function evalRQTaylor6adaptivekernel(  w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real

    term1 = -35/16 * C[3] / sqrt_K[9]
    term2 = 15/8 * A[2] / sqrt_K[7]
    term3 = 3003/16 * B[6] / sqrt_K[15]
    term4 = -3465/16 * C[1] * B[4] / sqrt_K[13]
    term5 = -105/16 * B[2] / sqrt_K[9] * w2_t[2]
    term6 = 15/16 * C[1] / sqrt_K[7] * w2_t[2]
    term7 = 945/16 * B[2] * C[2] / sqrt_K[11]
    term8 = 315/4 * A[1] * B[3] / sqrt_K[11]
    term9 = -105/4 * A[1] * B[1] * C[1] / sqrt_K[9]

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
end

function evalRQTaylor7adaptivekernel(  w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real
    #

    term1 = -2027025/5040 * B[7] / sqrt_K[17]
    term2 = -1091475/5040 * A[1] * B[4] / sqrt_K[13]
    term3 = -1091475/5040 * B[3] * C[2] / sqrt_K[13]
    term4 = -66150/5040 * B[1] * A[2] / sqrt_K[9]
    term5 = -33075/5040 * A[1] * C[2] / sqrt_K[9]
    term6 = 4725/5040 * A[1] / sqrt_K[7] * w2_t[2]
    term7 = 99225/5040 * B[1] * C[3] / sqrt_K[11]
    term8 = 99225/5040 * B[3] / sqrt_K[11] * w2_t[2]
    term9 = 2837835/5040 * C[1] * B[5] / sqrt_K[15]
    term10 = -33075/5040 * B[1] * C[1] / sqrt_K[9] * w2_t[2]
    term11 = 595350/5040 * A[1] * C[1] * B[2] / sqrt_K[11]

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 +
              term9 + term10 + term11
end

function evalRQTaylor8adaptivekernel(  w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real
    #

    term1 = 4725/40320 / sqrt_K[7] * w2_t[4]
    term2 = 99225/40320 * C[4] / sqrt_K[11]
    term3 = 34459425/40320 * B[8] / sqrt_K[19]
    term4 = -56756700/40320 * C[1] * B[6] / sqrt_K[17]
    term5 = -4365900/40320 * B[2] * C[3] / sqrt_K[13]
    term6 = -2182950/40320 * B[4] / sqrt_K[13] * w2_t[2]
    term7 = -264600/40320 * C[1] * A[2] / sqrt_K[9]
    term8 = -66150/40320 * C[2] / sqrt_K[9] * w2_t[2]
    term9 = 2381400/40320 * A[2] * B[2] / sqrt_K[11]
    term10 = 22702680/40320 * A[1] * B[5] / sqrt_K[15]
    term11 = 28378350/40320 * B[4] * C[2] / sqrt_K[15]
    term12 = -17463600/40320 * A[1] * C[1] * B[3] / sqrt_K[13]
    term13 = -264600/40320 * B[1] * A[1] / sqrt_K[9] * w2_t[2]
    term14 = 1190700/40320 * C[1] * B[2] / sqrt_K[11] * w2_t[2]
    term15 = 2381400/40320 * A[1] * B[1] * C[2] / sqrt_K[11]

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 +
              term9 + term10 + term11 + term12 + term13 + term14 +term15
end


function evalRQTaylor9adaptivekernel(  w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real
    #

    # term1 = -654729075//362880 * B[9] / sqrt_K[21]
    # term2 = -766215450//362880 * B[5] * C[2] / sqrt_K[17]
    # term3 = -510810300//362880 * A[1] * B[6] / sqrt_K[17]
    # term4 = -78586200//362880 * A[2] * B[3] / sqrt_K[13]
    # term5 = -9823275//362880 * B[1] * C[4] / sqrt_K[13]
    # term6 = -793800//362880 / sqrt_K[9] * A[3]
    # term7 = -297675//362880 * B[1] / sqrt_K[9] * w2_t[4]
    # term8 = 3572100//362880 * A[1] * C[3] / sqrt_K[11]
    # term9 = 51081030//362880 * B[5] / sqrt_K[15] * w2_t[2]
    # term10 = 170270100//362880 * B[3] * C[3] / sqrt_K[15]
    # term11 = 1240539300//362880 * C[1] * B[7] / sqrt_K[19]
    # term12 = -117879300//362880 * A[1] * B[2] * C[2] / sqrt_K[13]
    # term13 = -39293100//362880 * C[1] * B[3] / sqrt_K[13] * w2_t[2]
    # term14 = -1190700//362880 * C[1] *A[1] / sqrt_K[9] * w2_t[2]
    # term15 = 5358150//362880 * B[1] * C[2] / sqrt_K[11] * w2_t[2]
    # term16 = 10716300//362880 * A[1] * B[2] / sqrt_K[11] * w2_t[2]
    # term17 = 21432600//362880 * B[1] * C[1] * A[2] / sqrt_K[11]
    # term18 = 510810300//362880 * A[1] * C[1] * B[4] / sqrt_K[15]

    term1 = -654729075/362880 * B[9] / sqrt_K[21]
    term2 = -766215450/362880 * B[5] * C[2] / sqrt_K[17]
    term3 = -510810300/362880 * A[1] * B[6] / sqrt_K[17]
    term4 = -78586200/362880 * A[2] * B[3] / sqrt_K[13]
    term5 = -9823275/362880 * B[1] * C[4] / sqrt_K[13]
    term6 = -793800/362880 / sqrt_K[9] * A[3]
    term7 = -297675/362880 * B[1] / sqrt_K[9] * w2_t[4]
    term8 = 3572100/362880 * A[1] * C[3] / sqrt_K[11]
    term9 = 51081030/362880 * B[5] / sqrt_K[15] * w2_t[2]
    term10 = 170270100/362880 * B[3] * C[3] / sqrt_K[15]
    term11 = 1240539300/362880 * C[1] * B[7] / sqrt_K[19]
    term12 = -117879300/362880 * A[1] * B[2] * C[2] / sqrt_K[13]
    term13 = -39293100/362880 * C[1] * B[3] / sqrt_K[13] * w2_t[2]
    term14 = -1190700/362880 * C[1] *A[1] / sqrt_K[9] * w2_t[2]
    term15 = 5358150/362880 * B[1] * C[2] / sqrt_K[11] * w2_t[2]
    term16 = 10716300/362880 * A[1] * B[2] / sqrt_K[11] * w2_t[2]
    term17 = 21432600/362880 * B[1] * C[1] * A[2] / sqrt_K[11]
    term18 = 510810300/362880 * A[1] * C[1] * B[4] / sqrt_K[15]

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 +
              term9 + term10 + term11 + term12 + term13 + term14 + term15 +
              term16 + term17 + term18
end


function evalRQTaylor10adaptivekernel(   w1_t::T,
                                w2_t::Vector{T},
                                K::Vector{T},
                                sqrt_K::Vector{T},
                                B::Vector{T},
                                C::Vector{T},
                                A::Vector{T})::T where T <: Real

    term1 = -9823275/3628800 * C[5] / sqrt_K[13]
    term2 = 13749310575/3628800 * B[10] / sqrt_K[23]
    term3 = -29462808375/3628800 * C[1] * B[8] / sqrt_K[21]
    term4 = -6385128750/3628800 * B[4] * C[3] / sqrt_K[17]
    term5 = -1277025750/3628800 * B[6] / sqrt_K[17] * w2_t[2]
    term6 = -5953500/3628800 / sqrt_K[9] * A[2] * w2_t[2]
    term7 = -1488375/3628800 * C[1] / sqrt_K[9] * w2_t[4]
    term8 = 8930250/3628800 * C[3] / sqrt_K[11] * w2_t[2]
    term9 = 13395375/3628800 * B[2] / sqrt_K[11] * w2_t[4]
    term10 = 53581500/3628800 * A[2] * C[2] / sqrt_K[11]
    term11 = 638512875/3628800 * B[2] * C[4] / sqrt_K[15]
    term12 = 2554051500/3628800 * A[2] * B[4] / sqrt_K[15]
    term13 = 12405393000/3628800 * A[1] * B[7] / sqrt_K[19]
    term14 = 21709437750/3628800 * B[6] * C[2] / sqrt_K[19]
    term15 = -15324309000/3628800 * A[1] * C[1] * B[5] / sqrt_K[17]
    term16 = -1178793000/3628800 * C[1] * A[2] * B[2] / sqrt_K[13]
    term17 = -392931000/3628800 * A[1] * B[1] * C[3] / sqrt_K[13]
    term18 = -392931000/3628800 * A[1] * B[3] / sqrt_K[13] * w2_t[2]
    term19 = -294698250/3628800 * B[2] * C[2] / sqrt_K[13] * w2_t[2]
    term20 = 71442000/3628800 * B[1] / sqrt_K[11] * A[3]
    term21 = 1277025750/3628800 * C[1] * B[4] / sqrt_K[15] * w2_t[2]
    term22 = 5108103000/3628800 * A[1] * B[3] * C[2] / sqrt_K[15]
    term23 = 107163000/3628800 * B[1] * C[1] * A[1] / sqrt_K[11] * w2_t[2]

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 +
              term9 + term10 + term11 + term12 + term13 + term14 + term15 +
              term16 + term17 + term18 + term19 + term20 + term21 + term22 +
              term23
end
