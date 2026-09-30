using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra

FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. Tahdid ab'ad an-nizam wa 'adad al-madarat
nmf = 6               # 'adad madarat al-fermyon N_{mf} = 2q + 1[cite: 1]
nof = 2 * nmf         # naw'an min al-fermyonat (f0 wa f1)، majmu' al-madarat = 2 * N_{mf}[cite: 1]
nmb = nmf - 1         # 'adad madarat al-boson N_{mb} = 2q[cite: 1]
nob = 2 * nmb         # naw'an min al-bosonat (bp wa bm)، majmu' al-madarat = 2 * N_{mb}[cite: 1]

# 2. Bina' al-a'dad al-kammiya al-mahfuza (al-shuhna U(1)_e، L_z، wa S_z li-tamathul O(2))[cite: 1]
f_sz = zeros(Int64, nof)
b_sz = repeat([1, -1], nmb)
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 2) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    sz_sqnd
]

# Bina' asas al-tashkilaat li-kull qita': N_e = N_{mf}
# Li-anna +/- q mutanaksat tamaman fi al-taqa، naqoum bi-hisab sz >= 0 faqat
max_sz = 2            # yashmal sz = 0 wa sz = 1 wa sz = 2
cfs = Dict{Tuple{Int64, Int64}, SConfs}()
for lz in 0 : 1 
    for sz in 0 : max_sz
        if mod(lz, 2) == 0
            cfs[(lz, sz)] = SConfs(nof, nob, nmf, [nmf, lz, 2*sz], qnd)
        else
            cfs[(lz, sz)] = SConfs(nof, nob, nmf, [nmf, lz, Int(2*sz+1)], qnd)
        end
    end
end 

# 3. Tarkib Hamiltonian Super-Ising (bi-asas al-bosonat al-dawa'iriya bp, bm)[cite: 1]
FuzzifiED.ObsNormRadSq = Float64(nmf)
t = 1.5; U = 0.25; g = 1.0; m = 0.0793; p = 0.0615; q = 0.0662

f0 = GetFermionSObs(nmf, 2, 1)
f1 = GetFermionSObs(nmf, 2, 2)
bp = GetBosonSObs(nmb, 2, 1)  # boson b_+ (Sz = +1)
bm = GetBosonSObs(nmb, 2, 2)  # boson b_- (Sz = -1)

n0 = StoreComps(f0' * f0)
n1 = StoreComps(f1' * f1)
nbp = StoreComps(bp' * bp)
nbm = StoreComps(bm' * bm)
nb_tot = StoreComps(nbp + nbm)
nx = StoreComps(f0' * f1 + f1' * f0)
nr = StoreComps(n0 + n1 + nb_tot)

etap = StoreComps(f0' * bp)
etam = StoreComps(f0' * bm)
@assert abs(etap.s2) == 1 && abs(etam.s2) == 1

R = sqrt(nmf)
Detap = let e = etap, R = R
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

Detam = let e = etam, R = R
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

pair_obs = etap * Detam + etam * Detap
@assert pair_obs.s2 == 0

tms_hop = SimplifyTerms(GetIntegral(pair_obs))

tms_int = SimplifyTerms(
    [(GetIntegral(nr * nr));
        t * tms_hop; t * tms_hop';
        U * (GetIntegral(nx * Laplacian(nx)));
        g * (GetIntegral(nx * nb_tot))],
)

tms_hmt = SimplifyTerms(
    tms_int
    - q * GetIntegral(nx) - m * GetIntegral(n1) - p * GetIntegral(nb_tot),
)

tms_l2 = GetL2STerms(nmf, 2, nmb, 2) 

# 4. Tarkib mu'athir Casimir li-tamathul O(2)
function on_C2_terms(nmb::Int, N::Int)
    @assert N >= 2
    U = Matrix{ComplexF64}(I, N, N)
    for k in 1:div(N, 2)
        i = 2k - 1
        j = 2k
        U[i:j, i:j] = ComplexF64[ 1 -im; 1 im ] / sqrt(2)
    end

    terms = STerm[]
    for a in 1:(N-1), b in (a+1):N
        M = zeros(ComplexF64, N, N)
        M[a, b] =  1
        M[b, a] = -1
        Mc = U * M * U'
        Aab = GetBosonPolSTerms(nmb, N, Mc)
        append!(terms, -(Aab * Aab))
    end
    return SimplifyTerms(terms)
end

N = 2
tms_c2 = on_C2_terms(nmb, N)

# 5. Hall al-qiyam al-dhatiya (ED) wa qiyas al-a'dad al-kammiya[cite: 1]
result = []
n_states = 12  # 'adad al-halat al-matluba fi kull qita'[cite: 1]

for (lz, sz) in sort(collect(keys(cfs)))
    bs = SBasis(cfs[(lz, sz)])
    hmt = SOperator(bs, tms_hmt)
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, n_states)

    # Qiyas L^2
    l2 = SOperator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]

    # Qiyas mu'athir Casimir C_2 li-O(2)
    c2_op = SOperator(bs, tms_c2)
    c2_mat = OpMat(c2_op)
    c2_val = [ real(st[:, i]' * (c2_mat * st[:, i])) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        push!(result, [enrg[i], l2_val[i], c2_val[i], lz, sz])
    end
end

# 6. Tartib al-tayf، ma'ayarat enrg_T wa tibaa'at al-nata'ij bi-kamil[cite: 1]
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  # taqat al-faragh (Delta = 0)[cite: 1]

# Murashahat mu'athir al-juhd T (L^2 ≈ 6, C_2 ≈ 0)[cite: 1]
T_candidates = filter(st -> abs(st[2] - 6.0) < 0.1 && abs(st[3] - 0.0) < 0.1, result)

if isempty(T_candidates)
    println("Tanbih: Lam yatimm al-'uthur 'ala murashah li-mu'athir T!")
    enrg_T = result[end][1]
else
    if length(T_candidates) >= 2
        ratio = (T_candidates[1][1] - enrg_0) / (T_candidates[2][1] - enrg_0)
        # Nisbat al-fariq al-nazari: 2.584 / 3.000 ≈ 0.861[cite: 1]
        if 0.75 < ratio < 0.95
            enrg_T = T_candidates[2][1]
        else
            enrg_T = T_candidates[1][1]
        end
    else
        enrg_T = T_candidates[1][1]
    end
end

spec = [
    [
        round(3 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 6),
        round(st[1], digits = 6),
        round(st[2], digits = 4),
        round(Int, round(st[3])),
        st[4],
        st[5]
    ] for st in result
]

# Tibaa'at jami' al-asntur duna ikhtisar
println("="^88)
println("Tayf al-taqa al-kamel (Majmu' al-halat: $(length(spec))):")
println("     Delta         Taqa_E        L^2           C2_O2       2*Lz      Sz")
println("-"^88)
for row in spec
    println(
        rpad(string(round(row[1], digits=6)), 14),
        rpad(string(round(row[2], digits=6)), 14),
        rpad(string(round(row[3], digits=4)), 14),
        rpad(string(row[4]), 12),
        rpad(string(row[5]), 10),
        rpad(string(row[6]), 8)
    )
end
println("="^88)
