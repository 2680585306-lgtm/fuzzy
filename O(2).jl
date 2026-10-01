using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸与单粒子轨道数[cite: 1]
nmf = 6               # 费米子单粒子轨道数 N_{mf} = 2q + 1[cite: 1]
nof = 2 * nmf         # 2 个费米子 Flavor (f0 与 f1)，总费米子轨道数为 2 * N_{mf}[cite: 1]
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q (角动量与费米子相差 1/2)[cite: 1]
nob = 2 * nmb         # 2 个玻色子 Flavor (bp 与 bm，总轨道数为 2 * N_{mb})[cite: 1]

# 2. 构造守恒量子数 (电荷 U(1)_e、角动量 L_z 以及 O(2) 内部生成元 S_z)[cite: 1]
# 费米子荷为 0；玻色子采用手征基底: bp 荷为 +1，bm 荷为 -1
f_sz = zeros(Int64, nof)
# FuzzifiED 内部多 flavor 轨道为交错排列 (interleaved)，因此对每个空间轨道 m 交替赋予 +1 和 -1
b_sz = repeat([1, -1], nmb)
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 2) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    sz_sqnd
]

# 生成指定扇区的基底构型：总电荷数 N_e = N_{mf}
# 由于 +/- q 严格简并，只求解 sz >= 0 扇区
max_sz = 2            # 低能态主要分布在 sz = 0 (标量单态) 和 sz = 1 (双重态)；如需更高激发可设为 2
cfs = Dict{Tuple{Int64, Int64}, SConfs}()
for lz in 0 : 1 
    for sz in 0 : max_sz
        if mod(lz, 2) ==0
            cfs[(lz, 2sz)] = SConfs(nof, nob, nmf, [nmf, lz, 2*sz], qnd)
        else
            cfs[(lz, 2sz+1)] = SConfs(nof, nob, nmf, [nmf, lz, Int(2*sz+1)], qnd)
        end
    end
end 

# 3. 组装 Super-Ising 哈密顿量 (手征玻色子基底 bp, bm)[cite: 1]
FuzzifiED.ObsNormRadSq = Float64(nmf)
t = 1.5; U = 0.25; g = 1.0; m = 0.151901; p = 0.153822; q = 0.09567

f0 = GetFermionSObs(nmf, 2, 1)
f1 = GetFermionSObs(nmf, 2, 2)
bp = GetBosonSObs(nmb, 2, 1)  # 手征玻色子 b_+ (Sz = +1)
bm = GetBosonSObs(nmb, 2, 2)  # 手征玻色子 b_- (Sz = -1)

n0 = StoreComps(f0' * f0)
n1 = StoreComps(f1' * f1)
nbp = StoreComps(bp' * bp)
nbm = StoreComps(bm' * bm)
nb_tot = StoreComps(nbp + nbm)
nx = StoreComps(f0' * f1 + f1' * f0)
nr = StoreComps(n0 + n1 + nb_tot)

etap = StoreComps(f0' * bp)
etam = StoreComps(f0' * bm)
@assert abs(etap.s2) == 1 && abs(etam.s2) == 1 "eta must have spin weight +/-1/2"

R = sqrt(nmf)
Detap = let e = etap, R = R
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

Detam = let e = etam, R = R
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

# 在手征基底下，pair_obs 对应原 b0^2 + b1^2 的对跃迁项
pair_obs = etap * Detam + etam * Detap
@assert pair_obs.s2 == 0 "Pair conversion must be a rotational scalar"

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

# 4. 构造 O(2) 生成元算符 
function on_C2_terms( nmb::Int, N::Int;)

    @assert N >= 2

    # ------------------------------------------------------------
    # Transformation from real O(N) basis to the circular basis
    #
    # c = U b
    #
    # for every pair:
    #   c_+ = (b_1 - i b_2)/sqrt(2)
    #   c_- = (b_1 + i b_2)/sqrt(2)
    #
    # For odd N, the last flavour remains unchanged.
    # ------------------------------------------------------------

    U = Matrix{ComplexF64}(I, N, N)
    for k in 1:div(N, 2)
        i = 2k - 1
        j = 2k
        U[i:j, i:j] =
        ComplexF64[
                    1  -im
                    1   im
                ] / sqrt(2)
    end

    # ------------------------------------------------------------
    # C2 = - sum_{a<b} A_ab^2
    # ------------------------------------------------------------

    terms = STerm[]

    for a in 1:(N-1), b in (a+1):N

        # Real antisymmetric generator
        M = zeros(ComplexF64, N, N)

        M[a, b] =  1
        M[b, a] = -1

        # If c = U b, then
        #
        #     b† M b = c† (U M U†) c
        #
        Mc = U * M * U'

        Aab = GetBosonPolSTerms(nmb, N, Mc)

        # Q_ab = -i A_ab
        #
        # therefore
        #
        # Q_ab^2 = - A_ab^2
        append!(terms, -(Aab * Aab))
    end

    tms_c2 = SimplifyTerms(terms)

    return SimplifyTerms(tms_c2)
end
# N=2 
N=2
tms_c2 = on_C2_terms(nmb, N)
# 5. 精确对角化 (ED) 求解低能态并测量量子数
result = []
n_states = 10  # 单扇区求解 10 个态[cite: 1]

for (lz, sz) in sort(collect(keys(cfs)))
    @show lz, sz
    bs = SBasis(cfs[(lz, sz)])
    hmt = SOperator(bs, tms_hmt)
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, n_states)

    # 测量总角动量 L^2
    l2 = SOperator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]

    # 在该扇区内所有构型均具有确定的 Sz = sz，其 Casimir C_O(2) = Sz^2 严格等于 sz^2
    #c_val = fill(Float64(sz^2), length(enrg))
    c2_op = SOperator(bs, tms_c2)
    c2_mat = OpMat(c2_op)
    c2_val = [ real(st[:, i]' * c2_mat * st[:, i]) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        push!(result, [enrg[i], l2_val[i], c2_val[i], lz, sz])
    end
end

# 6. 能谱排序与定标输出
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  # 真空能量 (Δ = 0)[cite: 1]
enrg_T = filter(st -> abs(st[2]-6)<1e-4 && abs(st[3]-0)<1e-4, result)[1][1]
spec = [([3 * (st[1] - enrg_0) / (enrg_T - enrg_0); st]) for st in result]
display(permutedims(hcat(spec...)))


#= # 获取第一个 L^2 ≈ 6 的态 (能动张量 T，位于 sz = 0 扇区)[cite: 1]
T_candidates = filter(st -> abs(st[2] - 6) < 1e-3, result)
if !isempty(T_candidates)
    enrg_T = T_candidates[1][1]
    println("="^85)
    println("成功获取能动张量 T: 能量 = $(round(enrg_T, digits=6)), L^2 = $(round(T_candidates[1][2], digits=4))[cite: 1]")
    println("输出说明: [标度维数 Δ, 原始能量 E, 角动量 L^2, O(2) Casimir, 2*Lz, Sz, 对应原代码简并度]")
    println("-"^85)
    spec = [ 
        [
            round(3 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 6), 
            round(st[1], digits = 6), 
            round(st[2], digits = 4), 
            round(Int, st[3]), 
            st[4],
            st[5],
            st[5] == 0 ? 1 : 2
        ] for st in result 
    ]
    display(permutedims(hcat(spec...)))
else
    println("未找到 L^2 = 6 的态，输出原始测量数据 [E, L^2, C_O(2), 2*Lz, Sz]：")
    display(result)
end =#
