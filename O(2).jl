using FuzzifiED
using FuzzifiED.Fuzzifino

FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸与单粒子轨道数
nmf = 6               # 费米子单粒子轨道数 N_{mf} = 2q + 1
nof = 2 * nmf         # 2 个费米子 Flavor (f0 与 f1)
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q
nob = 2 * nmb         # 2 个玻色子 Flavor (手征基底 bp 与 bm)

# 2. 构造守恒量子数 (电荷 N_e, 角动量 2L_z, 以及对角化的 O(2) 荷 S_z)
# 费米子 S_z = 0; bp 为 +1; bm 为 -1
qnd_sz = SQNDiag("Sz", zeros(Int64, nof), [fill(1, nmb); fill(-1, nmb)])

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 2) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    qnd_sz
]

# 3. 组装 Super-Ising 哈密顿量
FuzzifiED.ObsNormRadSq = Float64(nmf)
t = 1.5; U = 0.25; g = 1.0; m = 0.0793; p = 0.0615; q = 0.0662

f0 = GetFermionSObs(nmf, 2, 1)
f1 = GetFermionSObs(nmf, 2, 2)
bp = GetBosonSObs(nmb, 2, 1)   # b+ (Sz = +1)
bm = GetBosonSObs(nmb, 2, 2)   # b- (Sz = -1)

n0 = StoreComps(f0' * f0)
n1 = StoreComps(f1' * f1)
nbp = StoreComps(bp' * bp)
nbm = StoreComps(bm' * bm)
nb_tot = StoreComps(nbp + nbm)

nx = StoreComps(f0' * f1 + f1' * f0)
nr = StoreComps(n0 + n1 + nb_tot)

eta_p = StoreComps(f0' * bp)
@assert abs(eta_p.s2) == 1 "eta_p must have spin weight +/-1/2"
Deta_p = let e = eta_p, R = sqrt(nmf)
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

eta_m = StoreComps(f0' * bm)
@assert abs(eta_m.s2) == 1 "eta_m must have spin weight +/-1/2"
Deta_m = let e = eta_m, R = sqrt(nmf)
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

pair_obs = StoreComps(eta_p * Deta_m + eta_m * Deta_p)
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

# 测量算符：L^2
tms_l2 = GetL2STerms(nmf, 2, nmb, 2)

# 4. 精确对角化 (计算足够多的低能态确保覆盖 T 态)
result = []
lz_sectors = [0, 1]   # 2L_z = 0 (玻色型/整数自旋), 2L_z = 1 (费米型/半整数自旋)
sz_sectors = [0, 1, 2] # 对应 O(2) Casimir C_2 = 0, 1, 4

for lz in lz_sectors
    for sz in sz_sectors
        cfs = SConfs(nof, nob, nmf, [nmf, lz, sz], qnd)
        bs = SBasis(cfs)
        if bs.dim == 0
            continue
        end

        hmt_mat = OpMat(SOperator(bs, tms_hmt))
        
        
        nev = min(16, bs.dim)
        enrg, st = GetEigensystem(hmt_mat, nev)

        l2_mat = OpMat(SOperator(bs, tms_l2))
        l2_vals = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]

        # 该扇区构型本征值严格为 sz，其 Casimir C_2 = sz^2
        c2_val = sz^2

        for i in eachindex(enrg)
            # 记录: [能量 E, L^2, C_2, Sz, 2Lz]
            push!(result, [enrg[i], l2_vals[i], c2_val, sz, lz])
        end
    end
end


sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  # 基态真空能量 (Δ = 0)

# 应力张量 T 必须满足：
# 1. 角动量 L = 2 (L^2 ≈ 6, 放宽容差至 0.4)
# 2. O(2) 标量 (Sz = 0, C_2 = 0)
# 3. 整数自旋玻色态 (2L_z 必为偶数，即 2Lz = 0)
T_candidates = filter(st -> abs(st[2] - 6.0) < 0.4 && st[4] == 0 && st[5] == 0, result)

if !isempty(T_candidates)
    enrg_T = T_candidates[1][1]
    println("成功匹配到应力张量 T！E_0 = $(round(enrg_0, digits=5)), E_T = $(round(enrg_T, digits=5))\n")
    println("激发态谱 [标度维数 Δ, 能量 E, L^2, O(2) Casimir, |Sz|, 2Lz]:")
    
    spec = [
        [
            round(3 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 4),
            round(st[1], digits = 4),
            round(st[2], digits = 3),
            round(Int, st[3]),
            st[4],
            st[5]
        ] for st in result
    ]
    display(permutedims(hcat(spec...)))
else
    println("警告：仍未自动锁定 T 态。以下为 (2Lz=0, Sz=0) 扇区计算出的所有低能态，供排查实际 L^2 分布：")
    diag_states = filter(st -> st[4] == 0 && st[5] == 0, result)
    diag_table = [
        [
            round(st[1], digits = 4),
            round(st[2], digits = 3),
            st[4],
            st[5]
        ] for st in diag_states
    ]
    println("[能量 E, 测得 L^2, Sz, 2Lz]:")
    display(permutedims(hcat(diag_table...)))
end
