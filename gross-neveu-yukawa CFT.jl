
using FuzzifiED
using FuzzifiED.Fuzzifino

FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸与单粒子轨道数[cite: 1]
nmf = 6               # 费米子单粒子轨道数 N_{mf} = 2q + 1
nof = 2 * nmf         # 2 个费米子 Flavor (f0 与 f1)，总费米子轨道数为 2 * N_{mf}
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q (角动量与费米子相差 1/2)
nob = 2 * nmb         # 2 个玻色子 Flavor (nob 设置为 2 * nmb)

# 2. 构造守恒量子数 (电荷 U(1)_e 与角动量 L_z)[cite: 1]

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 2) + SQNDiag(GetLz2QNDiag(nmf, 2), nob)
]

# 生成指定扇区的基底构型：总电荷数 N_e = N_{mf}，分别求解 L_z = 0 和 L_z = 1/2
cfs = Dict{Int64, SConfs}()
for lz = 0 : 1 
    cfs[lz] = SConfs(nof, nob, nmf, [nmf, lz], qnd)
end 

# 5. 组装 Super-Ising 哈密顿量 
FuzzifiED.ObsNormRadSq = Float64(nmf)
t = 1.5; U = 0.25; g = 1.0; m = 0.0793; p = 0.0615; q = 0.0662

f0 = GetFermionSObs(nmf, 2, 1)
f1 = GetFermionSObs(nmf, 2, 2)
b0 = GetBosonSObs(nmb, 2, 1)
b1 = GetBosonSObs(nmb, 2, 2)

n0 = StoreComps(f0' * f0)
n1 = StoreComps(f1' * f1)
nb0 = StoreComps(b0' * b0)
nb1 = StoreComps(b1' * b1)
nx = StoreComps(f0' * f1 + f1' * f0)
nr = StoreComps(n0 + n1 + nb0 + nb1)

# 分别导出 eta1 和 eta2 对应的导数算符
eta1 = StoreComps(f0' * b0)
@assert abs(eta1.s2) == 1 "eta1 must have spin weight +/-1/2"
Deta1 = let e = eta1, R = sqrt(nmf)
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

eta2 = StoreComps(f0' * b1)
@assert abs(eta2.s2) == 1 "eta2 must have spin weight +/-1/2"
Deta2 = let e = eta2, R = sqrt(nmf)
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

pair_obs1 = eta1 * Deta1
pair_obs2 = eta2 * Deta2
@assert pair_obs1.s2 == 0 "Pair conversion must be a rotational scalar"
@assert pair_obs2.s2 == 0 "Pair conversion must be a rotational scalar"

tms_hop1 = SimplifyTerms((GetIntegral(pair_obs1)))
tms_hop2 = SimplifyTerms((GetIntegral(pair_obs2)))

tms_int = SimplifyTerms(
    [(GetIntegral(nr * nr));
        t * tms_hop1; t * tms_hop1';
        t * tms_hop2; t * tms_hop2';
        U * (GetIntegral(nx * Laplacian(nx)));
        g * (GetIntegral(nx * nb0));       # 补充缺失的逗号
        g * (GetIntegral(nx * nb1))],
)

tms_hmt = SimplifyTerms(
    tms_int
    - q * GetIntegral(nx) - m * GetIntegral(n1) - p * GetIntegral(nb0) - p * GetIntegral(nb1),
)


tms_l2 = GetL2STerms(nmf, 2, nmb, 2) 

# 6. 精确对角化 (ED) 求解低能态[cite: 1]
result = []
for lz = 0 : 1
    bs = SBasis(cfs[lz])
    hmt = SOperator(bs, tms_hmt)
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, 10)

    l2 = SOperator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ st[:, i]' * l2_mat * st[:, i] for i in eachindex(enrg) ] 

    for i in eachindex(enrg)
        push!(result, [enrg[i], l2_val[i]])
    end
end

# 7. 能谱排序与输出
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  # 真空能量 (Δ = 0)[cite: 1]

# 安全获取第一个 L^2 ≈ 6 (即 L=2) 的态
T_candidates = filter(st -> abs(st[2] - 6) < 1e-3, result)
if !isempty(T_candidates)
    enrg_T = T_candidates[1][1]
    spec = [ round.([ 3 * (st[1] - enrg_0) / (enrg_T - enrg_0); st] .+ √eps(Float64), digits = 6) for st in result ] 
    display(permutedims(hcat(spec...)))
else
    println("未找到 L^2 = 6 的态，输出原始能量与角动量：")
    display(result)
end
