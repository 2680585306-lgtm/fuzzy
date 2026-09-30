using FuzzifiED
using FuzzifiED.Fuzzifino

FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸与单粒子轨道数
nmf = 6               # 费米子单粒子轨道数 N_{mf} = 2q + 1[cite: 1]
nof = 2 * nmf         # 2 个费米子 Flavor (f0 与 f1)，总费米子轨道数为 2 * N_{mf}[cite: 1]
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q (角动量与费米子相差 1/2)[cite: 1]
nob = 2 * nmb         # 2 个玻色子 Flavor (bp 与 bm，总轨道数为 2 * N_{mb})[cite: 1]

# 2. 构造守恒量子数 (电荷 U(1)_e, 角动量 2*L_z, 以及 O(2) 内部生成元 S_z)
# 费米子荷 0；玻色子 bp 荷 +1，bm 荷 -1
f_sz = zeros(Int64, nof)
b_sz = [fill(1, nmb); fill(-1, nmb)]
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 2) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    sz_sqnd
]

# 3. 生成指定扇区的基底构型
# 由于 +/- q 严格简并，只计算 sz >= 0 (低能态主要位于 sz = 0 与 sz = 1)
max_sz = 1            # 需算更高激发可设为 2
cfs = Dict{Tuple{Int64, Int64}, SConfs}()
for lz in 0 : 1 
    for sz in 0 : max_sz
        cfs[(lz, sz)] = SConfs(nof, nob, nmf, [nmf, lz, sz], qnd)
    end
end 

# 4. 组装 Super-Ising 哈密顿量 (手征玻色子基底，与原哈密顿量代数等价)
FuzzifiED.ObsNormRadSq = Float64(nmf)
t = 1.5; U = 0.25; g = 1.0; m = 0.0793; p = 0.0615; q = 0.0662

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

# 导出手征配对算符
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

# eta1*D(eta1) + eta2*D(eta2) 严格恒等于 etap*Detam + etam*Detap
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

# 构造 O(2) 内部生成元算符: Sz = N_{b+} - N_{b-}
sz_obs = StoreComps(nbp - nbm)
tms_sz = GetIntegral(sz_obs)

# 5. 精确对角化 (ED) 求解并测量 O(2) Casimir 算符
result = []
n_states = 15  # 单扇区求解 15 个态，足以覆盖低能区且精确捕获 L^2=6 的态

for (lz, sz) in sort(collect(keys(cfs)))
    bs = SBasis(cfs[(lz, sz)])
    hmt = SOperator(bs, tms_hmt)
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, n_states)

    # 测量总角动量 L^2
    l2 = SOperator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]

    # 测量 O(2) Casimir 算符: C = Sz^2
    sz_op = SOperator(bs, tms_sz)
    sz_mat = OpMat(sz_op)
    c_val = [ real(st[:, i]' * (sz_mat * (sz_mat * st[:, i]))) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        # 记录: [能量, L^2, C_O(2), L_z, S_z]
        push!(result, [enrg[i], l2_val[i], c_val[i], lz, sz])
    end
end

# 6. 能谱排序与定标输出
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  # 真空基态能量 (Δ = 0)[cite: 1]

# 筛选 L^2 ≈ 6 且处于单态标量扇区 (Sz = 0, C_O(2) = 0) 的态
# 容差设为 0.1 以适应 Nmf=6 小尺寸的微小偏移
T_candidates = filter(st -> abs(st[2] - 6) < 0.1 && abs(st[3]) < 0.1, result)

if !isempty(T_candidates)
    # 取该扇区满足条件的态作为 T 候选进行定标校准[cite: 1]
    enrg_T = T_candidates[1][1]
    println("="^80)
    println("成功定位 L^2 ≈ 6 态，定标能量 enrg_T = $(round(enrg_T, digits=6))")
    println("输出格式说明: [标度维数 Δ, 能量 E, L^2, O(2) Casimir, 2*Lz, Sz, 原代码对应简并度]")
    println("-"^80)
    
    spec = [ 
        [
            round(3 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 4), 
            round(st[1], digits = 6), 
            round(st[2], digits = 4), 
            round(Int, round(st[3])), 
            st[4], 
            st[5],
            st[5] == 0 ? 1 : 2  # Sz=0 在原代码为 1 重；Sz>0 对应原代码 +/-q 的 2 重简并
        ] for st in result 
    ]
    display(permutedims(hcat(spec...)))
else
    println("未找到 L^2 ≈ 6 态，输出原始能谱数据 [E, L^2, C_O(2), 2*Lz, Sz]：")
    display(result)
end
