# ==============================================================================
# GNY 临界点参数极其稳健的寻优与全谱计算脚本
# 策略: 严格遵循 Gao et al. 做法，仅使用绝对底层的 3 个态 (ψ, ∂σ, ∂ψ) 进行调优
# ==============================================================================

using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using Optim

FuzzifiED.SilentStd = true
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < eps(Float32)

# 1. 设定系统尺寸与单粒子轨道数
nmf = 6               # 费米子单粒子轨道数 N_{mf} = 2q + 1
nof = 2 * nmf         # 2 个费米子 Flavor (f0, f1)，总费米子轨道数为 2 * N_{mf}
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q
nob = 2 * nmb         # 2 个玻色子 Flavor (bp, bm)

# 2. 构造守恒量子数 (U(1)_e, 2*L_z, S_z)
f_sz = zeros(Int64, nof)
b_sz = repeat([1, -1], nmb)
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 2) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    sz_sqnd
]

# 预生成核心扇区 (仅用于优化)
# 键名严格定义为 (2Lz, Sz)
opt_cfs   = Dict{Tuple{Int64, Int64}, SConfs}()
opt_bases = Dict{Tuple{Int64, Int64}, SBasis}()
l2_mats_opt = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()

# (0, 0) 扇区: 包含 真空, σ, ∂σ
opt_cfs[(0, 0)]   = SConfs(nof, nob, nmf, [nmf, 0, 0], qnd)
opt_bases[(0, 0)] = SBasis(opt_cfs[(0, 0)])

# (1, 1) 扇区: 包含 ψ, ∂ψ
opt_cfs[(1, 1)]   = SConfs(nof, nob, nmf, [nmf, 1, 1], qnd)
opt_bases[(1, 1)] = SBasis(opt_cfs[(1, 1)])

tms_l2 = GetL2STerms(nmf, 2, nmb, 2)
for k in [(0, 0), (1, 1)]
    l2_mats_opt[k] = OpMat(SOperator(opt_bases[k], tms_l2))
end

# 3. 预组装常数项微观相互作用项
FuzzifiED.ObsNormRadSq = Float64(nmf)
t_hop = 1.5; U_int = 0.25; g_cpl = 1.0

f0 = GetFermionSObs(nmf, 2, 1)
f1 = GetFermionSObs(nmf, 2, 2)
bp = GetBosonSObs(nmb, 2, 1)
bm = GetBosonSObs(nmb, 2, 2)

n0     = StoreComps(f0' * f0)
n1     = StoreComps(f1' * f1)
nbp    = StoreComps(bp' * bp)
nbm    = StoreComps(bm' * bm)
nb_tot = StoreComps(nbp + nbm)
nx     = StoreComps(f0' * f1 + f1' * f0)
nr     = StoreComps(n0 + n1 + nb_tot)

etap = StoreComps(f0' * bp)
etam = StoreComps(f0' * bm)
R = sqrt(nmf)
Detap = let e = etap, R = R
    SSphereObs(-e.s2, e.l2m, (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end
Detam = let e = etam, R = R
    SSphereObs(-e.s2, e.l2m, (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

pair_obs = etap * Detam + etam * Detap
tms_hop  = SimplifyTerms(GetIntegral(pair_obs))

tms_base = SimplifyTerms([
    GetIntegral(nr * nr);
    t_hop * tms_hop; t_hop * tms_hop';
    U_int * GetIntegral(nx * Laplacian(nx));
    g_cpl * GetIntegral(nx * nb_tot)
])

tms_nx = SimplifyTerms(GetIntegral(nx))
tms_n1 = SimplifyTerms(GetIntegral(n1))
tms_nb = SimplifyTerms(GetIntegral(nb_tot))

# 4. 稳健的极简代价函数 (仅拟合底层态，绝不交叉)
function cost(params)
    q_val, m_val, p_val = params
    tms_hmt = [tms_base; -q_val * tms_nx; -m_val * tms_n1; -p_val * tms_nb]

    try
        # 扇区(0,0): 仅需解前 6 个态，绝对包含真空 I、标量 σ 和一阶导数 ∂σ
        hmt_00 = SOperator(opt_bases[(0, 0)], tms_hmt)
        enrg_00, st_00 = GetEigensystem(OpMat(hmt_00), 6)
        l2_00 = [ real(st_00[:, i]' * (l2_mats_opt[(0, 0)] * st_00[:, i])) for i in eachindex(enrg_00) ]

        # 扇区(1,1): 仅需解前 6 个态，绝对包含 ψ 和 ∂ψ
        hmt_11 = SOperator(opt_bases[(1, 1)], tms_hmt)
        enrg_11, st_11 = GetEigensystem(OpMat(hmt_11), 6)
        l2_11 = [ real(st_11[:, i]' * (l2_mats_opt[(1, 1)] * st_11[:, i])) for i in eachindex(enrg_11) ]

        idx_l0 = filter(i -> abs(l2_00[i]) < 0.25, 1:length(enrg_00))
        idx_l1 = filter(i -> abs(l2_00[i] - 2.0) < 0.25, 1:length(enrg_00))

        (length(idx_l0) < 2 || isempty(idx_l1)) && return 100.0

        E_0  = enrg_00[idx_l0[1]]  # 保证是真空态
        E_σ  = enrg_00[idx_l0[2]]  # 保证是 σ
        E_∂σ = enrg_00[idx_l1[1]]  # 保证是 ∂σ

        idx_fhalf  = filter(i -> abs(l2_11[i] - 0.75) < 0.25, 1:length(enrg_11))
        idx_f3half = filter(i -> abs(l2_11[i] - 3.75) < 0.25, 1:length(enrg_11))

        (isempty(idx_fhalf) || isempty(idx_f3half)) && return 100.0

        E_ψ  = enrg_11[idx_fhalf[1]]   # 保证是 ψ
        E_∂ψ = enrg_11[idx_f3half[1]]  # 保证是 ∂ψ

        dE_σ = E_σ - E_0
        dE_σ <= 1e-4 && return 100.0

        # 以 σ 定标 (Δ_σ ≡ 0.66)
        v_inv = 0.66 / dE_σ
        Δ_ψ   = (E_ψ - E_0) * v_inv
        Δ_∂σ  = (E_∂σ - E_0) * v_inv
        Δ_∂ψ  = (E_∂ψ - E_0) * v_inv

        # 仅拟合这 3 个绝对坚固的底层态
        χ2 = (Δ_ψ - 1.07)^2 + (Δ_∂σ - 1.66)^2 + (Δ_∂ψ - 2.07)^2
        return χ2
    catch
        return 100.0
    end
end

# 5. 执行 Nelder-Mead 寻优
# 强制让算法从一个合理的微观区域开始搜寻，而非你代码中崩溃的废点
init_params = [0.066, 0.079, 0.061]

println("="^88)
println("正在仅利用最底层稳健激发态 (ψ, ∂σ, ∂ψ) 执行高精度调优...")
println("="^88)

opt_res = optimize(
    cost,
    init_params,
    NelderMead(),
    Optim.Options(
        iterations = 80,
        g_tol = 1e-6,
        show_trace = false
    )
)

best_params = Optim.minimizer(opt_res)
println("优化完成！参数已拉回物理区域：")
println("最佳参数 [q, m, p] = ", round.(best_params, digits=6))
println("最小代价函数 Cost = ", round(Optim.minimum(opt_res), digits=8))
println("="^88)

# 6. 使用最优参数扩展计算全谱
max_sz = 2
cfs_all   = Dict{Tuple{Int64, Int64}, SConfs}()
bases_all = Dict{Tuple{Int64, Int64}, SBasis}()
l2_mats_all = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()

# 键名定义规则: (2Lz, Sz)，确保逻辑统一
for lz in 0 : 1 
    for sz in 0 : max_sz
        Sz_val = (mod(lz, 2) == 0) ? 2*sz : 2*sz + 1
        k = (lz, Sz_val)
        cfs_all[k] = SConfs(nof, nob, nmf, [nmf, lz, Sz_val], qnd)
        bases_all[k] = SBasis(cfs_all[k])
        l2_mats_all[k] = OpMat(SOperator(bases_all[k], tms_l2))
    end
end

q_opt, m_opt, p_opt = best_params
tms_hmt_best = SimplifyTerms([
    tms_base;
    -q_opt * tms_nx;
    -m_opt * tms_n1;
    -p_opt * tms_nb
])

full_result = []
n_states_full = 20  # 充分扩大各扇区求解态数

for (k, bs) in bases_all
    lz, Sz_val = k
    hmt = SOperator(bs, tms_hmt_best)
    enrg, st = GetEigensystem(OpMat(hmt), n_states_full)
    l2_mat = l2_mats_all[k]
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        push!(full_result, [enrg[i], l2_val[i], lz, Sz_val])
    end
end

sort!(full_result, by = st -> real(st[1]))
enrg_0 = full_result[1][1]  # 真空能量 (Δ = 0)

# 提取关键定标点
enrg_σ = filter(st -> abs(st[2]) < 0.25 && st[4] == 0 && st[1] > enrg_0 + 1e-4, full_result)[1][1]

# 在 L^2 ≈ 6, Sz = 0 中严格锁定第 2 个态作为能动张量 T
T_cands = filter(st -> abs(st[2] - 6.0) < 0.35 && st[4] == 0, full_result)
enrg_T = length(T_cands) >= 2 ? T_cands[2][1] : T_cands[1][1]

# 动态算符标签匹配函数
function identify_operator(L2, q, Delta)
    if abs(L2) < 0.2 && q == 0 && Delta < 0.1
        return "真空基态 I (Δ=0)"
    elseif abs(L2) < 0.2 && q == 0 && abs(Delta - 0.66) < 0.25
        return "σ (Table I: 0.66)"
    elseif abs(L2 - 0.75) < 0.2 && q == 1 && abs(Delta - 1.07) < 0.25
        return "ψ (Table I: 1.07)"
    elseif abs(L2 - 2.0) < 0.25 && q == 0 && abs(Delta - 1.66) < 0.25
        return "∂_μ σ (Table I: 1.66)"
    elseif abs(L2 - 2.0) < 0.25 && q == 0 && abs(Delta - 2.00) < 0.25
        return "J_μ (守恒流, Table I: 2.00)"
    elseif abs(L2 - 3.75) < 0.25 && q == 1 && abs(Delta - 2.07) < 0.25
        return "∂_μ ψ (Table I: 2.07)"
    elseif abs(L2) < 0.2 && q == 0 && abs(Delta - 2.14) < 0.35
        return "ϵ (标量初级, Table I: 2.14)"
    elseif abs(L2 - 2.0) < 0.25 && q == 0 && abs(Delta - 2.35) < 0.20
        return "J̃_μ (非守恒流, Table I: 2.31)"
    elseif abs(L2) < 0.2 && q == 2 && abs(Delta - 2.45) < 0.40
        return "σ_T (单态配对, Table I: 2.45)"
    elseif abs(L2) < 0.2 && q == 0 && abs(Delta - 2.66) < 0.35
        return "∂² σ (标量次级, Table I: 2.66)"
    elseif abs(L2 - 6.0) < 0.35 && q == 0 && Delta < 2.85
        return "∂_μ ∂_ν σ (张量次级, Table I: 2.66)"
    elseif abs(L2 - 6.0) < 0.35 && q == 0 && Delta >= 2.85
        return "T_μν (能动张量, Table I: 3.00)"
    elseif abs(L2 - 6.0) < 0.35 && q == 2 && abs(Delta - 3.03) < 0.30
        return "D_μν (p波配对流, Table I: 3.03)"
    elseif abs(L2 - 8.75) < 0.35 && q == 1 && abs(Delta - 3.07) < 0.35
        return "∂_μ ∂_ν ψ (旋量次级, Table I: 3.07)"
    elseif abs(L2 - 2.0) < 0.25 && q == 2 && abs(Delta - 3.45) < 0.40
        return "∂_μ σ_T (配对次级, Table I: 3.45)"
    else
        return ""
    end
end

spec = [
    [
        # 第 1 列：以 σ 定标 (Table I 官方标准, Δ_σ ≡ 0.66)
        round(0.66 * (st[1] - enrg_0) / (enrg_σ - enrg_0), digits = 4),
        # 第 2 列：以 T 定标 (对照用, Δ_T ≡ 3.00)
        round(3.00 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 4),
        round(st[1], digits = 6),
        round(st[2], digits = 4),
        st[3],
        st[4],
        identify_operator(st[2], abs(st[4]), 0.66 * (st[1] - enrg_0) / (enrg_σ - enrg_0))
    ] for st in full_result
]

n_print = min(35, length(spec))
println("="^100)
println("纯净无混叠共形算符谱 (共列出前 $n_print 项)：")
println("  Δ(σ定标)   Δ(T定标)      能量 E        角动量 L^2      2*Lz      Sz      Table I 对应算符态")
println("-"^100)
for i in 1:n_print
    row = spec[i]
    println(
        rpad(string(row[1]), 12),
        rpad(string(row[2]), 12),
        rpad(string(row[3]), 14),
        rpad(string(row[4]), 14),
        rpad(string(row[5]), 10),
        rpad(string(row[6]), 8),
        row[7]
    )
end
println("="^100)
