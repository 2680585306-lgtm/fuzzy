# ==============================================================================
# 3D Super-Ising (SCFT) 能谱寻优与共形塔自动标记 (nmf = 6)
# ==============================================================================

using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using Optim

FuzzifiED.SilentStd = true
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸与单粒子轨道数 (修改为 6)
nmf = 6              # 费米子单粒子轨道数 N_{mf} = 2q + 1 = 6
nof = 2 * nmf        # 2 个费米子 Flavor (f0 与 f1) -> 12
nmb = nmf - 1        # 玻色子单粒子轨道数 N_{mb} = 2q -> 5
nob = nmb            # 1 个玻色子 Flavor -> 5

# 2. 构造守恒量子数 (电荷 U(1)_e 与角动量 L_z parity)
qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 1) + SQNDiag(GetLz2QNDiag(nmf, 2), nob)
]

# 生成指定扇区的基底构型
cfs = Dict{Int64, SConfs}()
bases = Dict{Int64, SBasis}()
for lz = 0 : 1 
    cfs[lz] = SConfs(nof, nob, nmf, [nmf, lz], qnd)
    bases[lz] = SBasis(cfs[lz])
end 

# 3. 预组装微观观测量与相互作用 
FuzzifiED.ObsNormRadSq = Float64(nmf)
t = 1.5; U = 0.25; g = 1.0

f0 = GetFermionSObs(nmf, 2, 1)
f1 = GetFermionSObs(nmf, 2, 2)
b  = GetBosonSObs(nmb, 1, 1)

n0 = StoreComps(f0' * f0)
n1 = StoreComps(f1' * f1)
nb = StoreComps(b' * b)
nx = StoreComps(f0' * f1 + f1' * f0)
nr = StoreComps(n0 + n1 + nb)

eta = StoreComps(f0' * b)
Deta = let e=eta, R=sqrt(nmf)
    SSphereObs(-e.s2, e.l2m, (l2, m2) -> (-e.s2*(l2+1)/(2R)) * e.get_comp(l2, m2))
end
pair_obs = eta * Deta
tms_hop = SimplifyTerms((GetIntegral(pair_obs)))

tms_int = SimplifyTerms([
    (GetIntegral(nr * nr));
    t * tms_hop; t * tms_hop';
    U * (GetIntegral(nx * Laplacian(nx)));
    g * (GetIntegral(nx * nb))
])

# 定义唯象调控项 
tms_nx = SimplifyTerms(GetIntegral(nx))
tms_n1 = SimplifyTerms(GetIntegral(n1))
tms_nb = SimplifyTerms(GetIntegral(nb))

tms_l2 = GetL2STerms(nmf, 2, nmb, 1) 
l2_mats = Dict{Int64, OpMat{Float64}}()
for lz in 0:1
    l2_mats[lz] = OpMat(SOperator(bases[lz], tms_l2))
end

# ==============================================================================
# 理论靶标 (Super-Ising SCFT Bootstrap)
# ==============================================================================
const TARGET_SIGMA = 0.58444
const TARGET_PSI   = TARGET_SIGMA + 0.5  # SCFT 严格关系 ~ 1.08444

# 4. 针对 Super-Ising 的稳健代价函数
function cost(params)
    q_val, m_val, p_val = params
    # 组装含待定参数的哈密顿量
    tms_hmt = [tms_int; -q_val * tms_nx; -m_val * tms_n1; -p_val * tms_nb]

    try
        # lz = 0 (Bosonic / Scalar sector)
        hmt_0 = SOperator(bases[0], tms_hmt)
        # 为防止小格点下态挤出，提取的态数放宽到 10
        enrg_0, st_0 = GetEigensystem(OpMat(hmt_0), 10)
        l2_0 = [ real(st_0[:, i]' * (l2_mats[0] * st_0[:, i])) for i in eachindex(enrg_0) ]

        # lz = 1 (Fermionic / Spinor sector)
        hmt_1 = SOperator(bases[1], tms_hmt)
        enrg_1, st_1 = GetEigensystem(OpMat(hmt_1), 10)
        l2_1 = [ real(st_1[:, i]' * (l2_mats[1] * st_1[:, i])) for i in eachindex(enrg_1) ]

        # 提取关键态
        idx_l0 = filter(i -> abs(l2_0[i]) < 0.25, 1:length(enrg_0))
        idx_l1 = filter(i -> abs(l2_0[i] - 2.0) < 0.25, 1:length(enrg_0))
        (length(idx_l0) < 2 || isempty(idx_l1)) && return 100.0

        E_0  = enrg_0[idx_l0[1]]  # 真空
        E_σ  = enrg_0[idx_l0[2]]  # σ
        E_∂σ = enrg_0[idx_l1[1]]  # ∂_μ σ

        idx_fhalf = filter(i -> abs(l2_1[i] - 0.75) < 0.25, 1:length(enrg_1))
        isempty(idx_fhalf) && return 100.0
        E_ψ = enrg_1[idx_fhalf[1]] # ψ

        dE_σ = E_σ - E_0
        dE_σ <= 1e-4 && return 100.0

        v_inv = TARGET_SIGMA / dE_σ
        
        # 匹配 SCFT 靶标
        χ2 = ((E_ψ - E_0)*v_inv - TARGET_PSI)^2 + 
             ((E_∂σ - E_0)*v_inv - (TARGET_SIGMA + 1.0))^2
        return χ2
    catch
        return 100.0
    end
end

# 5. 执行参数寻优
# 初始猜测值采用附近相变点参数
init_params = [0.0680, 0.0809, 0.0776] 
println("="^100)
println("正在针对 3D Super-Ising (nmf=6) 执行参数寻优...")
opt_res = optimize(cost, init_params, NelderMead(), Optim.Options(iterations = 100, g_tol = 1e-6, show_trace = false))
best_params = Optim.minimizer(opt_res)
println("优化完成！最佳微观调控参数 [q, m, p] = ", round.(best_params, digits=6))
println("="^100)

# 6. 使用最优参数扩展计算大谱系
q_opt, m_opt, p_opt = best_params
tms_hmt_best = SimplifyTerms([tms_int; -q_opt * tms_nx; -m_opt * tms_n1; -p_opt * tms_nb])

full_result = []
n_states_full = 30  # 每个扇区提取前 30 个低能态

for lz = 0:1
    hmt = SOperator(bases[lz], tms_hmt_best)
    enrg, st = GetEigensystem(OpMat(hmt), n_states_full)
    l2_mat = l2_mats[lz]
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        push!(full_result, [enrg[i], l2_val[i], lz])
    end
end

# 7. 能谱排序与基准点提取
sort!(full_result, by = st -> real(st[1]))
enrg_0 = full_result[1][1] 

enrg_σ = filter(st -> abs(st[2]) < 0.25 && st[3] == 0 && st[1] > enrg_0 + 1e-4, full_result)[1][1]
T_cands = filter(st -> abs(st[2] - 6.0) < 0.35 && st[3] == 0, full_result)
enrg_T = length(T_cands) >= 2 ? T_cands[2][1] : (length(T_cands) == 1 ? T_cands[1][1] : 1.0) 

# ==============================================================================
# Super-Ising (N=1 SCFT) 专属算符识别器
# ==============================================================================
function identify_operator(L2, lz, Delta)
    L_obs = round((sqrt(1 + 4 * max(0, L2)) - 1) / 2, digits=1)

    # 预定义 Super-Ising 初级态
    # 格式: (名称, 理论Δ靶标, 自旋L, Lz parity 扇区)
    primaries = [
        ("I",   0.0,            0.0, 0),
        ("σ",   TARGET_SIGMA,   0.0, 0),
        ("ψ",   TARGET_PSI,     0.5, 1),
        ("ϵ",   1.5844,         0.0, 0),
        ("G_μ", 2.50,           1.5, 1), # 超对称流
        ("T_μν",3.00,           2.0, 0)  # 能动张量
    ]

    best_match = ""
    best_err = 100.0

    for p in primaries
        p_name, p_delta, p_L, p_lz = p
        
        # 1. 检验 Lz Parity 扇区是否匹配
        if lz != p_lz 
            continue 
        end

        # 2. 计算候选所属的后继态阶数 n
        n = round(Delta - p_delta)
        if n < 0 
            continue 
        end

        # 3. 角动量演化验证 (自旋加法)
        if L_obs < abs(p_L - n) - 0.1 || L_obs > (p_L + n) + 0.1
            continue
        end
        if abs((L_obs - p_L) - round(L_obs - p_L)) > 0.1
            continue
        end

        # 4. 残差校验 (由于 nmf=6 格点较小，容差稍微放宽)
        err = abs(Delta - (p_delta + n))
        threshold = 0.40 + 0.10 * n 
        
        # 5. 登记最优匹配
        if err < threshold && err < best_err
            best_err = err
            if n == 0
                best_match = "[$p_name Tower L0] $p_name (Primary, ~$(round(p_delta, digits=2)))"
            else
                best_match = "[$p_name Tower L$(Int(n))] Descendant (spin-$(L_obs), ~$(round(p_delta + n, digits=2)))"
            end
        end
    end

    return best_match == "" ? "Unknown / High Mixing" : best_match
end

# 组装输出矩阵，包含 T 定标和 σ 定标双重参考
spec = [
    begin
        Delta_sigma = TARGET_SIGMA * (st[1] - enrg_0) / (enrg_σ - enrg_0)
        Delta_T = (enrg_T == 1.0) ? 0.0 : 3.00 * (st[1] - enrg_0) / (enrg_T - enrg_0)
        
        # 采用 sigma 定标数据进行物理识别（通常在小格点下更稳定）
        op_name = identify_operator(st[2], st[3], Delta_sigma)
        
        [
            round(Delta_sigma, digits = 4),
            round(Delta_T, digits = 4),
            round(st[1], digits = 6),
            round(st[2], digits = 4),
            st[3],
            op_name
        ]
    end for st in full_result
]

println("="^110)
println(" 3D Super-Ising (N=1 SCFT) 涌现共形算符谱 (nmf=6)")
println("  Δ(σ定标)   Δ(T定标)      能量 E        角动量 L^2     Lz扇区   算符 / 共形塔结构")
println("-"^110)
for i in 1:length(spec)
    row = spec[i]
    if row[1] > 6.0 # 截断无意义的高能尾部
        break
    end
    println(
        rpad(string(row[1]), 12),
        rpad(string(row[2]), 12),
        rpad(string(row[3]), 14),
        rpad(string(row[4]), 14),
        rpad(string(row[5]), 8),
        row[6]
    )
end
println("="^110)
