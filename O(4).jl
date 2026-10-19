# 3D GNY-CFT (N_boson = 4, O(4) 内部对称性) 
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using Optim

FuzzifiED.SilentStd = true
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统几何与单粒子轨道数 (N_boson = 4)
nmf = 6
FuzzifiED.ObsNormRadSq = Float64(nmf)
nof = 2 * nmf             # 费米子总轨道数: 12
nmb = nmf - 1             # 玻色子单粒子轨道数: 5
N_boson = 4               # 4 个玻色子 Flavor (构成 O(4) 向量)
nob = N_boson * nmb       # 玻色子总轨道数: 20

# 2. 构造守恒量子数 (电荷 U(1)_e, 2*L_z, 以及 O(4) Cartan 生成元 S_z)
f_sz = zeros(Int64, nof)
b_sz = repeat([1, -1, 1, -1], nmb)
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, N_boson) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    sz_sqnd
]

# 3. 构造微观两体与单体微观算符 (仅执行一次)
den_e = StoreComps(GetFerDensitySObs(nmf, 2) + GetBosDensitySObs(nmb, N_boson))
tms_int_e = SimplifyTerms(GetIntegral(den_e * den_e))

obs_b1p = StoreComps(GetFermionSObs(nmf, 2, 1)' * GetBosonSObs(nmb, N_boson, 1))
obs_b1m = StoreComps(GetFermionSObs(nmf, 2, 1)' * GetBosonSObs(nmb, N_boson, 2))
obs_b2p = StoreComps(GetFermionSObs(nmf, 2, 1)' * GetBosonSObs(nmb, N_boson, 3))
obs_b2m = StoreComps(GetFermionSObs(nmf, 2, 1)' * GetBosonSObs(nmb, N_boson, 4))

tms_hop = SimplifyTerms(GetIntegral(
    obs_b1p * DPlus(obs_b1m) + obs_b1m * DPlus(obs_b1p) +
    obs_b2p * DPlus(obs_b2m) + obs_b2m * DPlus(obs_b2p)
))

den_σ = StoreComps(GetFerDensitySObs(nmf, 2, [0 1 ; 1 0]))
tms_int_σ = SimplifyTerms(GetIntegral(den_σ * Laplacian(den_σ)))

den_χχ = StoreComps(GetBosDensitySObs(nmb, N_boson))
tms_int_yuk = SimplifyTerms(GetIntegral(den_σ * den_χχ + den_χχ * den_σ) / 2)

tms_pol_σ  = SimplifyTerms(GetIntegral(den_σ))
tms_pol_χχ = SimplifyTerms(GetIntegral(GetBosDensitySObs(nmb, N_boson)))
tms_pol_ϵ  = SimplifyTerms(GetIntegral(GetFerDensitySObs(nmf, 2, [0 0 ; 0 1])))

# 施加正规序过滤
filter!(tm -> length(tm.cstr) > 4, tms_int_e)
filter!(tm -> length(tm.cstr) > 4, tms_hop)
filter!(tm -> length(tm.cstr) > 4, tms_int_σ)
filter!(tm -> length(tm.cstr) > 4, tms_int_yuk)

t_hop = 1.5; U_int = 0.25; g_cpl = 1.0
tms_base = SimplifyTerms([
    tms_int_e;
    - t_hop * (tms_hop + tms_hop');
    U_int * tms_int_σ;
    g_cpl * tms_int_yuk
])

tms_l2 = GetL2STerms(nmf, 2, nmb, N_boson)


opt_sectors = [(0, 0), (1, 1)]
opt_bases = Dict{Tuple{Int64, Int64}, SBasis}()
mats_base = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()
mats_σ    = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()
mats_ϵ    = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()
mats_χ    = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()
mats_l2   = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()

println("正在预先组装稀疏矩阵...")
for k in opt_sectors
    lz2, sz = k
    conf = SConfs(nof, nob, nmf, [nmf, lz2, sz], qnd)
    bs = SBasis(conf)
    opt_bases[k] = bs
    mats_base[k] = OpMat(SOperator(bs, tms_base))
    mats_σ[k]    = OpMat(SOperator(bs, tms_pol_σ))
    mats_ϵ[k]    = OpMat(SOperator(bs, tms_pol_ϵ))
    mats_χ[k]    = OpMat(SOperator(bs, tms_pol_χχ))
    mats_l2[k]   = OpMat(SOperator(bs, tms_l2))
end


Δ_σ_target = 0.768
Δ_ψ_target = 1.095

function fast_cost(params)
    q_val, m_val, p_val = params
    try
        # (0, 0) 扇区: 仅矩阵线性组合
        H_00 = mats_base[(0, 0)] - q_val * mats_σ[(0, 0)] - m_val * mats_ϵ[(0, 0)] - p_val * mats_χ[(0, 0)]
        enrg_00, st_00 = GetEigensystem(H_00, 4)
        l2_00 = [ real(st_00[:, i]' * (mats_l2[(0, 0)] * st_00[:, i])) for i in 1:4 ]

        # (1, 1) 扇区
        H_11 = mats_base[(1, 1)] - q_val * mats_σ[(1, 1)] - m_val * mats_ϵ[(1, 1)] - p_val * mats_χ[(1, 1)]
        enrg_11, st_11 = GetEigensystem(H_11, 4)
        l2_11 = [ real(st_11[:, i]' * (mats_l2[(1, 1)] * st_11[:, i])) for i in 1:4 ]

        # 筛选态
        idx_l0 = filter(i -> abs(l2_00[i]) < 0.25, 1:4)
        idx_l1 = filter(i -> abs(l2_00[i] - 2.0) < 0.25, 1:4)
        (length(idx_l0) < 2 || isempty(idx_l1)) && return 100.0

        E_0  = enrg_00[idx_l0[1]]
        E_σ  = enrg_00[idx_l0[2]]
        E_∂σ = enrg_00[idx_l1[1]]

        idx_fhalf  = filter(i -> abs(l2_11[i] - 0.75) < 0.25, 1:4)
        idx_f3half = filter(i -> abs(l2_11[i] - 3.75) < 0.25, 1:4)
        (isempty(idx_fhalf) || isempty(idx_f3half)) && return 100.0

        E_ψ  = enrg_11[idx_fhalf[1]]
        E_∂ψ = enrg_11[idx_f3half[1]]

        dE_σ = E_σ - E_0
        dE_σ <= 1e-4 && return 100.0

        # 定标并计算残差
        v_inv = Δ_σ_target / dE_σ
        Δ_ψ   = (E_ψ - E_0) * v_inv
        Δ_∂σ  = (E_∂σ - E_0) * v_inv
        Δ_∂ψ  = (E_∂ψ - E_0) * v_inv

        return (Δ_ψ - Δ_ψ_target)^2 + (Δ_∂σ - (Δ_σ_target + 1.0))^2 + (Δ_∂ψ - (Δ_ψ_target + 1.0))^2
    catch
        return 100.0
    end
end


init_params = [0.0698, 0.0774, 0.0722]

println("="^80)
println("执行快速 Nelder-Mead 寻优...")
println("="^80)

opt_res = optimize(
    fast_cost,
    init_params,
    NelderMead(),
    Optim.Options(iterations = 60, show_trace = true, show_every = 10)
)

best_params = Optim.minimizer(opt_res)
println("="^80)
println("寻优极速完成！最优临界参数:")
println("  q (hx) = ", round(best_params[1], digits=6))
println("  m (hz) = ", round(best_params[2], digits=6))
println("  p (μ)  = ", round(best_params[3], digits=6))
println("  Cost   = ", round(Optim.minimum(opt_res), digits=8))
println("="^80)

#  构造 O(4) Casimir 算符
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
        M[a, b] = 1; M[b, a] = -1
        Mc = U * M * U'
        Aab = GetBosonPolSTerms(nmb, N, Mc)
        append!(terms, -(Aab * Aab))
    end
    return SimplifyTerms(terms)
end
tms_c2 = on_C2_terms(nmb, N_boson)


q_opt, m_opt, p_opt = best_params
tms_hmt_best = SimplifyTerms([
    tms_base;
    - q_opt * tms_pol_σ;
    - m_opt * tms_pol_ϵ;
    - p_opt * tms_pol_χχ
])

target_sectors = [
    (0, 0),  # 真空 I, σ, ∂σ, ϵ, ∂²σ, T
    (1, 1),  # ψ, ∂ψ
    (4, 0),  # 能动张量最高权态 T (Lz = 2, Sz = 0)
    (0, 2),  # 配对张量态 σ_T (Sz = 2)
]

cfs = Dict{Tuple{Int64, Int64}, SConfs}()
for (lz2, sz) in target_sectors
    cfs[(lz2, sz)] = SConfs(nof, nob, nmf, [nmf, lz2, sz], qnd)
end

result = []
n_states = 18

for (k, bs_conf) in cfs
    lz2, Sz_val = k
    bs = SBasis(bs_conf)
    hmt = SOperator(bs, tms_hmt_best)
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, min(n_states, bs.dim))

    l2 = SOperator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ real(st[:, i]' * l2_mat * st[:, i]) for i in eachindex(enrg) ]

    c2_op = SOperator(bs, tms_c2)
    c2_mat = OpMat(c2_op)
    c2_val = [ real(st[:, i]' * c2_mat * st[:, i]) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        push!(result, [real(enrg[i]), l2_val[i], c2_val[i], lz2, Sz_val])
    end
end

sort!(result, by = st -> st[1])
enrg_0 = result[1][1]

# 提取标量原初态 σ
sigma_cands = filter(st -> abs(st[2]) < 0.25 && st[4] == 0 && st[5] == 0 && (st[1] - enrg_0) > 1e-4, result)
enrg_σ = sigma_cands[1][1]

# 定位能动张量 T
T_in_lz2 = filter(st -> st[4] == 4 && st[5] == 0 && abs(st[2] - 6.0) < 0.35 && abs(st[3]) < 0.25, result)
if !isempty(T_in_lz2)
    enrg_T = T_in_lz2[1][1]
    println("在 (2Lz=4, Sz=0) 扇区定位到能动张量最高权态 T：E_T = $(round(enrg_T, digits=6))")
else
    T_in_lz0 = filter(st -> st[4] == 0 && st[5] == 0 && abs(st[2] - 6.0) < 0.45 && abs(st[3]) < 0.25 && (st[1] - enrg_0) > (enrg_σ - enrg_0), result)
    enrg_T = !isempty(T_in_lz0) ? T_in_lz0[1][1] : (enrg_0 + (3.0 / Δ_σ_target) * (enrg_σ - enrg_0))
end


println("\n" * "="^95)
println("(2Lz=0, Sz=2) 扇区前 8 个激发态 (寻找 O(4) 配对张量 σ_T):")
println("  Δ(T定标)   Δ(σ定标)       原始能量 E       角动量 L^2      Casimir C_2   可能对应的算符")
println("-"^95)

sector_02_states = filter(st -> st[4] == 0 && st[5] == 2, result)
for st in sector_02_states[1:min(8, length(sector_02_states))]
    delta_T = round(3.0 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 4)
    delta_σ = round(Δ_σ_target * (st[1] - enrg_0) / (enrg_σ - enrg_0), digits = 4)
    l2_round = round(st[2], digits = 3)
    c2_round = round(st[3], digits = 2)
    e_round  = round(st[1], digits = 5)

    label = ""
    if abs(st[2]) < 0.35
        label = "★ σ_T 候选 (O(4) 无迹对称配对, Bootstrap 理论 Δ ≈ 2.35)"
    elseif abs(st[2] - 2.0) < 0.4
        label = "∂_μ σ_T 候选 (矢量配对, 理论 Δ ≈ 3.35)"
    elseif abs(st[2] - 6.0) < 0.6
        label = "D_μν 候选 (张量配对流)"
    end

    println(rpad(string(delta_T), 12), rpad(string(delta_σ), 14), rpad(string(e_round), 16), rpad(string(l2_round), 16), rpad(string(c2_round), 14), label)
end
println("="^95)

# 全局能谱输出
spec = [
    [
        round(3.0 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 4),
        round(Δ_σ_target * (st[1] - enrg_0) / (enrg_σ - enrg_0), digits = 4),
        round(st[1], digits = 6),
        round(st[2], digits = 4),
        round(Int, round(st[3])),
        st[4],
        st[5]
    ]
    for st in result
]

println("\n============= 3D GNY-CFT (N_boson = 4, O(4) 对称性) 全局标度能谱 =============")
println("  Δ(T定标)   Δ(σ定标)      能量 E        角动量 L^2    O(4) C2    2*Lz    Sz")
println("-"^80)
for i in 1:min(35, length(spec))
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
println("="^80)
