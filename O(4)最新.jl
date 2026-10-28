# ==============================================================================
# O(4) GNY 临界点全谱计算 (集成 O(4) 卡西米尔算符测量与精确共形塔识别)
# ==============================================================================

using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using Optim

FuzzifiED.SilentStd = true
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < eps(Float32)

# 1. 设定系统尺寸与单粒子轨道数 (nmf=4)
nmf = 4               
nof = 2 * nmf         
nmb = nmf - 1         
nob = 4 * nmb         

# 2. 构造对角守恒量子数
f_sz = zeros(Int64, nof)
b_sz = repeat([1, -1, 1, -1], nmb) 
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 4) + SQNDiag(GetLz2QNDiag(nmf, 2), nob), 
    sz_sqnd
]

opt_cfs   = Dict{Tuple{Int64, Int64}, SConfs}()
opt_bases = Dict{Tuple{Int64, Int64}, SBasis}()
l2_mats_opt = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()

opt_cfs[(0, 0)]   = SConfs(nof, nob, nmf, [nmf, 0, 0], qnd)
opt_bases[(0, 0)] = SBasis(opt_cfs[(0, 0)])

opt_cfs[(1, 1)]   = SConfs(nof, nob, nmf, [nmf, 1, 1], qnd)
opt_bases[(1, 1)] = SBasis(opt_cfs[(1, 1)])

tms_l2 = GetL2STerms(nmf, 2, nmb, 4) 
for k in [(0, 0), (1, 1)]
    l2_mats_opt[k] = OpMat(SOperator(opt_bases[k], tms_l2))
end

# 3. 预组装微观哈密顿量 
FuzzifiED.ObsNormRadSq = Float64(nmf)
t_hop = 1.5; U_int = 0.25; g_cpl = 1.0

f0 = GetFermionSObs(nmf, 2, 1); f1 = GetFermionSObs(nmf, 2, 2)
b1 = GetBosonSObs(nmb, 4, 1); b2 = GetBosonSObs(nmb, 4, 2)
b3 = GetBosonSObs(nmb, 4, 3); b4 = GetBosonSObs(nmb, 4, 4)

n0 = StoreComps(f0' * f0); n1 = StoreComps(f1' * f1)
nb1 = StoreComps(b1' * b1); nb2 = StoreComps(b2' * b2)
nb3 = StoreComps(b3' * b3); nb4 = StoreComps(b4' * b4)
nb_tot = StoreComps(nb1 + nb2 + nb3 + nb4)
nx = StoreComps(f0' * f1 + f1' * f0)
nr = StoreComps(n0 + n1 + nb_tot)

eta1 = StoreComps(f0' * b1); eta2 = StoreComps(f0' * b2)
eta3 = StoreComps(f0' * b3); eta4 = StoreComps(f0' * b4)
R = sqrt(nmf)
Deta1 = let e=eta1, R=R; SSphereObs(-e.s2, e.l2m, (l2,m2)->(-e.s2*(l2+1)/(2R))*e.get_comp(l2,m2)) end
Deta2 = let e=eta2, R=R; SSphereObs(-e.s2, e.l2m, (l2,m2)->(-e.s2*(l2+1)/(2R))*e.get_comp(l2,m2)) end
Deta3 = let e=eta3, R=R; SSphereObs(-e.s2, e.l2m, (l2,m2)->(-e.s2*(l2+1)/(2R))*e.get_comp(l2,m2)) end
Deta4 = let e=eta4, R=R; SSphereObs(-e.s2, e.l2m, (l2,m2)->(-e.s2*(l2+1)/(2R))*e.get_comp(l2,m2)) end

pair_obs = (eta1 * Deta2 + eta2 * Deta1) + (eta3 * Deta4 + eta4 * Deta3)
tms_hop  = SimplifyTerms(GetIntegral(pair_obs))

tms_base = SimplifyTerms([GetIntegral(nr * nr); t_hop * tms_hop; t_hop * tms_hop'; U_int * GetIntegral(nx * Laplacian(nx)); g_cpl * GetIntegral(nx * nb_tot)])
tms_nx = SimplifyTerms(GetIntegral(nx))
tms_n1 = SimplifyTerms(GetIntegral(n1))
tms_nb = SimplifyTerms(GetIntegral(nb_tot))

# ==============================================================================
# 4. 构造 O(N) 二阶卡西米尔算符 C_2[O(N)] (这里 N = 4 对应 O(4))
# ==============================================================================
function on_C2_terms(nmb::Int, N::Int)
    @assert N >= 2
    U = Matrix{ComplexF64}(I, N, N)
    for k in 1:div(N, 2)
        i = 2k - 1
        j = 2k
        U[i:j, i:j] = ComplexF64[1 -im; 1 im] / sqrt(2)
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

tms_c2 = on_C2_terms(nmb, 4)

# O(4) GNY 理论参考靶标
const TARGET_SIGMA = 0.7578
const TARGET_PSI   = 1.0436

# 5. 稳健极简代价函数
function cost(params)
    q_val, m_val, p_val = params
    tms_hmt = [tms_base; -q_val * tms_nx; -m_val * tms_n1; -p_val * tms_nb]

    try
        hmt_00 = SOperator(opt_bases[(0, 0)], tms_hmt)
        enrg_00, st_00 = GetEigensystem(OpMat(hmt_00), 10)
        l2_00 = [ real(st_00[:, i]' * (l2_mats_opt[(0, 0)] * st_00[:, i])) for i in eachindex(enrg_00) ]

        hmt_11 = SOperator(opt_bases[(1, 1)], tms_hmt)
        enrg_11, st_11 = GetEigensystem(OpMat(hmt_11), 10)
        l2_11 = [ real(st_11[:, i]' * (l2_mats_opt[(1, 1)] * st_11[:, i])) for i in eachindex(enrg_11) ]

        idx_l0 = filter(i -> abs(l2_00[i]) < 0.25, 1:length(enrg_00))
        idx_l1 = filter(i -> abs(l2_00[i] - 2.0) < 0.25, 1:length(enrg_00))
        (length(idx_l0) < 2 || isempty(idx_l1)) && return 100.0

        E_0  = enrg_00[idx_l0[1]]
        E_σ  = enrg_00[idx_l0[2]]
        E_∂σ = enrg_00[idx_l1[1]]

        idx_fhalf  = filter(i -> abs(l2_11[i] - 0.75) < 0.25, 1:length(enrg_11))
        idx_f3half = filter(i -> abs(l2_11[i] - 3.75) < 0.25, 1:length(enrg_11))
        (isempty(idx_fhalf) || isempty(idx_f3half)) && return 100.0

        E_ψ  = enrg_11[idx_fhalf[1]]
        E_∂ψ = enrg_11[idx_f3half[1]]

        dE_σ = E_σ - E_0
        dE_σ <= 1e-4 && return 100.0

        v_inv = TARGET_SIGMA / dE_σ
        χ2 = ((E_ψ - E_0)*v_inv - TARGET_PSI)^2 + 
             ((E_∂σ - E_0)*v_inv - (TARGET_SIGMA + 1.0))^2 + 
             ((E_∂ψ - E_0)*v_inv - (TARGET_PSI + 1.0))^2
        return χ2
    catch
        return 100.0
    end
end

# 6. 执行 Nelder-Mead 寻优
init_params = [0.066, 0.079, 0.061]
println("="^95)
println("正在针对 O(4) GNY (nmf=4) 执行参数寻优...")
opt_res = optimize(cost, init_params, NelderMead(), Optim.Options(iterations = 80, g_tol = 1e-6, show_trace = false))
best_params = Optim.minimizer(opt_res)
println("优化完成！最佳参数 [q, m, p] = ", round.(best_params, digits=6))
println("="^95)

# 7. 使用最优参数扩展计算大尺度全谱 (同时测量 O(4) Casimir 算符)
max_sz = 2
cfs_all   = Dict{Tuple{Int64, Int64}, SConfs}()
bases_all = Dict{Tuple{Int64, Int64}, SBasis}()
l2_mats_all = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()
c2_mats_all = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()

for lz in 0 : 1 
    for sz in 0 : max_sz
        Sz_val = (mod(lz, 2) == 0) ? 2*sz : 2*sz + 1
        k = (lz, Sz_val)
        cfs_all[k] = SConfs(nof, nob, nmf, [nmf, lz, Sz_val], qnd)
        bases_all[k] = SBasis(cfs_all[k])
        l2_mats_all[k] = OpMat(SOperator(bases_all[k], tms_l2))
        c2_mats_all[k] = OpMat(SOperator(bases_all[k], tms_c2))
    end
end

q_opt, m_opt, p_opt = best_params
tms_hmt_best = SimplifyTerms([tms_base; -q_opt * tms_nx; -m_opt * tms_n1; -p_opt * tms_nb])

full_result = []
n_states_full = 150 

for (k, bs) in bases_all
    lz, Sz_val = k
    hmt = SOperator(bs, tms_hmt_best)
    enrg, st = GetEigensystem(OpMat(hmt), n_states_full)
    l2_mat = l2_mats_all[k]
    c2_mat = c2_mats_all[k]
    
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]
    c2_val = [ real(st[:, i]' * (c2_mat * st[:, i])) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        push!(full_result, [enrg[i], l2_val[i], c2_val[i], lz, Sz_val])
    end
end

sort!(full_result, by = st -> real(st[1]))
enrg_0 = full_result[1][1] 

enrg_σ = filter(st -> abs(st[2]) < 0.25 && st[5] == 0 && st[1] > enrg_0 + 1e-4, full_result)[1][1]
T_cands = filter(st -> abs(st[2] - 6.0) < 0.35 && st[5] == 0, full_result)
enrg_T = length(T_cands) >= 2 ? T_cands[2][1] : (length(T_cands) == 1 ? T_cands[1][1] : 1.0) 

# ==============================================================================
# 全自动通用化共形塔 (Conformal Tower) 算法识别器
# ==============================================================================
function identify_operator(L2, C2, q, Delta)
    L_obs = round((sqrt(1 + 4 * max(0, L2)) - 1) / 2, digits=1)

    primaries = [
        ("I",   0.0,          0.0, 0, "+"),
        ("σ",   TARGET_SIGMA, 0.0, 0, "-"),
        ("ψ",   TARGET_PSI,   0.5, 1, "+"),
        ("ϵ",   1.899,        0.0, 0, "+"),
        ("J_μ", 2.00,         1.0, 0, "+"),
        ("σ_T", 2.293,        0.0, 2, "-"),
        ("T_μν",3.00,         2.0, 0, "+")
    ]

    best_match = ""
    best_err = 100.0
    best_P = ""

    for p in primaries
        p_name, p_delta, p_L, p_q, p_parity = p
        
        if q != p_q 
            continue 
        end

        n = round(Delta - p_delta)
        if n < 0 
            continue 
        end

        if L_obs < abs(p_L - n) - 0.1 || L_obs > (p_L + n) + 0.1
            continue
        end
        if abs((L_obs - p_L) - round(L_obs - p_L)) > 0.1
            continue
        end

        err = abs(Delta - (p_delta + n))
        threshold = 0.35 + 0.08 * n 
        
        if err < threshold && err < best_err
            best_err = err
            best_P = p_parity 
            
            if n == 0
                best_match = "[$p_name Tower L0] $p_name (Primary, ~$(round(p_delta, digits=2)))"
            else
                best_match = "[$p_name Tower L$(Int(n))] Descendant (spin-$(L_obs), ~$(round(p_delta + n, digits=2)))"
            end
        end
    end

    if best_match == ""
        return ("", "")
    else
        return (best_P, best_match)
    end
end

# 组装输出矩阵 (加入 O(4) 卡西米尔 C2 列)
spec = [
    begin
        Delta = TARGET_SIGMA * (st[1] - enrg_0) / (enrg_σ - enrg_0)
        P_str, op_name = identify_operator(st[2], st[3], abs(st[5]), Delta)
        Delta_T = (enrg_T == 1.0) ? 0.0 : 3.00 * (st[1] - enrg_0) / (enrg_T - enrg_0)
        [
            round(Delta, digits = 4),
            round(Delta_T, digits = 4),
            round(st[1], digits = 6),
            round(st[2], digits = 4),
            round(st[3], digits = 2), # O(4) Casimir 数值
            st[4],
            st[5],
            P_str,
            op_name
        ]
    end for st in full_result
]

n_print = min(250, length(spec))
println("="^120)
println(" 深度探索 O(4) GNY 大尺度共形算符谱 (已集成卡西米尔算符 C2 输出)")
println("  Δ(σ定标)   Δ(T定标)      能量 E        角动量 L^2    O(4) C2    2*Lz      Sz     P(理论)    算符 / 共形塔级次")
println("-"^120)
for i in 1:n_print
    row = spec[i]
    if row[1] > 10.5
        break
    end
    
    println(
        rpad(string(row[1]), 12),
        rpad(string(row[2]), 12),
        rpad(string(row[3]), 14),
        rpad(string(row[4]), 14),
        rpad(string(row[5]), 11),
        rpad(string(row[6]), 10),
        rpad(string(row[7]), 8),
        rpad(string(row[8]), 10),
        row[9]
    )
end
println("="^120)
