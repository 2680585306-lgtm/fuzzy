# ==============================================================================
# [个人电脑 PC 专版] O(2) GNY / Super-Ising 稳健寻优与全谱计算 (nmf = 7)
# 策略: 仅使用绝对底层的 3 个态 (ψ, ∂σ, ∂ψ) 进行调优，集成 O(2) Casimir 测量
# ==============================================================================

using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using Optim

# 将底层 BLAS 矩阵运算绑定到 Julia 启动时分配的 CPU 核心数
BLAS.set_num_threads(Threads.nthreads())

FuzzifiED.SilentStd = true
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < eps(Float32)

# ==============================================================================
# 1. 设定系统尺寸与单粒子轨道数
# ==============================================================================
nmf = 7               # 费米子单粒子轨道数 (7个格点)
nof = 2 * nmf         # 14 个费米子轨道
nmb = nmf - 1         # 6 个玻色子轨道
nob = 2 * nmb         # 12 个玻色子总轨道 (O(2) 对应 2 个 Flavor)

println("="^95)
println("启动 O(2) GNY 本地计算任务 (集成 Casimir C2)")
println("当前系统尺寸: nmf = $nmf")
println("费米子轨道数: $nof, 玻色子轨道数: $nob")
println("分配 CPU 线程数: $(Threads.nthreads())")
println("="^95)

# ==============================================================================
# 2. 构造守恒量子数 (U(1)_e, 2*L_z, S_z)
# ==============================================================================
f_sz = zeros(Int64, nof)
b_sz = repeat([1, -1], nmb)
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 2) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    sz_sqnd
]

# 预生成核心扇区 (仅用于优化)
opt_cfs   = Dict{Tuple{Int64, Int64}, SConfs}()
opt_bases = Dict{Tuple{Int64, Int64}, SBasis}()
l2_mats_opt = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()

opt_cfs[(0, 0)]   = SConfs(nof, nob, nmf, [nmf, 0, 0], qnd)
opt_bases[(0, 0)] = SBasis(opt_cfs[(0, 0)])

opt_cfs[(1, 1)]   = SConfs(nof, nob, nmf, [nmf, 1, 1], qnd)
opt_bases[(1, 1)] = SBasis(opt_cfs[(1, 1)])

tms_l2 = GetL2STerms(nmf, 2, nmb, 2)
for k in [(0, 0), (1, 1)]
    l2_mats_opt[k] = OpMat(SOperator(opt_bases[k], tms_l2))
end

# ==============================================================================
# 3. 预组装常数项微观相互作用项
# ==============================================================================
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

# ==============================================================================
# 4. 构造 O(N) 二阶卡西米尔算符 C_2[O(N)] (这里 N = 2 对应 O(2))
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

tms_c2 = on_C2_terms(nmb, 2)

# ==============================================================================
# 5. 稳健的极简代价函数 (加入激进的垃圾回收以防 PC 内存溢出)
# ==============================================================================
function cost(params)
    q_val, m_val, p_val = params
    tms_hmt = [tms_base; -q_val * tms_nx; -m_val * tms_n1; -p_val * tms_nb]

    try
        hmt_00 = SOperator(opt_bases[(0, 0)], tms_hmt)
        enrg_00, st_00 = GetEigensystem(OpMat(hmt_00), 6)
        l2_00 = [ real(st_00[:, i]' * (l2_mats_opt[(0, 0)] * st_00[:, i])) for i in eachindex(enrg_00) ]

        hmt_11 = SOperator(opt_bases[(1, 1)], tms_hmt)
        enrg_11, st_11 = GetEigensystem(OpMat(hmt_11), 6)
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

        # 以 σ 定标 (Δ_σ ≡ 0.66)
        v_inv = 0.66 / dE_σ
        Δ_ψ   = (E_ψ - E_0) * v_inv
        Δ_∂σ  = (E_∂σ - E_0) * v_inv
        Δ_∂ψ  = (E_∂ψ - E_0) * v_inv

        χ2 = (Δ_ψ - 1.07)^2 + (Δ_∂σ - 1.66)^2 + (Δ_∂ψ - 2.07)^2
        
        # 本地防爆内存机制：强制回收大矩阵内存
        GC.gc() 
        return χ2
    catch
        return 100.0
    end
end

# ==============================================================================
# 6. 执行 Nelder-Mead 寻优
# ==============================================================================
init_params = [0.066, 0.079, 0.061]

println("="^88)
println("正在仅利用最底层稳健激发态 (ψ, ∂σ, ∂ψ) 执行高精度调优 (nmf=7)...")
println("【提示】优化极其耗时，将实时打印迭代过程，请耐心等待。")
println("="^88)

opt_res = optimize(
    cost,
    init_params,
    NelderMead(),
    Optim.Options(
        iterations = 80,
        g_tol = 1e-6,
        show_trace = true  # 保持开启，以便监控是否卡死
    )
)

best_params = Optim.minimizer(opt_res)
println("优化完成！参数已拉回物理区域：")
println("最佳参数 [q, m, p] = ", round.(best_params, digits=6))
println("最小代价函数 Cost = ", round(Optim.minimum(opt_res), digits=8))
println("="^88)

# ==============================================================================
# 7. 使用最优参数扩展计算大尺度全谱 (集成 O(2) Casimir 算符)
# ==============================================================================
println("参数优化完毕，开始逐个扇区生成全谱...")
max_sz = 2
cfs_all   = Dict{Tuple{Int64, Int64}, SConfs}()
bases_all = Dict{Tuple{Int64, Int64}, SBasis}()
l2_mats_all = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()
c2_mats_all = Dict{Tuple{Int64, Int64}, OpMat{Float64}}() # 新增 C2 矩阵字典

for lz in 0 : 1 
    for sz in 0 : max_sz
        Sz_val = (mod(lz, 2) == 0) ? 2*sz : 2*sz + 1
        k = (lz, Sz_val)
        cfs_all[k] = SConfs(nof, nob, nmf, [nmf, lz, Sz_val], qnd)
        bases_all[k] = SBasis(cfs_all[k])
        l2_mats_all[k] = OpMat(SOperator(bases_all[k], tms_l2))
        c2_mats_all[k] = OpMat(SOperator(bases_all[k], tms_c2)) # 预组装 C2 矩阵
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

# ★ 将每个扇区求解的态数大幅增加至 120，扩大能谱覆盖面
n_states_full = 120  

for (k, bs) in bases_all
    lz, Sz_val = k
    println(">>> 正在求解扇区 (Lz=$lz, Sz=$Sz_val)...")
    hmt = SOperator(bs, tms_hmt_best)
    enrg, st = GetEigensystem(OpMat(hmt), n_states_full)
    l2_mat = l2_mats_all[k]
    c2_mat = c2_mats_all[k]
    
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]
    c2_val = [ real(st[:, i]' * (c2_mat * st[:, i])) for i in eachindex(enrg) ] # 计算本征态的 C2 值

    for i in eachindex(enrg)
        # 结果推入格式：E, L^2, C2, 2*Lz, Sz
        push!(full_result, [enrg[i], l2_val[i], c2_val[i], lz, Sz_val])
    end
    
    GC.gc() # 扇区计算后及时释放内存
end

sort!(full_result, by = st -> real(st[1]))
enrg_0 = full_result[1][1]  

# 提取关键定标点 (由于插入了 C2，Sz 的索引从 st[4] 变为了 st[5])
enrg_σ = filter(st -> abs(st[2]) < 0.25 && st[5] == 0 && st[1] > enrg_0 + 1e-4, full_result)[1][1]
T_cands = filter(st -> abs(st[2] - 6.0) < 0.35 && st[5] == 0, full_result)
enrg_T = length(T_cands) >= 2 ? T_cands[2][1] : T_cands[1][1]

# ==============================================================================
# 8. 动态算符标签匹配函数
# ==============================================================================
function identify_operator(L2, C2, q, Delta)
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
        return "σ_T (单态配单, Table I: 2.45)"
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
        round(0.66 * (st[1] - enrg_0) / (enrg_σ - enrg_0), digits = 4),
        round(3.00 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 4),
        round(st[1], digits = 6),
        round(st[2], digits = 4),
        round(st[3], digits = 2), # O(2) Casimir 测量值
        st[4],
        st[5],
        identify_operator(st[2], st[3], abs(st[5]), 0.66 * (st[1] - enrg_0) / (enrg_σ - enrg_0))
    ] for st in full_result
]

# ★ 扩大打印数量以覆盖提取出来的全谱
n_print = min(150, length(spec))
println("="^110)
println("纯净无混叠共形算符谱 (已扩大计算与展示范围，共列出前 $n_print 项)：")
println("  Δ(σ定标)   Δ(T定标)      能量 E        角动量 L^2    O(2) C2    2*Lz      Sz      Table I 对应算符态")
println("-"^110)
for i in 1:n_print
    row = spec[i]
    println(
        rpad(string(row[1]), 12),
        rpad(string(row[2]), 12),
        rpad(string(row[3]), 14),
        rpad(string(row[4]), 14),
        rpad(string(row[5]), 11),
        rpad(string(row[6]), 10),
        rpad(string(row[7]), 8),
        row[8]
    )
end
println("="^110)
