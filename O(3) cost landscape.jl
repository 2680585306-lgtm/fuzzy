# ==============================================================================
# O(3) GNY 临界点参数优化、全谱计算与二维损失景观对数可视化
# ==============================================================================

using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
using Optim
using Plots  # 用于绘制参数空间的二维热力图

FuzzifiED.SilentStd = true
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < eps(Float32)

# ==============================================================================
# 1. 设定系统尺寸与单粒子轨道数 (O(3) 通常取 nmf = 5)
# ==============================================================================
nmf = 5               # 费米子单粒子轨道数 N_{mf} = 2q + 1
nof = 2 * nmf         # 2 个费米子 Flavor (f0, f1)，总费米子轨道数为 2 * N_{mf}
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q
nob = 3 * nmb         # 3 个玻色子 Flavor (b1, b2, b3)，对应 O(3) 矢量分量

# ==============================================================================
# 2. 构造守恒量子数 (U(1)_e, 2*L_z, S_z)
# ==============================================================================
f_sz = zeros(Int64, nof)
# O(3) 3个玻色子分量的 Sz 电荷映射分别为 +1, -1, 0
b_sz = repeat([1, -1, 0], nmb)
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 3) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    sz_sqnd
]

# 预生成核心扇区 (仅用于优化调参，大幅节省时间)
opt_cfs   = Dict{Tuple{Int64, Int64}, SConfs}()
opt_bases = Dict{Tuple{Int64, Int64}, SBasis}()
l2_mats_opt = Dict{Tuple{Int64, Int64}, OpMat{Float64}}()

# (0, 0) 扇区: 包含 真空, σ, ∂σ
opt_cfs[(0, 0)]   = SConfs(nof, nob, nmf, [nmf, 0, 0], qnd)
opt_bases[(0, 0)] = SBasis(opt_cfs[(0, 0)])

# (1, 1) 扇区: 包含 ψ, ∂ψ
opt_cfs[(1, 1)]   = SConfs(nof, nob, nmf, [nmf, 1, 1], qnd)
opt_bases[(1, 1)] = SBasis(opt_cfs[(1, 1)])

tms_l2 = GetL2STerms(nmf, 2, nmb, 3)
for k in [(0, 0), (1, 1)]
    l2_mats_opt[k] = OpMat(SOperator(opt_bases[k], tms_l2))
end

# ==============================================================================
# 3. 预组装常数项微观相互作用项 (O(3) 3分量耦合)
# ==============================================================================
FuzzifiED.ObsNormRadSq = Float64(nmf)
t_hop = 1.5; U_int = 0.25; g_cpl = 1.0

f0 = GetFermionSObs(nmf, 2, 1)
f1 = GetFermionSObs(nmf, 2, 2)
b1 = GetBosonSObs(nmb, 3, 1)
b2 = GetBosonSObs(nmb, 3, 2)
b3 = GetBosonSObs(nmb, 3, 3)

n0     = StoreComps(f0' * f0)
n1     = StoreComps(f1' * f1)
nb1    = StoreComps(b1' * b1)
nb2    = StoreComps(b2' * b2)
nb3    = StoreComps(b3' * b3)
nb_tot = StoreComps(nb1 + nb2 + nb3)
nx     = StoreComps(f0' * f1 + f1' * f0)
nr     = StoreComps(n0 + n1 + nb_tot)

eta1 = StoreComps(f0' * b1)
eta2 = StoreComps(f0' * b2)
eta3 = StoreComps(f0' * b3)
R = sqrt(nmf)

Deta1 = let e = eta1, R = R
    SSphereObs(-e.s2, e.l2m, (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end
Deta2 = let e = eta2, R = R
    SSphereObs(-e.s2, e.l2m, (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end
Deta3 = let e = eta3, R = R
    SSphereObs(-e.s2, e.l2m, (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

# O(3) 跃迁项：复分量对 (1, 2) 加实分量 (3)
pair_obs = (eta1 * Deta2 + eta2 * Deta1) + (eta3 * Deta3)
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
# 4. 构造 O(N) 二阶卡西米尔算符 C_2[O(N)] (N = 3)
# ==============================================================================
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

tms_c2 = on_C2_terms(nmf-1, 3)

# O(3) GNY 理论参考靶标
const TARGET_SIGMA = 0.724
const TARGET_PSI   = 1.054

# ==============================================================================
# 5. 稳健的极简代价函数
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

        v_inv = TARGET_SIGMA / dE_σ
        Δ_ψ   = (E_ψ - E_0) * v_inv
        Δ_∂σ  = (E_∂σ - E_0) * v_inv
        Δ_∂ψ  = (E_∂ψ - E_0) * v_inv

        χ2 = (Δ_ψ - TARGET_PSI)^2 + (Δ_∂σ - (TARGET_SIGMA + 1.0))^2 + (Δ_∂ψ - (TARGET_PSI + 1.0))^2
        return χ2
    catch
        return 100.0
    end
end

# ==============================================================================
# 6. 执行 Nelder-Mead 寻优
# ==============================================================================
init_params = [0.1, 0.18, 0.18]

println("="^88)
println("正在针对 O(3) GNY 模型执行高精度参数调优...")
println("="^88)

opt_res = optimize(
    cost,
    init_params,
    NelderMead(),
    Optim.Options(iterations = 80, g_tol = 1e-6, show_trace = false)
)

best_params = Optim.minimizer(opt_res)
println("优化完成！参数已拉回物理区域：")
println("最佳参数 [q, m, p] = ", round.(best_params, digits=6))
println("最小代价函数 Cost = ", round(Optim.minimum(opt_res), digits=8))
println("="^88)

# ==============================================================================
# 7. 使用最优参数扩展计算全谱
# ==============================================================================
max_sz = 2
cfs_all     = Dict{Tuple{Int64, Int64}, SConfs}()
bases_all   = Dict{Tuple{Int64, Int64}, SBasis}()
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
tms_hmt_best = SimplifyTerms([
    tms_base;
    -q_opt * tms_nx;
    -m_opt * tms_n1;
    -p_opt * tms_nb
])

full_result = []
n_states_full = 30  

for (k, bs) in bases_all
    lz, Sz_val = k
    hmt = SOperator(bs, tms_hmt_best)
    enrg, st = GetEigensystem(OpMat(hmt), n_states_full)
    
    l2_mat = l2_mats_all[k]
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]
    
    c2_mat = c2_mats_all[k]
    c2_val = [ real(st[:, i]' * (c2_mat * st[:, i])) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        push!(full_result, [enrg[i], l2_val[i], c2_val[i], lz, Sz_val])
    end
end

sort!(full_result, by = st -> real(st[1]))
enrg_0 = full_result[1][1]  

enrg_σ = filter(st -> abs(st[2]) < 0.25 && st[5] == 0 && st[1] > enrg_0 + 1e-4, full_result)[1][1]
T_cands = filter(st -> abs(st[2] - 6.0) < 0.35 && st[5] == 0, full_result)
enrg_T = length(T_cands) >= 2 ? T_cands[2][1] : T_cands[1][1]

# ==============================================================================
# 8. O(3) 动态算符标签匹配函数
# ==============================================================================
function identify_operator(L2, C2, q, Delta)
    c2_match(expected) = abs(C2 - expected) < 0.6

    if abs(L2) < 0.2 && q == 0 && Delta < 0.1 && c2_match(0.0)
        return "真空基态 I (Δ=0)"
    elseif abs(L2) < 0.2 && q == 0 && abs(Delta - TARGET_SIGMA) < 0.25 && c2_match(0.0)
        return "σ ~ φ"
    elseif abs(L2 - 0.75) < 0.2 && q == 1 && abs(Delta - TARGET_PSI) < 0.25 && c2_match(2.0)
        return "ψ_i"
    elseif abs(L2 - 2.0) < 0.25 && q == 0 && abs(Delta - (TARGET_SIGMA+1.0)) < 0.25 && c2_match(0.0)
        return "∂_μ σ"
    elseif abs(L2 - 2.0) < 0.25 && q == 0 && abs(Delta - 2.00) < 0.25 && c2_match(2.0)
        return "J_μ^ij (O(3) 流)"
    elseif abs(L2 - 3.75) < 0.25 && q == 1 && abs(Delta - (TARGET_PSI+1.0)) < 0.25 && c2_match(2.0)
        return "∂_μ ψ_i"
    elseif abs(L2) < 0.2 && q == 0 && abs(Delta - 1.8074) < 0.35 && c2_match(0.0)
        return "ϵ ~ φ^2"
    elseif abs(L2) < 0.2 && q == 0 && abs(Delta - 2.358) < 0.40 && c2_match(6.0)
        return "σ_T (对称无迹张量)"
    elseif abs(L2 - 6.0) < 0.35 && q == 0 && Delta >= 2.85 && c2_match(0.0)
        return "T_μν (能动张量)"
    else
        return ""
    end
end

spec = [
    [
        round(TARGET_SIGMA * (st[1] - enrg_0) / (enrg_σ - enrg_0), digits = 4),
        round(3.00 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 4),
        round(st[1], digits = 6),
        round(st[2], digits = 4),
        round(st[3], digits = 4), 
        st[4],
        st[5],
        identify_operator(st[2], st[3], abs(st[5]), TARGET_SIGMA * (st[1] - enrg_0) / (enrg_σ - enrg_0))
    ] for st in full_result
]

n_print = min(35, length(spec))
println("="^118)
println("O(3) GNY 纯净共形算符谱 (已集成 O(3) Casimir 校验，共列出前 $n_print 项)：")
println("  Δ(σ定标)   Δ(T定标)      能量 E        角动量 L^2    O(3) C2    2*Lz      Sz     对应算符态")
println("-"^118)
for i in 1:n_print
    row = spec[i]
    println(
        rpad(string(row[1]), 12),
        rpad(string(row[2]), 12),
        rpad(string(row[3]), 14),
        rpad(string(row[4]), 14),
        rpad(string(row[5]), 11),
        rpad(string(row[6]), 10),
        rpad(string(row[7]), 7),
        rpad(string(row[8]), 25)
    )
end
println("="^118)

# ==============================================================================
# 9. 参数空间二维损失景观 (Cost Landscape) 扫描与对数尺度可视化
# ==============================================================================
println("正在准备绘制 O(3) 二维参数景观图 (Fixed p = $(round(p_opt, digits=5)))...")

delta = 0.01
n_grid = 35 
q_range = range(q_opt - delta, q_opt + delta, length=n_grid)
m_range = range(m_opt - delta, m_opt + delta, length=n_grid)

Z = zeros(length(m_range), length(q_range))

println("开始扫描参数网格 ($(n_grid) x $(n_grid) = $(n_grid^2) 个计算点)...")
for (i, m_val) in enumerate(m_range)
    for (j, q_val) in enumerate(q_range)
        Z[i, j] = cost([q_val, m_val, p_opt])
    end
    print("\r网格扫描进度: $(round(i / n_grid * 100, digits=1))%  ")
end
println("\n扫描完成，正在渲染图像...")

Z_log = log10.(Z .+ 1e-10)

# 绘制高精度彩色等高线图
p_plot = contourf(
    q_range, m_range, Z_log,
    levels = 100,             
    linewidth = 0,              
    color = :turbo,             
    xlabel = "q",               
    ylabel = "m",               
    colorbar_title = "Log10(Cost χ²)",
    framestyle = :box,
    size = (750, 600)           
)

# 标出极值点
legend_info = " Fixed p = $(round(p_opt, digits=4))\n Min: q=$(round(q_opt, digits=4)), m=$(round(m_opt, digits=4))"
scatter!(
    p_plot,
    [q_opt], [m_opt],
    color = :white,             
    markerstrokecolor = :black,
    marker = :star5,
    markersize = 12,
    label = legend_info,        
    legend = :topright,
    legendfontsize = 9
)

display(p_plot)

save_path = joinpath(pwd(), "GNY_O3_Cost_Landscape_Log.png")
savefig(p_plot, save_path)

println("O(3) 图像已成功保存在您的电脑上，路径为：")
println(save_path)
println("="^118)

println("【提示】图像窗口已弹出。请按回车键 (Enter) 退出程序...")
readline()
