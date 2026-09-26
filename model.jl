# 这个脚本计算 3D Super-Ising 超对称共形场论（SCFT）的能谱。
# 对应论文 arXiv:2509.08038 中的 Section 8, Figure 12 与 Figure 13。
# 包含 2 个费米子 Flavor (f0, f1) 与 1 个玻色子 Flavor (b)。

using FuzzifiED
using FuzzifiED.Fuzzifino
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸与单粒子轨道数
nmf = 9               # 费米子单粒子轨道数 N_{mf} = 2q + 1 (此处取 N_{mf} = 9)
nof = 2 * nmf         # 2 个费米子 Flavor (f0 与 f1)，总费米子轨道数为 2 * N_{mf}
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q (角动量与费米子相差 1/2)
nob = nmb             # 1 个玻色子 Flavor

# 2. 构造守恒量子数 (电荷 U(1)_e 与角动量 L_z)
qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 1) + SQNDiag(GetLz2QNDiag(nmf, 2), nob)
]

# 生成指定扇区的基底构型：总电荷数 N_e = N_{mf}，分别求解 L_z = 0 和 L_z = 1/2
cfs = Dict{Int64, SConfs}()
for lz = 0 : 1 
    cfs[lz] = SConfs(nof, nob, nmf, [nof, lz], qnd)
end 

# 3. 构造微观模算符 (SMod)
# 对应三种 Flavour 间的创生/湮灭算符组合
amd_f0f0 = GetFermionSMod(nmf, 2, 1) * GetFermionSMod(nmf, 2, 1)  # f0-f0 作用
amd_f1f0 = GetFermionSMod(nmf, 2, 2) * GetFermionSMod(nmf, 2, 1)  # f1-f0 作用 (对应 σ 算符)
amd_f0b  = GetFermionSMod(nmf, 2, 1) * GetBosonSMod(nmb, 1, 1)   # f0-b 作用  (对应 χ 算符)
amd_f1f1 = GetFermionSMod(nmf, 2, 2) * GetFermionSMod(nmf, 2, 2)  # f1-f1 作用
amd_bb   = GetBosonSMod(nmb, 1, 1)   * GetBosonSMod(nmb, 1, 1)   # b-b 作用

# 4. 缩合构造相互作用项 (ContractMod)
tms_hop = ContractMod(amd_f0f0', amd_bb, nmf - 2)      # 动能对转换项 \eta D_+ \eta (t 项)
tms_u   = ContractMod(amd_f1f0', amd_f1f0, nmf - 2)     # 标量梯度导数相互作用 n_x \nabla^2 n_x (U 项)
tms_g   = ContractMod(amd_f1f0', amd_f0b, nmf - 3/2)    # Yukawa 耦合项 n_x n_b (g 项)

tms_h   = STerms(GetF1F0STerms(nmf, 2, 1, 2))            # 极化项 n_x (h 项)
tms_m1  = STerms(GetFermionNTerms(nmf, 2, 2, nob))       # f1 密度项 n_1 (\mu_1 项)
tms_mb  = STerms(GetBosonNTerms(nof, nmb, 1, 1))         # b 密度项 n_b (\mu_b 项)
tms_e2  = GetE2STerms(nmf, 2, nmb, 1)                    # 电荷密度平方项 n_e^2

# 5. 组装 Super-Ising 哈密顿量 (设定最佳临界点参数，参照论文 Table 2 N_{mf}=9)
t = 1.5; U = 0.25; g = 1.0
h = 0.0680; mu1 = 0.0809; mub = 0.0776

tms_hmt = SimplifyTerms(
    1.0 * tms_e2
    + t * (tms_hop + tms_hop')
    + U * tms_u
    + g * (tms_g + tms_g')
    - h * tms_h
    - mu1 * tms_m1
    - mub * tms_mb
)

# 构造总角动量平方算符 L^2 (用于确定本征态的自旋 l)
tms_l2 = GetL2STerms(nmf, 2, nmb, 1) 

# 6. 精确对角化 (ED) 求解前 30 个低能态
result = []
for lz = 0 : 1
    bs = SBasis(cfs[lz])
    hmt = SOperator(bs, tms_hmt)
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, 30)

    l2 = SOperator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ st[:, i]' * l2_mat * st[:, i] for i in eachindex(enrg) ] 

    for i in eachindex(enrg)
        push!(result, [enrg[i], l2_val[i]])
    end
end

# 7. 能谱归一化与 CFT 缩放维度 (Scaling Dimensions Δ) 计算
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  # 单位算符 \mathbb{I} 的真空能量 (Δ = 0)

# 在 Super-Ising 中，利用应力张量 T (l=2) 或拟合标定能谱
# 导出归一化后的 [缩放维度 Δ, 原始能量 E, 角动量 L^2]
spec = [ round.([ (st[1] - enrg_0) ; st] .+ √eps(Float64), digits = 6) for st in result ] 

# 打印输出低能 Super-Ising 能谱
display(permutedims(hcat(spec...)))
