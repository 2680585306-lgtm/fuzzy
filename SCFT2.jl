# super_ising_spectrum.jl
# 计算 3D Super-Ising 超对称共形场论（SCFT）的能谱 (arXiv:2509.08038 Section 8)

using FuzzifiED
using FuzzifiED.Fuzzifino

FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸与单粒子轨道数[cite: 1]
nmf = 9               # 费米子单粒子轨道数 N_{mf} = 2q + 1 (取 N_{mf} = 9)[cite: 1]
nof = 2 * nmf         # 2 个费米子 Flavor (f0 与 f1)，总费米子轨道数为 2 * N_{mf}[cite: 1]
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q (角动量与费米子相差 1/2)[cite: 1]
nob = nmb             # 1 个玻色子 Flavor[cite: 1]

# 2. 构造守恒量子数 (电荷 U(1)_e 与角动量 L_z)[cite: 1]
qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 1) + SQNDiag(GetLz2QNDiag(nmf, 2), nob)
]

# 生成指定扇区的基底构型：总电荷数 N_e = N_{mf}，分别求解 L_z = 0 和 L_z = 1/2[cite: 1]
cfs = Dict{Int64, SConfs}()
for lz = 0 : 1 
    cfs[lz] = SConfs(nof, nob, nmf, [nmf, lz], qnd)
end 
 
# 3. 构造微观模算符 (SMod)[cite: 1]
amd_f0f0 = GetFermionSMod(nmf, 2, 1) * GetFermionSMod(nmf, 2, 1)  # f0-f0 作用[cite: 1]
amd_f1f0 = GetFermionSMod(nmf, 2, 2) * GetFermionSMod(nmf, 2, 1)  # f1-f0 作用 (对应 σ 算符)[cite: 1]
amd_f0b  = GetFermionSMod(nmf, 2, 1) * GetBosonSMod(nmb, 1, 1)   # f0-b 作用  (对应 χ 算符)[cite: 1]
amd_bb = GetBosonSMod(nmb, 1, 1) * GetBosonSMod(nmb, 1, 1)
amd_f1f1 = GetFermionSMod(nmf, 2, 2) * GetFermionSMod(nmf, 2, 2) 
amd_f1b = GetFermionSMod(nmf, 2, 2) * GetBosonSMod(nmb, 1, 1) 
amd_f0 = GetFermionSMod(nmf, 2, 1) 
amd_f1 = GetFermionSMod(nmf, 2, 2) 
amd_b = GetBosonSMod(nmb, 1, 1)


# 4. 缩合构造相互作用项 (ContractMod)[cite: 1]
tms_hop = ContractMod(amd_f0f0', amd_f0b, nmf - 2)      # 动能对转换项 \eta D_+ \eta (t 项)[cite: 1]
tms_u   = ContractMod(amd_f1f0', amd_f1f0, nmf - 2)     # 标量梯度导数相互作用 n_x \nabla^2 n_x (U 项)[cite: 1]
tms_g   = ContractMod(amd_f1b', amd_f0b, nmf - 3/2)    # Yukawa 耦合项 n_x n_b (g 项)[cite: 1]
tms_nx = ContractMod(amd_f1', amd_f0, 0)
tms_n1 = ContractMod(amd_f1', amd_f1, 0)  # f1† f1
tms_nb = ContractMod(amd_b', amd_b, 0)

# 5. 组装 Super-Ising 哈密顿量 
t = 1.5; U = 0.25; g = 1.0; m = 0.0809; p = 0.0776; q = 0.0680

tms_hmt = SimplifyTerms(
    t * (tms_hop + tms_hop')
    + U * tms_u
    + g * (tms_g + tms_g')
    - m * tms_n1
    - p * tms_nb
    - q * (tms_nx +tms_nx')
)

# 构造总角动量平方算符 L^2 
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

# 7. 能谱排序与输出
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  # 单位算符 \mathbb{I} 的真空能量 (Δ = 0)[cite: 1]

# 导出 [能量间隔 (E - E_0), 原始能量 E, 角动量 L^2][cite: 1]
spec = [ round.([ (st[1] - enrg_0) ; st] .+ √eps(Float64), digits = 6) for st in result ] 

display(permutedims(hcat(spec...)))
