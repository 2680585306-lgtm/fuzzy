# SCFT.jl - 完全使用基础 SMod 构造的 Super-Ising SCFT 代码 (N_{mf} = 6)

using FuzzifiED
using FuzzifiED.Fuzzifino

FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸 (N_{mf} = 6)
nmf = 6               # 费米子单粒子轨道数
nof = 2 * nmf         # 2 个费米子 Flavor (f0, f1)
nmb = nmf - 1         # 玻色子单粒子轨道数
nob = nmb             # 1 个玻色子 Flavor

# 2. 构造守恒量子数
qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 1) + SQNDiag(GetLz2QNDiag(nmf, 2), nob)
]

cfs = Dict{Int64, SConfs}()
for lz = 0 : 1 
    cfs[lz] = SConfs(nof, nob, nmf, [nmf, lz], qnd)
end 

# ==============================================================================
# 3. 基础单粒子/双粒子 amd (SMod) 定义
# ==============================================================================

# 单粒子算符 (1-body SMod)
amd_f0 = GetFermionSMod(nmf, 2, 1)    # 费米子 f0
amd_f1 = GetFermionSMod(nmf, 2, 2)    # 费米子 f1
amd_b  = GetBosonSMod(nmb, 1, 1)      # 玻色子 b

# 双粒子算符 (2-body SMod)
amd_f0f0 = amd_f0 * amd_f0 
amd_f1f0 = amd_f1 * amd_f0 
amd_f0b  = amd_f0 * amd_b   
amd_bb   = amd_b  * amd_b   

# ==============================================================================
# 4. 相互作用项与单体项缩合 (完全使用 ContractMod)
# ==============================================================================

# (1) 相互作用二体项
tms_hop = ContractMod(amd_f0f0', amd_f0b, nmf - 2)      
tms_u   = ContractMod(amd_f1f0', amd_f1f0, nmf - 2)     
tms_g   = ContractMod(amd_f1f0', amd_f0b, nmf - 3/2)    

# (2) 横向场项: f1^\dagger * f0 (单体混合)
tms_h   = ContractMod(amd_f1', amd_f0, 0)

# (3) 粒子数/密度项: n = c^\dagger * c 或 b^\dagger * b (J=0 缩合)
tms_m1  = ContractMod(amd_f1', amd_f1, 0)    # f1 粒子数密度
tms_mb  = ContractMod(amd_b',  amd_b,  0)    # 玻色子 b 粒子数密度

# (4) 总电荷平方项 N_e^2: (n_f0 + n_f1 + n_b)^2
tms_nf0 = ContractMod(amd_f0', amd_f0, 0)
tms_ne  = tms_nf0 + tms_m1 + tms_mb
tms_e2  = tms_ne * tms_ne

# ==============================================================================
# 5. 组装 Super-Ising 哈密顿量与计算
# ==============================================================================
t = 1.5; U = 0.25; g = 1.0
h = 0.0662; mu1 = 0.0793; mub = 0.0615[cite: 1]

tms_hmt = SimplifyTerms(
    1.0 * tms_e2
    + t * (tms_hop + tms_hop')
    + U * tms_u
    + g * (tms_g + tms_g')
    - h * (tms_h + tms_h')
    - mu1 * tms_m1
    - mub * tms_mb
)

# 构造角动量 L^2 算符
tms_l2 = GetL2STerms(nmf, 2, nmb, 1) 

# 6. 对角化求解
result = []
for lz = 0 : 1
    bs = SBasis(cfs[lz])
    hmt = SOperator(bs, tms_hmt)
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, 20)

    l2 = SOperator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ st[:, i]' * l2_mat * st[:, i] for i in eachindex(enrg) ] 

    for i in eachindex(enrg)
        push!(result, [enrg[i], l2_val[i]])
    end
end

# 7. 能谱输出
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  

spec = [ round.([ (st[1] - enrg_0) ; st] .+ √eps(Float64), digits = 6) for st in result ] 

display(permutedims(hcat(spec...)))
