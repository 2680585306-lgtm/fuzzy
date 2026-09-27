using FuzzifiED
using LinearAlgebra

# -------------------------------------------------------------------
# 1. 系统参数设置 (Super-Ising SCFT)
# -------------------------------------------------------------------
nmf = 6                  # 费米子单粒子轨道数 N_mf
nmb = nmf - 1            # 玻色子单粒子轨道数 N_mb = N_mf - 1 (角动量错配 1/2)
nof = 2 * nmf            # 费米子总轨道数 (2 个 Flavor: f0, f1)
nob = nmb                # 玻色子总轨道数 (1 个 Flavor: b)

# N_mf = 6 时的 Super-Ising 临界点微扰参数 (参考论文 Table 2)
t = 0.5
U = 0.1
g = 0.2
h = 0.0662
mu1 = 0.0793
mub = 0.0615

# -------------------------------------------------------------------
# 2. 定义守恒量子数 (QND) 与 构造 Hilbert 空间基底
# -------------------------------------------------------------------
# (1) 费米子总数守恒
qnd_N = GetNeSQNDiag(nof, nob)

# (2) 总角动量 Lz 守恒: 包含了 2*f0 + 2*f1 + 2*b 的角动量贡献
# f0, f1 占据 (nmf-1)/2 的 spin，玻色子 b 占据 (nmb-1)/2 的 spin
qnd_Lz = GetBosonLz2SQNDiag(nof, nmb, 1) + SQNDiag(GetLz2QNDiag(nmf, 2), nob)

qnd = [qnd_N, qnd_Lz]

# 设定粒子数扇区 Ne = nmf (半填充)
Ne = nof ÷ 2

# 构造 Hilbert 空间配置
cfs = Dict{Int, SConfs}()
bases = Dict{Int, SBasis}()

# 扫描 Lz 扇区
for lz in -10:10
    # 注意：SConfs 参数需明确指定多 Flavor 结构
    cfs[lz] = SConfs(nof, nob, nmf, [Ne, lz], qnd)
    if length(cfs[lz]) > 0
        bases[lz] = SBasis(cfs[lz])
    end
end

# -------------------------------------------------------------------
# 3. 构造算子 (SMod) 与 哈密顿量微扰项缩合
# -------------------------------------------------------------------
# 构造多 Flavor 的费米子与玻色子 SMod 算子
# f0: Flavor 1, f1: Flavor 2
amd_f0 = GetFermionSMod(nmf, 2, 1)
amd_f1 = GetFermionSMod(nmf, 2, 2)
amd_b  = GetBosonSMod(nmb, 1, 1)

# --- 算子缩合 ---
# (a) 粒子密度项: n_e = f0^\dagger f0 + f1^\dagger f1
tms_n0 = ContractMod(amd_f0', amd_f0, 0)
tms_n1 = ContractMod(amd_f1', amd_f1, 0)
tms_ne = tms_n0 + tms_n1
tms_e2 = tms_ne * tms_ne

# (b) 超对称跃迁项 / 动能项
tms_hop = ContractMod(amd_f0', amd_b, 0)

# (c) 玻色子与费米子相互作用项
tms_nb = ContractMod(amd_b', amd_b, 0)
tms_nx = ContractMod(amd_f1', amd_f0, 0) + ContractMod(amd_f0', amd_f1, 0)  # n_x = f1^\dagger f0 + f0^\dagger f1

tms_u = tms_nx * tms_nx
tms_g = tms_nx * tms_nb

# (d) 组合生成总哈密顿量项
tms_hmt = SimplifyTerms(
    1.0 * tms_e2 +
    t * (tms_hop + tms_hop') +
    U * tms_u +
    g * tms_g -
    h * tms_nx -
    mu1 * tms_n1 -
    mub * tms_nb
)

# -------------------------------------------------------------------
# 4. 对角化求解能谱
# -------------------------------------------------------------------
println("==========================================")
println(" Super-Ising SCFT Spectrum (N_mf = $nmf) ")
println("==========================================")

for lz in sort(collect(keys(bases)))
    basis = bases[lz]
    dim = length(basis)
    if dim == 0
        continue
    end
    
    # 将抽象项转换为矩阵并求解前 5 个特征值
    hmt_mat = OpMat(tms_hmt, basis)
    n_states = min(5, dim)
    
    vals, _ = GetEigensystem(hmt_mat, n_states)
    
    println("Lz = $(lz/2) | Dim = $dim | Energies:")
    for v in vals
        println("  ", round(real(v), digits=6))
    end
end
