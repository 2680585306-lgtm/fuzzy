# model.jl
# 3D Super-Ising 超对称共形场论（SCFT）能谱计算 (N_{mf} = 7 )

using FuzzifiED
using FuzzifiED.Fuzzifino

FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸 (改为 N_{mf} = 7 以避免内存溢出崩溃)[cite: 1, 4]
nmf = 7               # 费米子单粒子轨道数 N_{mf} = 7[cite: 1, 4]
nof = 2 * nmf         # 2 个费米子 Flavor (f0, f1)，总费米子轨道数 = 14[cite: 4]
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 6[cite: 4]
nob = nmb             # 1 个玻色子 Flavor[cite: 4]

# 2. 构造守恒量子数 (电荷 U(1)_e 与角动量 L_z)[cite: 4]
qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 1) + SQNDiag(GetLz2QNDiag(nmf, 2), nob)
]

# 生成指定扇区的基底构型：总电荷数 N_e = N_{mf}，分别求解 L_z = 0 和 L_z = 1/2[cite: 4]
cfs = Dict{Int64, SConfs}()
for lz = 0 : 1 
    cfs[lz] = SConfs(nof, nob, nmf, [nof, lz], qnd)
end 

# 3. 构造微观模算符 (SMod)[cite: 4]
amd_f0f0 = GetFermionSMod(nmf, 2, 1) * GetFermionSMod(nmf, 2, 1) 
amd_f1f0 = GetFermionSMod(nmf, 2, 2) * GetFermionSMod(nmf, 2, 1) 
amd_f0b  = GetFermionSMod(nmf, 2, 1) * GetBosonSMod(nmb, 1, 1)   
amd_f1f1 = GetFermionSMod(nmf, 2, 2) * GetFermionSMod(nmf, 2, 2)  
amd_bb   = GetBosonSMod(nmb, 1, 1)   * GetBosonSMod(nmb, 1, 1)   

# 4. 缩合构造相互作用项 (ContractMod)[cite: 4]
tms_hop = ContractMod(amd_f0f0', amd_f0b, nmf - 2)      
tms_u   = ContractMod(amd_f1f0', amd_f1f0, nmf - 2)     
tms_g   = ContractMod(amd_f1f0', amd_f0b, nmf - 3/2)    

tms_h   = STerms(GetF1F0STerms(nmf, 2, 1, 2))            
tms_m1  = STerms(GetFermionNTerms(nmf, 2, 2, nob))       
tms_mb  = STerms(GetBosonNTerms(nof, nmb, 1, 1))         
tms_e2  = GetE2STerms(nmf, 2, nmb, 1)                    

# 5. 组装 Super-Ising 哈密顿量 (使用论文 Table 2 中 N_{mf}=7 的参数)[cite: 1, 4]
t = 1.5; U = 0.25; g = 1.0
h = 0.0698; mu1 = 0.0774; mub = 0.0722   # <--- 更新为 N_{mf}=7 的最佳参数[cite: 1, 4]

tms_hmt = SimplifyTerms(
    1.0 * tms_e2
    + t * (tms_hop + tms_hop')
    + U * tms_u
    + g * (tms_g + tms_g')
    - h * tms_h
    - mu1 * tms_m1
    - mub * tms_mb
)

# 构造总角动量平方算符 L^2[cite: 4]
tms_l2 = GetL2STerms(nmf, 2, nmb, 1) 

# 6. 精确对角化 (ED) 求解前 20 个低能态[cite: 4]
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

# 7. 能谱排序与输出[cite: 4]
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  

spec = [ round.([ (st[1] - enrg_0) ; st] .+ √eps(Float64), digits = 6) for st in result ] 

display(permutedims(hcat(spec...)))
