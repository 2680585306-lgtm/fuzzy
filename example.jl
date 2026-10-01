using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra
FuzzifiED.ElementType = Float64
≈(x, y) = abs(x - y) < √eps(Float64)

# 1. 设定系统尺寸与单粒子轨道数[cite: 1]
nmf = 6               # 费米子单粒子轨道数 N_{mf} = 2q + 1[cite: 1]
nof = 2 * nmf         # 2 个费米子 Flavor (f0 与 f1)，总费米子轨道数为 2 * N_{mf}[cite: 1]
nmb = nmf - 1         # 玻色子单粒子轨道数 N_{mb} = 2q (角动量与费米子相差 1/2)[cite: 1]
nob = 2 * nmb         # 2 个玻色子 Flavor (bp 与 bm，总轨道数为 2 * N_{mb})[cite: 1]

# 2. 构造守恒量子数 (电荷 U(1)_e、角动量 L_z 以及 O(2) 内部生成元 S_z)[cite: 1]
f_sz = zeros(Int64, nof)
b_sz = repeat([1, -1], nmb)
sz_sqnd = SQNDiag("Sz", f_sz, b_sz)

qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 2) + SQNDiag(GetLz2QNDiag(nmf, 2), nob),
    sz_sqnd
]

# 生成指定扇区的基底构型：总电荷数 N_e = N_{mf}
max_sz = 2            # 低能态主要分布在 sz = 0 (标量单态) 和 sz = 1 (双重态)
cfs = Dict{Tuple{Int64, Int64}, SConfs}()
for lz in 0 : 1 
    for sz in 0 : max_sz
        if mod(lz, 2) == 0
            cfs[(lz, sz)] = SConfs(nof, nob, nmf, [nmf, lz, 2*sz], qnd)
        else
            cfs[(lz, sz)] = SConfs(nof, nob, nmf, [nmf, lz, Int(2*sz+1)], qnd)
        end
    end
end 

# 3. 组装 Super-Ising 哈密顿量 (手征玻色子基底 bp, bm)[cite: 1]
FuzzifiED.ObsNormRadSq = Float64(nmf)
t = 1.5; U = 0.25; g = 1.0; m = 0.0805; p = 0.0602; q = 0.0671
f0 = GetFermionSObs(nmf, 2, 1)
f1 = GetFermionSObs(nmf, 2, 2)
bp = GetBosonSObs(nmb, 2, 1)  # 手征玻色子 b_+ (Sz = +1)
bm = GetBosonSObs(nmb, 2, 2)  # 手征玻色子 b_- (Sz = -1)

n0 = StoreComps(f0' * f0)
n1 = StoreComps(f1' * f1)
nbp = StoreComps(bp' * bp)
nbm = StoreComps(bm' * bm)
nb_tot = StoreComps(nbp + nbm)
nx = StoreComps(f0' * f1 + f1' * f0)
nr = StoreComps(n0 + n1 + nb_tot)

etap = StoreComps(f0' * bp)
etam = StoreComps(f0' * bm)
@assert abs(etap.s2) == 1 && abs(etam.s2) == 1 "eta must have spin weight +/-1/2"

R = sqrt(nmf)
Detap = let e = etap, R = R
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

Detam = let e = etam, R = R
    SSphereObs(-e.s2, e.l2m,
        (l2, m2) -> (-e.s2 * (l2 + 1) / (2R)) * e.get_comp(l2, m2))
end

pair_obs = etap * Detam + etam * Detap
@assert pair_obs.s2 == 0 "Pair conversion must be a rotational scalar"

tms_hop = SimplifyTerms(GetIntegral(pair_obs))

tms_int = SimplifyTerms(
    [(GetIntegral(nr * nr));
        t * tms_hop; t * tms_hop';
        U * (GetIntegral(nx * Laplacian(nx)));
        g * (GetIntegral(nx * nb_tot))],
)

tms_hmt = SimplifyTerms(
    tms_int
    - q * GetIntegral(nx) - m * GetIntegral(n1) - p * GetIntegral(nb_tot),
)

tms_l2 = GetL2STerms(nmf, 2, nmb, 2) 

# 4. 构造 O(2) 生成元算符 
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

N = 2
tms_c2 = on_C2_terms(nmb, N)

# 5. 精确对角化 (ED) 求解低能态并测量量子数
result = []
n_states = 12  # 适度增加单扇区求解态数以确保覆盖到自旋 2 扇区

for (lz, sz) in sort(collect(keys(cfs)))
    bs = SBasis(cfs[(lz, sz)])
    hmt = SOperator(bs, tms_hmt)
    hmt_mat = OpMat(hmt)
    enrg, st = GetEigensystem(hmt_mat, n_states)

    l2 = SOperator(bs, tms_l2)
    l2_mat = OpMat(l2)
    l2_val = [ real(st[:, i]' * (l2_mat * st[:, i])) for i in eachindex(enrg) ]

    c2_op = SOperator(bs, tms_c2)
    c2_mat = OpMat(c2_op)
    c2_val = [ real(st[:, i]' * c2_mat * st[:, i]) for i in eachindex(enrg) ]

    for i in eachindex(enrg)
        push!(result, [enrg[i], l2_val[i], c2_val[i], lz, sz])
    end
end

# 6. 能谱排序、稳健识别能动张量 T 并定标输出
sort!(result, by = st -> real(st[1]))
enrg_0 = result[1][1]  # 真空能量 (Δ = 0)[cite: 1]

# 放宽判据容差至 0.1，防止小尺寸数值截断误差导致遗漏
T_candidates = filter(st -> abs(st[2] - 6.0) < 0.1 && abs(st[3] - 0.0) < 0.1, result)

if isempty(T_candidates)
    println("【警告】未找到满足 L^2 ≈ 6 且 O(2) Casimir ≈ 0 的候选态！")
    display(result)
else
    println("="^85)
    println("筛选到 $(length(T_candidates)) 个处于 L^2 ≈ 6, Sz = 0 扇区的候选态：")
    for (idx, cand) in enumerate(T_candidates)
        println("  候选 $(idx): 能量 E = $(round(cand[1], digits=6)), L^2 = $(round(cand[2], digits=4)), C_2 = $(round(cand[3], digits=4))")
    end

    # 判定选择逻辑：区分 ∂²σ (Δ ≈ 2.58) 与 T (Δ = 3.00)[cite: 1]
    if length(T_candidates) >= 2
        ratio = (T_candidates[1][1] - enrg_0) / (T_candidates[2][1] - enrg_0)
        # 理论间距比值: 2.584 / 3.000 ≈ 0.861[cite: 1]
        if 0.75 < ratio < 0.95
            println("-> 检测到两个激发态能量比值为 $(round(ratio, digits=3)) ≈ 2.58/3.0[cite: 1]：")
            println("   候选 1 确认为标量二次导数态 ∂²σ (Δ ≈ 2.58)[cite: 1]")
            println("   选定【候选 2】作为真正的能动张量 T (Δ = 3.00)[cite: 1]")
            enrg_T = T_candidates[2][1]
        else
            println("-> 能量比值未落入典型 ∂²σ 区间，默认选取候选 1 作为 T (enrg_T = $(round(T_candidates[1][1], digits=6)))")
            enrg_T = T_candidates[1][1]
        end
    else
        println("-> 当前解空间仅包含 1 个自旋 2 态，选定【候选 1】作为能动张量 T")
        enrg_T = T_candidates[1][1]
    end

    println("最终定标基准 enrg_T = $(round(enrg_T, digits=6))")
    println("-"^85)
    println("输出说明: [标度维数 Δ, 原始能量 E, 角动量 L^2, O(2) Casimir, 2*Lz, Sz]")
    println("-"^85)

    spec = [
        [
            round(3 * (st[1] - enrg_0) / (enrg_T - enrg_0), digits = 6),
            round(st[1], digits = 6),
            round(st[2], digits = 4),
            round(Int, round(st[3])),
            st[4],
            st[5]
        ] for st in result
    ]
    display(permutedims(hcat(spec...)))
end
