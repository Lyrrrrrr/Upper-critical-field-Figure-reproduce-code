"""
Week 5 任务A2：固定Tc，改变费米能级，计算V(μ)关系
Task A2: Fix Tc, vary Fermi level (chemical potential), calculate V(μ)

物理意义：
- 在不同的费米能级位置，维持相同Tc所需的配对强度V
- 反映态密度N(E_F)的变化效应

关键：K和-K谷的Zeeman劈裂必须反号（时间反演对称性）
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.constants import k as k_B, e
import os

# ========================================================================
# 物理参数
# ========================================================================
a = 3.18                    # 晶格常数 (Å)
t1 = 146e-3                 # 最近邻跳跃 (eV)
t2 = -0.40 * t1             # 次近邻
t3 = 0.25 * t1              # 第三近邻
mu_guess = 0.0              # 参考化学势

k_B_eV = k_B / e            # 玻尔兹曼常数 (eV/K)
mu_B_eVT = 5.7883818012e-5  # 玻尔磁子 (eV/T)

# ========================================================================
# 高对称点
# ========================================================================
K = np.array([0.0, 4*np.pi/(3*a)])                  # K点
Gamma = np.array([0.0, 0.0])                        # Γ点
M = np.array([np.pi/(np.sqrt(3)*a), np.pi/a])      # M点

# ========================================================================
# 能带色散（紧束缚模型）
# ========================================================================
def eps_Mo(kx, ky, mu):
    """MoS2导带色散关系"""
    return (2*t1*(np.cos(ky*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(0.5*ky*a))
            +2*t2*(np.cos(np.sqrt(3)*kx*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(1.5*ky*a))
            +2*t3*(np.cos(2*ky*a) + 2*np.cos(np.sqrt(3)*kx*a)*np.cos(ky*a))-mu)

# ========================================================================
# Zeeman型自旋轨道耦合（SOI）
# ========================================================================
def core_g(kx, ky):
    """g向量的核心部分"""
    return np.sin(ky*a) - 2*np.cos(np.sqrt(3)/2*kx*a)*np.sin(0.5*ky*a)

def f_k(kx, ky):
    """f(k) = |core_g(k)|"""
    return np.abs(core_g(kx, ky))

def F_k(kx, ky, fK, beta):
    """形状因子 F(k) = β·tanh[f(K)-f(k)] - 1"""
    return beta * np.tanh(fK - f_k(kx, ky)) - 1.0

def gzz(kx, ky, fK, beta):
    """完整的g_z(k) = F(k)·core_g(k)"""
    return F_k(kx, ky, fK, beta) * core_g(kx, ky)

# ========================================================================
# β参数标定
# ========================================================================
def find_kF_on_path(E_F):
    """在K→M路径上找费米动量k_F"""
    def path_around_K(smax=0.2, N=301):
        v_GK = K - Gamma
        v_KM = M - K
        L_GK = np.linalg.norm(v_GK)
        L_KM = np.linalg.norm(v_KM)
        u1 = np.linspace(1 - smax/L_GK, 1.0, N)
        u2 = np.linspace(0.0, smax/L_KM, N)
        k_GK = Gamma + u1[:,None]*v_GK
        k_KM = K + u2[:,None]*v_KM
        kx = np.concatenate([k_GK[:,0], k_KM[:,0]])
        ky = np.concatenate([k_GK[:,1], k_KM[:,1]])
        s_GK = -(1.0 - u1) * L_GK
        s_KM = u2 * L_KM
        s = np.concatenate([s_GK, s_KM])
        return kx, ky, s
    
    kx_l, ky_l, s_l = path_around_K(0.2, 301)
    mask_R = s_l >= 0
    E_vals = eps_Mo(kx_l[mask_R], ky_l[mask_R], mu_guess) - E_F
    kF_idx = np.argmin(np.abs(E_vals))
    return kx_l[mask_R][kF_idx], ky_l[mask_R][kF_idx]

def beta_from_target(target_meV, E_F):
    """标定β参数，使费米面处自旋劈裂为target_meV"""
    kF_kx, kF_ky = find_kF_on_path(E_F)
    fK = f_k(K[0], K[1])
    coreK = abs(core_g(K[0], K[1]))
    alpha = (3e-3) / (2.0*max(coreK, 1e-18))
    
    betas = np.linspace(0.5, 200.0, 2001)
    spls = np.array([2*alpha*abs(gzz(kF_kx, kF_ky, fK, b)) for b in betas])
    return float(betas[np.argmin(np.abs(spls - target_meV*1e-3))]), alpha

# ========================================================================
# 费米面shell构建
# ========================================================================
def k_patch(center, rad=0.20, Nk=91):
    """在center周围创建圆形k点patch"""
    qx = np.linspace(-rad, rad, Nk)
    qy = np.linspace(-rad, rad, Nk)
    Qx, Qy = np.meshgrid(qx, qy, indexing='xy')
    mask = (Qx**2 + Qy**2 <= rad**2)
    kx = center[0] + Qx[mask]
    ky = center[1] + Qy[mask]
    return kx, ky

def build_FS_shell(mu_guess, EF, rad=0.20, Nk=91, n_keep=1200):
    """构建K和-K谷的费米面shell"""
    kxK, kyK = k_patch(K, rad, Nk)
    kxKm, kyKm = k_patch(-K, rad, Nk)
    xiK = eps_Mo(kxK, kyK, mu_guess) - EF
    xiKm = eps_Mo(kxKm, kyKm, mu_guess) - EF
    idxK = np.argsort(np.abs(xiK))[:n_keep]
    idxKm = np.argsort(np.abs(xiKm))[:n_keep]
    return (kxK[idxK], kyK[idxK]), (kxKm[idxKm], kyKm[idxKm])

# ========================================================================
# 超导磁化率计算 - 关键：K和-K谷反号
# ========================================================================
def chi_singlet_vectorized(T, H, alpha, gz_func, kK, kKm, mu_guess, EF, Nw):
    """
    计算不可约超导磁化率
    关键修正：-K谷的Zeeman劈裂必须反号（时间反演对称性）
    """
    kxK, kyK = kK
    kxKm, kyKm = kKm
    Nk = kxK.size
    
    # 计算能量和Zeeman场
    xiK = eps_Mo(kxK, kyK, mu_guess) - EF
    lamK = alpha * gz_func(kxK, kyK)          # K谷: +ΔZ
    
    xiKm = eps_Mo(kxKm, kyKm, mu_guess) - EF
    lamKm = -alpha * gz_func(kxKm, kyKm)      # -K谷: -ΔZ (反号!)
    
    # 松原频率
    wn = (2*np.arange(Nw)+1) * np.pi * k_B_eV * T
    
    # K谷的Green函数
    A = 1j*wn[:,None] - xiK[None,:]
    Bx = -mu_B_eVT * H
    Bz = -lamK[None,:]
    denom = (A*A - Bx*Bx - Bz*Bz)
    Guu = (A - Bz)/denom
    Gdd = (A + Bz)/denom
    Gud = (-Bx)/denom
    Gdu = (-Bx)/denom
    
    # -K谷的Green函数
    A2 = -1j*wn[:,None] - xiKm[None,:]
    Bz2 = -lamKm[None,:]
    denom2 = (A2*A2 - Bx*Bx - Bz2*Bz2)
    Guu_m = (A2 - Bz2)/denom2
    Gdd_m = (A2 + Bz2)/denom2
    Gud_m = (-Bx)/denom2
    Gdu_m = (-Bx)/denom2
    
    # 求和
    term = Guu*Gdd_m - Gdu*Gud_m
    chi = (k_B_eV*T) * np.real(term.sum()) / Nk
    return float(chi)

# ========================================================================
# 任务A2核心函数：固定Tc，求解V
# ========================================================================
def determine_V_for_fixed_Tc(Tc_fixed, alpha, gz_func, kK, kKm, mu_guess, EF, Nw):
    """
    对于给定的Tc和费米能级EF，反推所需的配对强度V
    
    物理：V · χ_sc(Tc, 0, EF) = 1
    求解：V = 1 / χ_sc(Tc, 0, EF)
    """
    chi = chi_singlet_vectorized(Tc_fixed, 0.0, alpha, gz_func, 
                                 kK, kKm, mu_guess, EF, Nw)
    return 1.0 / chi

# ========================================================================
# 主程序：任务A2
# ========================================================================
if __name__ == "__main__":
    print("="*70)
    print("Week 5 任务A2：固定Tc，计算V与化学势μ的关系")
    print("Task A2: Fix Tc, calculate V(μ) relationship")
    print("="*70)
    
    # ------------------------------------------------------------------
    # 步骤1：设置参数
    # ------------------------------------------------------------------
    print("\n[步骤1] 参数设置")
    
    # 费米能级参考点
    E_K_raw = eps_Mo(K[0], K[1], mu_guess)
    E_F_ref = E_K_raw + 0.15  # 参考费米能级
    print(f"参考费米能级: E_F_ref = {E_F_ref:.4f} eV")
    
    # 自旋劈裂参数标定
    target_meV = 13.0  # 费米面处的Zeeman劈裂 (meV)
    beta_ref, alpha_ref = beta_from_target(target_meV, E_F_ref)
    print(f"自旋劈裂参数: α = {alpha_ref:.6f}, β = {beta_ref:.2f}")
    print(f"目标劈裂: Δ_Z(k_F) = {target_meV} meV")
    
    # 构建g_z函数
    fK = f_k(K[0], K[1])
    gz_func = lambda kx, ky: gzz(kx, ky, fK, beta_ref)
    
    # 固定的临界温度
    Tc_fixed = 6.5  # K (实验值)
    print(f"固定Tc: {Tc_fixed} K")
    
    # 计算参数
    Nw = 800  # 松原频率数量
    n_keep = 1200  # 费米面采样点数
    
    # ------------------------------------------------------------------
    # 步骤2：定义扫描范围
    # ------------------------------------------------------------------
    print(f"\n[步骤2] 定义费米能级扫描范围")
    
    scan_range = 0.2  # eV
    N_points = 51     # 扫描点数
    
    E_F_values = np.linspace(E_F_ref - scan_range, 
                            E_F_ref + scan_range, 
                            N_points)
    
    print(f"扫描范围: [{E_F_values[0]:.4f}, {E_F_values[-1]:.4f}] eV")
    print(f"扫描点数: {N_points}")
    
    # ------------------------------------------------------------------
    # 步骤3：对每个E_F计算所需的V
    # ------------------------------------------------------------------
    print(f"\n[步骤3] 计算V(E_F)关系...")
    print(f"对于每个费米能级E_F，求解使Tc={Tc_fixed}K的配对强度V")
    
    V_results = []
    chi_results = []  # 同时记录χ值，用于分析
    
    for E_F_val in tqdm(E_F_values, desc="扫描E_F"):
        # 构建该E_F下的费米面
        kK_new, kKm_new = build_FS_shell(mu_guess, E_F_val, n_keep=n_keep)
        
        # 计算磁化率
        chi_val = chi_singlet_vectorized(Tc_fixed, 0.0, alpha_ref, gz_func,
                                        kK_new, kKm_new, mu_guess, E_F_val, Nw)
        
        # 求解V：V = 1/χ
        V_val = 1.0 / chi_val
        
        V_results.append(V_val)
        chi_results.append(chi_val)
    
    V_results = np.array(V_results)
    chi_results = np.array(chi_results)
    
    # ------------------------------------------------------------------
    # 步骤4：结果分析
    # ------------------------------------------------------------------
    print(f"\n[步骤4] 结果摘要")
    
    idx_ref = N_points // 2  # 参考点索引
    idx_min = np.argmin(V_results)
    idx_max = np.argmax(V_results)
    
    print(f"\n配对强度V的变化范围:")
    print(f"  最小值: V = {V_results[idx_min]:.6f} @ E_F = {E_F_values[idx_min]:.4f} eV")
    print(f"  参考点: V = {V_results[idx_ref]:.6f} @ E_F = {E_F_ref:.4f} eV")
    print(f"  最大值: V = {V_results[idx_max]:.6f} @ E_F = {E_F_values[idx_max]:.4f} eV")
    print(f"  变化幅度: {(V_results[idx_max]/V_results[idx_min] - 1)*100:.1f}%")
    
    print(f"\n磁化率χ的变化范围:")
    print(f"  最小值: χ = {chi_results.min():.6f}")
    print(f"  最大值: χ = {chi_results.max():.6f}")
    
    print(f"\n物理解释:")
    print(f"  V最小处 → χ最大 → N(E_F)最大 → 态密度峰值")
    print(f"  V最大处 → χ最小 → N(E_F)最小 → 态密度谷值")
    
    # ------------------------------------------------------------------
    # 步骤5：绘制结果
    # ------------------------------------------------------------------
    print(f"\n[步骤5] 绘制图表")
    
    # 创建双面板图
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # === 面板1：V vs E_F ===
    ax1 = axes[0]
    ax1.plot(E_F_values, V_results, 'b-', linewidth=2.5, label='V(E_F)')
    ax1.axvline(E_F_ref, color='r', linestyle='--', linewidth=1.5, 
                label=f'ref point (E_F={E_F_ref:.3f} eV)')
    ax1.scatter([E_F_values[idx_min]], [V_results[idx_min]], 
               color='green', s=150, zorder=5, marker='v', 
               label=f'minV')
    ax1.scatter([E_F_values[idx_max]], [V_results[idx_max]], 
               color='red', s=150, zorder=5, marker='^', 
               label=f'maxV')
    
    ax1.set_xlabel('Fermi Energy $E_F$ (eV)', fontsize=13)
    ax1.set_ylabel('Pairing Interaction $V$ (dimensionless)', fontsize=13)
    ax1.set_title(f'Matching strength V vs fermi level (fix Tc = {Tc_fixed} K)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10, loc='best')
    
    # === 面板2：χ vs E_F ===
    ax2 = axes[1]
    ax2.plot(E_F_values, chi_results, 'g-', linewidth=2.5, 
            label=r'$\chi_{sc}$(E_F)')
    ax2.axvline(E_F_ref, color='r', linestyle='--', linewidth=1.5,
               label=f'ref point')
    ax2.scatter([E_F_values[idx_min]], [chi_results[idx_min]], 
               color='red', s=150, zorder=5, marker='v',
               label='Minχ (MaxV处)')
    ax2.scatter([E_F_values[idx_max]], [chi_results[idx_max]], 
               color='green', s=150, zorder=5, marker='^',
               label='Maxχ (minV处)')
    
    ax2.set_xlabel('Fermi Energy $E_F$ (eV)', fontsize=13)
    ax2.set_ylabel(r'Susceptibility $\chi_{sc}$ (dimensionless)', fontsize=13)
    ax2.set_title(r' susceptivity $\chi_{sc}$ vs fermi level', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # 保存到当前目录
    output_file = 'TaskA2_V_vs_FermiLevel.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 图片已保存: {output_file}")
    plt.show()
    
    # ------------------------------------------------------------------
    # 步骤6：保存数据
    # ------------------------------------------------------------------
    print(f"\n[步骤6] 保存数据")
    
    # 保存主要结果
    data = np.column_stack((E_F_values, V_results, chi_results))
    header = f"Tc_fixed = {Tc_fixed} K, Delta_Z = {target_meV} meV\n"
    header += "E_F(eV)  V(dimensionless)  chi_sc(dimensionless)"
    
    data_file = 'TaskA2_V_vs_EF_data.txt'
    np.savetxt(data_file, data, header=header, fmt='%.8f')
    print(f"✓ 数据已保存: {data_file}")
    
    # ------------------------------------------------------------------
    # 步骤7：物理解释
    # ------------------------------------------------------------------
    print(f"\n" + "="*70)
    print("物理解释：")
    print("="*70)
    print("""
1. V与χ的反比关系：V = 1/χ
   - χ大 → V小：高态密度区域，弱配对即可达到相同Tc
   - χ小 → V大：低态密度区域，需要强配对才能维持Tc

2. V(E_F)曲线形状：
   - 谷底：对应态密度N(E_F)的峰值（最佳掺杂）
   - 峰顶：对应态密度N(E_F)的谷值（远离最优）

3. 与任务A1的关系：
   - A1: 固定V，变E_F → 得到Tc(E_F)
   - A2: 固定Tc，变E_F → 得到V(E_F)
   - V(E_F)曲线是Tc(E_F)曲线的"倒影"

4. 实验意义：
   - 理解不同掺杂下的有效配对强度
   - 指导材料优化：在态密度高的区域更容易实现超导
    """)
    
    print("="*70)
    print("计算完成！")
    print("="*70)