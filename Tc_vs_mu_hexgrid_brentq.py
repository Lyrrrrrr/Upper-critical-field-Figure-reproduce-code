import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm
from scipy.constants import k as k_B, e
import csv






# Tight-binding model parameters
a = 3.18
t1 = 146e-3
t2 = -0.40 * t1
t3 = 0.25 * t1
mu = 0

# Physical constants in convenient units
mu_B_eVT = 5.7883818012e-5  # Bohr magneton in eV/T
k_B_eV = k_B / e           # Boltzmann constant in eV/K

# Zeeman-type spin-splitting parameters (initial values)
alpha = 0
beta = 0

# Define the critical temperature and convert it to units of eV
Tc_K = 6.5
Tc_eV = k_B_eV * Tc_K














# energy dispersion, corresponding to H_{kin} term
def eps_Mo(kx, ky, mu):
    # --- type-safety: ensure numpy arrays (fixes list*float TypeError) ---
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    return (2*t1*(np.cos(ky*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(0.5*ky*a))
            +2*t2*(np.cos(np.sqrt(3)*kx*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(1.5*ky*a))
            +2*t3*(np.cos(2*ky*a) + 2*np.cos(np.sqrt(3)*kx*a)*np.cos(ky*a)) - mu)


# g_z(k) for Zeeman-type SOI
def core_g(kx, ky):
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    
    # sin(ky a) - 2 cos(√3/2 kx a) sin(ky a/2)
    return np.sin(ky*a) - 2*np.cos(np.sqrt(3)/2*kx*a)*np.sin(0.5*ky*a)

def f_k(kx, ky):
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)

    return np.abs(core_g(kx, ky))

def F_k(kx, ky, fK, beta):
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    # F(k) = beta * tanh[ f(K) - f(k) ] - 1
    return beta * np.tanh(fK - f_k(kx, ky)) - 1.0

def gzz(kx, ky, fK, beta):
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    return F_k(kx, ky, fK, beta) * core_g(kx, ky)

# g_R(k) for Rashba-type SOI
def gR_core(kx, ky):
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    gx_core = -np.sin(ky*a) - np.cos(np.sqrt(3)/2*kx*a)*np.sin(0.5*ky*a)
    gy_core = np.sqrt(3) * np.sin(np.sqrt(3)/2*kx*a)*np.cos(0.5*ky*a)
    return gx_core, gy_core

def gR_vec(kx, ky, fK, beta):
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    Fk = F_k(kx, ky, fK, beta)
    gx_core, gy_core = gR_core(kx, ky)
    return Fk * gx_core, Fk * gy_core

def gR_mag(kx, ky, fK, beta):
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    gx, gy = gR_vec(kx, ky, fK, beta)
    return np.sqrt(gx*gx + gy*gy)


# High symmetry points
Gamma = np.array([0.0, 0.0])
K = np.array([0.0, 4*np.pi/(3*a)])
M = np.array([np.pi/(np.sqrt(3)*a), np.pi/a])

# Path, Γ→K (neg s) and K→M (pos s), near K
def path_around_K(smax=0.2, N=501):
    v_GK = K - Gamma
    v_KM = M - K
    L_GK = np.linalg.norm(v_GK)
    L_KM = np.linalg.norm(v_KM)

    u1 = np.linspace(1 - smax/L_GK, 1.0, N)   # Γ→K near K
    u2 = np.linspace(0.0, smax/L_KM, N)       # K→M near K

    k_GK = Gamma + u1[:,None]*v_GK
    k_KM = K     + u2[:,None]*v_KM

    s_GK = -(1.0 - u1) * L_GK    # [-smax, 0]
    s_KM =  (u2      ) * L_KM    # [ 0, +smax]

    kx = np.concatenate([k_GK[:,0], k_KM[:,0]])
    ky = np.concatenate([k_GK[:,1], k_KM[:,1]])
    s  = np.concatenate([s_GK,       s_KM])

    return kx, ky, s

kx, ky, s = path_around_K(smax=0.2, N=501)

# Align to E_F
mu_guess = 0.0
E_K_raw  = eps_Mo(0, 4*np.pi/(3*a), mu_guess)     # Energy at K point with initial mu_guess
E_F      = E_K_raw + 0.15
E = lambda kx,ky: eps_Mo(kx,ky,mu_guess)-E_F


kx_l, ky_l, s_l = path_around_K(0.2, 301)
mask_R = s_l >= 0
kF_idx = np.argmin(np.abs(E(kx_l[mask_R], ky_l[mask_R])))
kF_kx, kF_ky = kx_l[mask_R][kF_idx], ky_l[mask_R][kF_idx]

# alpha and beta calibration
fK    = f_k(K[0], K[1])
coreK = abs(core_g(K[0], K[1]))
alpha = (3e-3) / (2.0*max(coreK,1e-18))

def beta_from_target(target_meV):
    betas = np.linspace(0.5, 200.0, 2001)
    spls  = np.array([2*alpha*abs(gzz(kF_kx, kF_ky, fK, b)) for b in betas])
    return float(betas[np.argmin(np.abs(spls - target_meV*1e-3))])


# ============================================================================
# HEXAGONAL BRILLOUIN ZONE IMPLEMENTATION (与seg_1保持一致)
# ============================================================================

def in_hexagonal_BZ(kx, ky, a):
    """
    检查k点是否在六边形第一布里渊区内
    """
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    K_y = 4*np.pi/(3*a)
    ky_abs = np.abs(ky)
    kx_abs = np.abs(kx)
    slope = 1.0 / np.sqrt(3)
    cond_slanted = ky_abs <= K_y - slope * kx_abs
    kx_max = K_y * np.sqrt(3) / 2
    cond_vertical = kx_abs <= kx_max
    return cond_slanted & cond_vertical


def generate_hexagonal_BZ_grid(a, Nk):
    """
    生成严格在六边形第一布里渊区内的k点网格（与seg_1完全一致）
    """
    K_y = 4*np.pi/(3*a)
    kx_max_hex = K_y * np.sqrt(3) / 2
    ky_max_hex = K_y
    
    kx_vals = np.linspace(-kx_max_hex * 1.01, kx_max_hex * 1.01, Nk)
    ky_vals = np.linspace(-ky_max_hex * 1.01, ky_max_hex * 1.01, Nk)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals, indexing='xy')
    
    mask = in_hexagonal_BZ(kx_grid, ky_grid, a)
    
    kx_1BZ = kx_grid[mask]
    ky_1BZ = ky_grid[mask]
    N_k_in_BZ = mask.sum()
    
    return kx_1BZ, ky_1BZ, mask, N_k_in_BZ


# ============================================================================
# 生成六边形BZ网格（与seg_1保持一致的采样方式）
# ============================================================================
print("="*70)
print("HEXAGONAL BRILLOUIN ZONE SAMPLING FOR Tc CALCULATION")
print("="*70)

Nk_BZ = 500  # 六边形BZ采样密度（可调整）

kx_1BZ, ky_1BZ, hex_mask, N_k_in_BZ = generate_hexagonal_BZ_grid(a, Nk_BZ)

print(f"Generated hexagonal BZ grid:")
print(f"  Sampling density: {Nk_BZ} x {Nk_BZ} candidate grid")
print(f"  k-points inside 1st BZ: {N_k_in_BZ}")

# 预计算用于归一化
N_k_total = float(N_k_in_BZ)


# ============================================================================
# 使用六边形BZ全遍历的费米面采样
# ============================================================================

def build_FS_shell_hexBZ(EF, n_keep):
    """
    从六边形BZ内的所有k点中选择最接近费米面的点
    
    优点：
    - 自然包含所有3个等价K点和3个等价K'点的邻域
    - 与seg_1的DOS计算采样方式完全一致
    
    Parameters:
    -----------
    EF : float
        费米能级
    n_keep : int
        保留的k点数量（接近费米面的点）
    
    Returns:
    --------
    kx_FS, ky_FS : arrays
        接近费米面的k点坐标
    """
    # 使用全局的六边形BZ网格
    xi = eps_Mo(kx_1BZ, ky_1BZ, mu_guess) - EF
    
    # 按|xi|排序，选择最接近费米面的点
    idx_sorted = np.argsort(np.abs(xi))
    n_actual = min(n_keep, len(idx_sorted))
    idx_keep = idx_sorted[:n_actual]
    
    return kx_1BZ[idx_keep], ky_1BZ[idx_keep]


# ------------------------ Susceptibility calculation ------------------------
# 2x2 identity and Pauli matrices
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1],[1, 0]], dtype=complex)
sz = np.array([[1, 0],[0,-1]], dtype=complex)

# Encapsulated as g_z(kx, ky) without additional parameters
gz_func = lambda kx, ky: gzz(kx, ky, fK, beta)

# calculate χ using the formula with vectorization over k-grid
# 修改：使用六边形BZ全遍历的k点
def chi_singlet_vectorized(T, H, alpha_Z, alpha_R, gz_func, gR_func, kK, mu_guess, EF, Nw):
    """
    计算单重态配对susceptibility
    
    修改说明：
    - kK现在是(kx_array, ky_array)的tuple，包含六边形BZ内接近费米面的所有k点
    - 不再区分K和-K谷，因为六边形BZ遍历已自然包含所有等价点
    """
    # 处理输入格式
    if isinstance(kK, (tuple, list)) and len(kK) == 2:
        if isinstance(kK[0], (tuple, list)) and len(kK[0]) == 2:
            # 旧格式: ((kx1, ky1), (kx2, ky2))
            (kx1, ky1), (kx2, ky2) = kK
            kxK = np.concatenate([np.asarray(kx1, dtype=float), np.asarray(kx2, dtype=float)])
            kyK = np.concatenate([np.asarray(ky1, dtype=float), np.asarray(ky2, dtype=float)])
        else:
            # 新格式: (kx_array, ky_array)
            kxK = np.asarray(kK[0], dtype=float)
            kyK = np.asarray(kK[1], dtype=float)
    else:
        kxK, kyK = kK
        kxK = np.asarray(kxK, dtype=float)
        kyK = np.asarray(kyK, dtype=float)

    # Energy at k
    EK   = eps_Mo(kxK,  kyK,  mu_guess) - EF

    # Zeeman-type component DeltaZ at k
    DeltaZ_K  = alpha_Z * gz_func(kxK,  kyK)

    # Rashba components g_Rx, g_Ry at k
    gRxK, gRyK   = gR_func(kxK,  kyK)

    # Matsubara frequencies
    wn = (2*np.arange(Nw)+1) * np.pi * k_B_eV * T

    # use [:, None] and [None, :] to vectorize over k and omega_n. (broadcasting)
    A1   = 1j*wn[:,None] - EK[None,:]    # G(k, i omega_n)
    
    # Using symmetry properties: eps(-k) = eps(k)
    A2  = -1j*wn[:,None] - EK[None,:]   # G(-k, -i omega_n)

    # 3 components of effective field B at K
    Bx  = (-mu_B_eVT * H) + alpha_R * gRxK[None,:]      # B_x = -mu_B*H + alpha_R * g_Rx(k)
    By  = (alpha_R * gRyK)[None,:]                      # B_y = alpha_R * g_Ry(k)
    Bz  = (-DeltaZ_K)[None,:]                           # B_z = -Delta_Z(k)

    # 3 components of effective field B at -k (using symmetry properties)
    # Bx2: gRx(-k) = -gRx(k) (odd function)
    Bx2 = (-mu_B_eVT * H) - alpha_R * gRxK[None,:]
    # By2: gRy(-k) = gRy(k) (even function)
    By2 = (alpha_R * gRyK)[None,:]
    # Bz2: DeltaZ(-k) = -DeltaZ(k) (odd function)
    Bz2 = -(-DeltaZ_K)[None,:]

    # 2×2 green's function elements G(k, i omega_n)
    denominator1  = (A1*A1  - Bx*Bx  - By*By  - Bz*Bz)
    Guu    = (A1 - Bz)/denominator1
    Gdd    = (A1 + Bz)/denominator1
    Gud    = (Bx - 1j*By)/denominator1
    Gdu    = (Bx + 1j*By)/denominator1

    # 2×2 green's function elements G(-k, -i omega_n)
    denominator2 = (A2*A2 - Bx2*Bx2 - By2*By2 - Bz2*Bz2)
    Guu_m  = (A2 - Bz2)/denominator2
    Gdd_m  = (A2 + Bz2)/denominator2
    Gud_m  = (Bx2 - 1j*By2)/denominator2
    Gdu_m  = (Bx2 + 1j*By2)/denominator2

    # singlet pairing susceptibility
    term = Guu*Gdd_m - Gdu*Gud_m
    chi  = (k_B_eV*T) * np.real(term.sum()) / kxK.size
    return float(chi)

# find V from Tc at H=0
def determine_V(Tc, alpha_Z, alpha_R, gz_func, gR_func, kK, mu_guess, EF, Nw):
    chi_Tc0 = chi_singlet_vectorized(Tc, 0.0, alpha_Z, alpha_R, gz_func, gR_func, kK, mu_guess, EF, Nw)
    return 1.0 / chi_Tc0










# -*- coding: utf-8 -*-
# 说明: 横坐标改为 E - E_F,并仅从 -0.15 eV 起绘图

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------ Find Tc ----------------------------
# Find Tc at given H=0 using Brent's method (faster than bisection)
from scipy.optimize import brentq

def find_Tc(V, alpha_Z, alpha_R, gz_func, gR_func, kK, mu_guess, EF, Nw, Tmax=20.0, Tmin=0.01):
    # Define the root-finding function: f(T) = V*chi(T) - 1
    def f(T):
        chi = chi_singlet_vectorized(T, 0.0, alpha_Z, alpha_R, gz_func, gR_func, kK, mu_guess, EF, Nw)
        return V * chi - 1.0
    
    # Check if it's superconducting at all (at T_min)
    f_min = f(Tmin)
    if f_min < 0:
        return 0.0  # Not superconducting
    
    # Check if it's normal at Tmax
    f_max = f(Tmax)
    if f_max > 0:
        print(f"Warning: Still superconducting at Tmax={Tmax}K for EF={EF}. Increase Tmax.")
        return Tmax
    
    # Use Brent's method for root finding (faster than bisection)
    Tc = brentq(f, Tmin, Tmax, xtol=1e-4)
    return Tc

# ------------------------ Calculate Tc(mu) for fixed V ------------------------
print("\nCalibrating fixed V...")
target_meV_for_V = 13.0
beta_for_V = beta_from_target(target_meV_for_V)

gz_fun_fixed = lambda kx, ky, fK=fK, b=beta_for_V: gzz(kx, ky, fK, b)
gR_fun_fixed = lambda kx, ky, fK=fK, b=beta_for_V: gR_vec(kx, ky, fK, b)

Tc_initial = 6.5       # K
Nw = 1600              # Matsubara frequencies
n_keep_value = 6000    # FS sample size (从整个六边形BZ中选取)

# Zeeman-type coupling fixed
alpha_Z_fixed = alpha

# 使用六边形BZ全遍历方式构建费米面采样
print(f"Building FS shell from hexagonal BZ at E_F...")
kx_FS_calib, ky_FS_calib = build_FS_shell_hexBZ(E_F, n_keep_value)
kK_calib = (kx_FS_calib, ky_FS_calib)
print(f"FS sample size (for V calibration): {len(kx_FS_calib)}")

# EF scan setup
EF_center = E_F
EF_scan_range = 0.4      # eV
N_mu_points = 200
EF_range = np.linspace(EF_center - 0.15, EF_center + 0.5, N_mu_points)
dE_range = EF_range - EF_center  # 横坐标: E - E_F

# 仅从该位置开始绘图
x_min = -0.15
mask_global = dE_range >= x_min

# eta values
eta_list = [0.02]

# ============================================================================
# CSV EXPORT STORAGE - NEW ADDITION
# ============================================================================
csv_data_storage = {}  # Dictionary to store data for each eta
# ============================================================================

# Plot
plt.figure(figsize=(9, 6))
print(f"Starting Tc(E - E_F) calculation for eta list: {eta_list}")

for eta in eta_list:
    print(f"\n--- Processing eta = {eta:.2f} ---")

    # Rashba coupling for this eta
    alpha_R_current = eta * alpha_Z_fixed

    # Calibrate V at EF_center to satisfy Tc(EF_center) = Tc_initial for this eta
    V_calibrated = determine_V(Tc_initial, alpha_Z_fixed, alpha_R_current,
                               gz_fun_fixed, gR_fun_fixed,
                               kK_calib, mu_guess, EF_center, Nw)
    print(f"  Calibrated V = {V_calibrated:.6f} for eta = {eta:.2f}")
    print(V_calibrated)
    Tc_results_for_this_eta = []

    # Scan EF_range; 计算用全范围,绘图再截取
    desc = f"  Scanning E (eta={eta:.2f})"
    for EF_val in tqdm(EF_range, desc=desc):
        # 使用六边形BZ全遍历方式重建费米面采样
        kx_FS_new, ky_FS_new = build_FS_shell_hexBZ(EF_val, n_keep_value)
        kK_new = (kx_FS_new, ky_FS_new)

        # Find Tc using calibrated V
        Tc_new = find_Tc(V_calibrated, alpha_Z_fixed, alpha_R_current,
                         gz_fun_fixed, gR_fun_fixed,
                         kK_new, mu_guess, EF_val, Nw, Tmax=20.0)
        Tc_results_for_this_eta.append(Tc_new)

    # ============================================================================
    # STORE DATA FOR CSV EXPORT - NEW ADDITION
    # ============================================================================
    csv_data_storage[eta] = {
        'dE_full': dE_range.copy(),
        'Tc_full': np.array(Tc_results_for_this_eta).copy(),
        'V_calibrated': V_calibrated
    }
    # ============================================================================

    # 截取从 x_min 开始的部分
    dE_plot = dE_range[mask_global]
    Tc_plot = np.array(Tc_results_for_this_eta)[mask_global]

    # 绘图
    plt.plot(dE_plot, Tc_plot, lw=2.2,
             marker='o', markersize=3,
             label=rf'$\eta={eta:.2f}$')

# ============================================================================
# EXPORT DATA TO CSV FILES - NEW ADDITION
# ============================================================================
print("\n--- Exporting Tc vs mu data to CSV files ---")

for eta, data_dict in csv_data_storage.items():
    # Create filename for this eta value
    tc_filename = f'Tc_vs_mu_eta_{eta:.3f}.csv'
    
    # Write CSV file
    with open(tc_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header with metadata
        writer.writerow([f'# Tc vs mu data for eta = {eta:.3f}'])
        writer.writerow([f'# Calibrated V = {data_dict["V_calibrated"]:.8f}'])
        writer.writerow([f'# Tc_initial = {Tc_initial:.2f} K'])
        writer.writerow([f'# Hexagonal BZ sampling: {N_k_in_BZ} k-points'])
        writer.writerow([f'# FS shell size: {n_keep_value}'])
        writer.writerow(['mu (E-E_F) [eV]', 'Tc [K]'])
        
        # Write data
        for mu_val, tc_val in zip(data_dict['dE_full'], data_dict['Tc_full']):
            writer.writerow([f'{mu_val:.6f}', f'{tc_val:.6f}'])
    
    print(f"Exported Tc data for eta={eta:.3f} to {tc_filename}")

print("CSV export complete!\n")
# ============================================================================

# Finalize plot
plt.xlabel(r'$E - E_F$ (eV)')
plt.ylabel(r'$T_c$ (K)')
plt.title(r'$T_c$ vs. $E - E_F$ for different $\eta$ (Hexagonal BZ sampling)')
plt.axvline(0.0, ls='--', label=f'Calibration ($T_c={Tc_initial}\\,$K)')
plt.xlim(left=x_min)
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()
