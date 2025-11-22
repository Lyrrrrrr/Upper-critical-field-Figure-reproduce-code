import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm
from scipy.constants import k as k_B, e
import csv


# ============================================================================
# DOS SAMPLING PARAMETERS (USER CONFIGURABLE)
# ============================================================================
# 用户可以在这里自由配置DOS的采样范围和采样点数
E_min_dos = -0.15 # DOS计算的最小能量 (eV)
E_max_dos = 0.5   # DOS计算的最大能量 (eV)
num_dos_points = 400 # DOS采样点数（直接使用linspace生成的点数）
sigma_dos = 0.003# 高斯展宽参数 (eV)，用于平滑DOS曲线
# ============================================================================


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
def path_around_K(smax=0.2, N=1001):
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

# ------------------------ Fermi shell ------------------------
# K and -K patches in k-space
def k_patch(center, rad, Nk):
    qx = np.linspace(-rad, rad, Nk)
    qy = np.linspace(-rad, rad, Nk)
    Qx, Qy = np.meshgrid(qx, qy, indexing='xy')
    mask = (Qx**2 + Qy**2 <= rad**2)
    kx = center[0] + Qx[mask]
    ky = center[1] + Qy[mask]
    return kx, ky

# Build Fermi surface shell samples around K and -K
def build_FS_shell(mu_guess, EF, rad, Nk, n_keep):
    kxK,  kyK  = k_patch(K,   rad, Nk)
    kxKm, kyKm = k_patch(-K,  rad, Nk)
    xiK  = eps_Mo(kxK,  kyK,  mu_guess) - EF
    xiKm = eps_Mo(kxKm, kyKm, mu_guess) - EF
    idxK  = np.argsort(np.abs(xiK))[:n_keep]
    idxKm = np.argsort(np.abs(xiKm))[:n_keep]
    return (kxK[idxK], kyK[idxK]), (kxKm[idxKm], kyKm[idxKm])

# ------------------------ Susceptibility calculation ------------------------
# 2x2 identity and Pauli matrices
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1],[1, 0]], dtype=complex)
sz = np.array([[1, 0],[0,-1]], dtype=complex)

# k grid setup
Nk = 1000
kx_vals = np.linspace(-np.pi/a, np.pi/a, Nk)
ky_vals = np.linspace(-np.pi/a, np.pi/a, Nk)
# Precompute k-grid and dispersion once
kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals, indexing='xy')
eps_grid_global = eps_Mo(kx_grid, ky_grid, mu)

# Encapsulated as g_z(kx, ky) without additional parameters
gz_func = lambda kx, ky: gzz(kx, ky, fK, beta)

# calculate χ using the formula with vectorization over k-grid
def chi_singlet_vectorized(T, H, alpha_Z, alpha_R, gz_func, gR_func, kK, mu_guess, EF, Nw):

    # K-point k-grid
    if isinstance(kK, (tuple, list)) and len(kK) == 2 and isinstance(kK[0], (tuple, list)) and len(kK[0]) == 2:
        (kx1, ky1), (kx2, ky2) = kK
        kxK = np.concatenate([np.asarray(kx1, dtype=float), np.asarray(kx2, dtype=float)])
        kyK = np.concatenate([np.asarray(ky1, dtype=float), np.asarray(ky2, dtype=float)])
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


print("Starting calibration...")

# a) Calibrate E_F
mu_guess = 0.0
E_K_raw = eps_Mo(K[0], K[1], mu_guess)
E_F = E_K_raw + 0.15
E_kin = lambda kx, ky: eps_Mo(kx, ky, mu_guess) - E_F
print(f"Set Fermi Level E_F = {E_F:.4f} eV")

# b) Calibrate alpha_Z (using 3 meV splitting at K)
fK = f_k(K[0], K[1])
coreK = abs(core_g(K[0], K[1]))
alpha_Z = (3e-3) / (2.0 * max(coreK, 1e-18))
print(f"Calibrated alpha_Z = {alpha_Z:.6f}")

# c) Calibrate beta (using 13 meV splitting at k_F)
try:
    kx_path, ky_path, s_path = path_around_K(0.2, 301)
    mask_R = s_path >= 0
    Ei_path = E_kin(kx_path[mask_R], ky_path[mask_R])
    kF_idx = np.argmin(np.abs(Ei_path))
    kF_kx, kF_ky = kx_path[mask_R][kF_idx], ky_path[mask_R][kF_idx]
    
    betas_calib = np.linspace(0.5, 200.0, 2001)
    spls_calib = np.array([2*alpha_Z*abs(gzz(kF_kx, kF_ky, fK, b)) for b in betas_calib])
    target_meV_kf = 13.0
    beta = float(betas_calib[np.argmin(np.abs(spls_calib - target_meV_kf*1e-3))])
    print(f"Calibrated beta = {beta:.4f}")

except NameError:
    print("Warning: path_around_K function not found. Skipping beta calibration.")
    print("Using beta provided by user (which might be 0).")
    pass


# d) Calibrate alpha_R (Rashba strength)
eta = 0.02
alpha_R = eta * alpha_Z
print(f"Calibrated alpha_R = {alpha_R:.6f} (using eta={eta})")
print("--- Calibration Complete ---")


# ============================================================================
# HEXAGONAL BRILLOUIN ZONE IMPLEMENTATION
# ============================================================================
# 六边形第一布里渊区的精确定义

def in_hexagonal_BZ(kx, ky, a):
    """
    检查k点是否在六边形第一布里渊区内
    
    六边形BZ的6个顶点是K点（和K'点交替排列）：
    - (0, ±K_y) - 顶部和底部顶点
    - (±K_y*√3/2, ±K_y/2) - 其他4个顶点
    
    边界由6条直线定义，可以通过对称性简化为2个条件。
    
    Parameters:
    -----------
    kx, ky : array-like
        k点坐标 (Å⁻¹)
    a : float
        晶格常数 (Å)
    
    Returns:
    --------
    mask : boolean array
        True表示该点在第一BZ内
    """
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    
    # K点的y坐标（六边形顶点到中心的距离）
    K_y = 4*np.pi/(3*a)
    
    # 使用对称性，只需检查绝对值
    ky_abs = np.abs(ky)
    kx_abs = np.abs(kx)
    
    # 条件1: 水平边界（不需要，因为斜边条件已经包含）
    # |ky| ≤ K_y 会被条件2自动满足
    
    # 条件2: 斜边界
    # 从顶点 (0, K_y) 到顶点 (K_y*√3/2, K_y/2) 的直线方程：
    # 斜率 = (K_y/2 - K_y) / (K_y*√3/2 - 0) = -1/√3
    # 直线方程: ky = K_y - kx/√3
    # 内部条件: |ky| ≤ K_y - |kx|/√3
    slope = 1.0 / np.sqrt(3)
    cond_slanted = ky_abs <= K_y - slope * kx_abs
    
    # 条件3: kx边界（六边形的左右两条垂直边）
    # 从 (K_y*√3/2, K_y/2) 到 (K_y*√3/2, -K_y/2) 是垂直边
    # |kx| ≤ K_y*√3/2
    kx_max = K_y * np.sqrt(3) / 2
    cond_vertical = kx_abs <= kx_max
    
    # 两个条件都必须满足
    return cond_slanted & cond_vertical


def generate_hexagonal_BZ_grid(a, Nk):
    """
    生成严格在六边形第一布里渊区内的k点网格
    
    Parameters:
    -----------
    a : float
        晶格常数 (Å)
    Nk : int
        每个方向的采样密度（生成Nk×Nk的候选网格，然后过滤）
    
    Returns:
    --------
    kx_1BZ, ky_1BZ : arrays
        在第一BZ内的k点坐标
    mask : 2D boolean array
        六边形mask
    N_k_in_BZ : int
        在BZ内的k点总数
    """
    # 高对称点坐标
    K_y = 4*np.pi/(3*a)      # K点的y坐标（六边形顶点到中心的距离）
    M_x = np.pi/(np.sqrt(3)*a)  # M点的x坐标
    M_y = np.pi/a            # M点的y坐标
    
    # 六边形的顶点（以K点为顶点）：
    # 顶部和底部顶点: (0, ±K_y)
    # 其他4个顶点通过60°旋转得到: (±K_y*√3/2, ±K_y/2)
    # 所以kx的范围是 [-K_y*√3/2, K_y*√3/2]，不是 [-M_x, M_x]！
    kx_max_hex = K_y * np.sqrt(3) / 2  # 六边形在kx方向的最大范围
    ky_max_hex = K_y                    # 六边形在ky方向的最大范围
    
    # 生成覆盖整个六边形的矩形网格
    kx_vals = np.linspace(-kx_max_hex * 1.01, kx_max_hex * 1.01, Nk)
    ky_vals = np.linspace(-ky_max_hex * 1.01, ky_max_hex * 1.01, Nk)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals, indexing='xy')
    
    # 应用六边形mask
    mask = in_hexagonal_BZ(kx_grid, ky_grid, a)
    
    # 提取在BZ内的k点
    kx_1BZ = kx_grid[mask]
    ky_1BZ = ky_grid[mask]
    N_k_in_BZ = mask.sum()
    
    return kx_1BZ, ky_1BZ, mask, N_k_in_BZ


# ============================================================================
# --- 2. Generate 2D k-mesh (HEXAGONAL BZ) ---
# ============================================================================
print("\n" + "="*70)
print("HEXAGONAL BRILLOUIN ZONE SAMPLING")
print("="*70)

Nk = 1000  # 采样密度

# 生成六边形BZ内的k点
kx_1BZ, ky_1BZ, hex_mask, N_k_in_BZ = generate_hexagonal_BZ_grid(a, Nk)

# 计算理论面积和实际覆盖
A_1BZ_theory = 8 * np.pi**2 / (np.sqrt(3) * a**2)
K_y = 4*np.pi/(3*a)
kx_max_hex = K_y * np.sqrt(3) / 2  # 六边形kx范围
ky_max_hex = K_y                    # 六边形ky范围
# k空间网格间距
dk_x = (2 * kx_max_hex * 1.01) / (Nk - 1)
dk_y = (2 * ky_max_hex * 1.01) / (Nk - 1)
dk2 = dk_x * dk_y  # 每个k点对应的面积元
A_sampled = N_k_in_BZ * dk2

print(f"Generated hexagonal BZ grid:")
print(f"  Sampling density: {Nk} × {Nk} candidate grid")
print(f"  k-points inside 1st BZ: {N_k_in_BZ}")
print(f"  Theoretical 1st BZ area: {A_1BZ_theory:.4f} Å⁻²")
print(f"  Sampled area (approx): {A_sampled:.4f} Å⁻²")
print(f"  Coverage ratio: {A_sampled/A_1BZ_theory:.4f}")

# 存储用于后续计算的N_k_total
N_k_total = float(N_k_in_BZ)


# --- 3. Calculate All Energies on the Hexagonal Grid ---
print("\nCalculating energies for all k-points in hexagonal BZ...")
E_k_grid = E_kin(kx_1BZ, ky_1BZ)
gz_grid = gzz(kx_1BZ, ky_1BZ, fK, beta)
gr_grid = gR_mag(kx_1BZ, ky_1BZ, fK, beta)
Gabs_grid = np.sqrt((alpha_Z * gz_grid)**2 + (alpha_R * gr_grid)**2)
E_plus_all = E_k_grid + Gabs_grid
E_minus_all = E_k_grid - Gabs_grid
all_energies = np.concatenate([E_plus_all.flatten(), E_minus_all.flatten()])
print("...Energy calculation complete.")
print(f"  Total energy states: {len(all_energies)} (2 bands × {N_k_in_BZ} k-points)")


# --- 4. Sort All Energies ---
print("Sorting all energies...")
all_energies_sorted = np.sort(all_energies)
print("...Sorting complete.")


# --- 5. Define N(E) and g(E) (DOS) Functions ---

def N_per_cell(E):
    """
    Calculates N(E), the integrated number of states per unit cell
    with energy <= E.
    """
    count = np.searchsorted(all_energies_sorted, E, side='right')
    return count / N_k_total

def DOS_per_cell_gaussian(E_points, sigma=0.003):
    """
    Calculates g(E), the density of states per unit cell using Gaussian broadening.
    
    Parameters:
    -----------
    E_points : array-like
        Energy values where DOS should be calculated (directly from linspace)
    sigma : float
        Gaussian broadening parameter (eV), controls the smoothness of DOS
    
    Returns:
    --------
    E_points : array
        The input energy values (mu values)
    g_E : array
        DOS values at each energy point
    """
    E_points = np.asarray(E_points)
    g_E = np.zeros_like(E_points)
    
    # For each requested energy point, sum Gaussian contributions from all states
    # g(E) = sum_i (1/sqrt(2*pi*sigma^2)) * exp(-(E - E_i)^2 / (2*sigma^2))
    # Normalized per unit cell and per eV
    
    prefactor = 1.0 / (N_k_total * np.sqrt(2 * np.pi * sigma**2))
    
    print(f"Calculating DOS at {len(E_points)} energy points using Gaussian broadening...")
    for i, E in enumerate(E_points):
        # Vectorized calculation of Gaussian contributions
        gaussian_weights = np.exp(-0.5 * ((all_energies - E) / sigma)**2)
        g_E[i] = prefactor * np.sum(gaussian_weights)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(E_points)} points calculated")
    
    print("DOS calculation complete.")
    return E_points, g_E

print("Functions N_per_cell(E) and DOS_per_cell_gaussian(E_points, sigma) are now defined.")


# --- 6. Example Usage and Plotting ---
print(f"\nPreparing to calculate DOS over range [{E_min_dos:.3f}, {E_max_dos:.3f}] eV")
print(f"Number of sampling points: {num_dos_points}")
print(f"Gaussian broadening: σ = {sigma_dos:.6f} eV")

# a) Create energy grid using linspace (这些点会精确包含边界)
E_range = np.linspace(E_min_dos, E_max_dos, num_dos_points)
print(f"Energy grid created: first point = {E_range[0]:.6f} eV, last point = {E_range[-1]:.6f} eV")

# b) Calculate N(E) over this range
print("\nCalculating N(E)...")
N_E_vals = np.array([N_per_cell(E) for E in E_range])
print("N(E) calculation complete.")

# c) Calculate g(E) (DOS) using Gaussian broadening at the same energy points
E_dos, g_E_dos = DOS_per_cell_gaussian(E_range, sigma=sigma_dos)

# ============================================================================
# CSV EXPORT FUNCTIONALITY
# ============================================================================
print("\n--- Exporting data to CSV files ---")

# Export N(E) data
n_e_filename = 'N_vs_mu_hexBZ.csv'
with open(n_e_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['# Hexagonal BZ sampling'])
    writer.writerow([f'# k-points in 1st BZ: {N_k_in_BZ}'])
    writer.writerow(['mu (E-E_F) [eV]', 'N(E) [states per unit cell]'])
    for e_val, n_val in zip(E_range, N_E_vals):
        writer.writerow([f'{e_val:.6f}', f'{n_val:.8f}'])
print(f"Exported N(E) data to {n_e_filename}")
print(f"  N(E) range: [{E_range[0]:.6f}, {E_range[-1]:.6f}] eV ({len(E_range)} points)")

# Export DOS data (already at the correct energy points)
dos_filename = 'DOS_vs_mu_hexBZ.csv'
with open(dos_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['# Hexagonal BZ sampling'])
    writer.writerow([f'# k-points in 1st BZ: {N_k_in_BZ}'])
    writer.writerow([f'# Gaussian broadening sigma: {sigma_dos} eV'])
    writer.writerow(['mu (E-E_F) [eV]', 'DOS(E) [states/eV per unit cell]'])
    for e_val, dos_val in zip(E_dos, g_E_dos):
        writer.writerow([f'{e_val:.6f}', f'{dos_val:.8f}'])
print(f"Exported DOS data to {dos_filename}")
print(f"  DOS range: [{E_dos[0]:.6f}, {E_dos[-1]:.6f}] eV ({len(E_dos)} points)")

print("CSV export complete!\n")
# ============================================================================

# d) Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Hexagonal BZ visualization
ax_bz = axes[0, 0]
# 高对称点坐标（用于绘图）
K_y_plot = 4*np.pi/(3*a)
M_x_plot = np.pi/(np.sqrt(3)*a)
M_y_plot = np.pi/a

# 正确的正六边形顶点（K点是顶点，不是M点！）
# 六边形顶点到中心的距离 = K_y
# 6个顶点通过60°旋转得到
hex_vertices = np.array([
    [0, K_y_plot],                                    # 顶部 K
    [-K_y_plot*np.sqrt(3)/2, K_y_plot/2],            # 左上
    [-K_y_plot*np.sqrt(3)/2, -K_y_plot/2],           # 左下
    [0, -K_y_plot],                                   # 底部 K'
    [K_y_plot*np.sqrt(3)/2, -K_y_plot/2],            # 右下
    [K_y_plot*np.sqrt(3)/2, K_y_plot/2],             # 右上
    [0, K_y_plot]  # 闭合
])
ax_bz.plot(hex_vertices[:, 0], hex_vertices[:, 1], 'b-', lw=2, label='1st BZ boundary')
ax_bz.scatter(kx_1BZ[::50], ky_1BZ[::50], s=0.5, alpha=0.3, c='red', label='Sampled k-points')
ax_bz.plot(0, 0, 'ko', markersize=8)
ax_bz.annotate('Γ', (0.02, 0.02), fontsize=12)
ax_bz.plot(0, K_y_plot, 'go', markersize=6)
ax_bz.annotate('K', (0.02, K_y_plot+0.03), fontsize=10, color='green')
ax_bz.plot(M_x_plot, M_y_plot, 'mo', markersize=6)
ax_bz.annotate('M', (M_x_plot+0.02, M_y_plot), fontsize=10, color='purple')
ax_bz.set_xlabel('kx (Å⁻¹)')
ax_bz.set_ylabel('ky (Å⁻¹)')
ax_bz.set_title(f'Hexagonal 1st Brillouin Zone\n({N_k_in_BZ} k-points sampled)')
ax_bz.set_aspect('equal')
ax_bz.legend(loc='upper right', fontsize=8)
ax_bz.grid(True, alpha=0.3)

# Plot 2: Band structure along high-symmetry path
ax_band = axes[0, 1]
kx_path, ky_path, s_path = path_around_K(0.5, 501)
E_path = E_kin(kx_path, ky_path)
gz_path = gzz(kx_path, ky_path, fK, beta)
gr_path = gR_mag(kx_path, ky_path, fK, beta)
Gabs_path = np.sqrt((alpha_Z * gz_path)**2 + (alpha_R * gr_path)**2)
E_plus_path = E_path + Gabs_path
E_minus_path = E_path - Gabs_path
ax_band.plot(s_path, E_plus_path*1000, 'r-', lw=1.5, label='E₊')
ax_band.plot(s_path, E_minus_path*1000, 'b-', lw=1.5, label='E₋')
ax_band.axhline(0, color='k', ls='--', lw=0.5)
ax_band.axvline(0, color='gray', ls=':', lw=0.5)
ax_band.set_xlabel('k along Γ-K-M (Å⁻¹)')
ax_band.set_ylabel('E - E_F (meV)')
ax_band.set_title('Band Structure near K point')
ax_band.legend()
ax_band.grid(True, ls=':')

# Plot 3: N(E)
ax_N = axes[1, 0]
ax_N.plot(E_range, N_E_vals, lw=2, color='navy')
ax_N.set_xlabel(r'$E - E_F$ (eV)')
ax_N.set_ylabel(r'$N(E)$ (States per Unit Cell)')
ax_N.set_title('Integrated Density of States $N(E)$')
ax_N.grid(True, ls=':')
ax_N.axhline(0, color='k', lw=0.5)
ax_N.axvline(0, color='r', ls='--', lw=1, label='$E_F = 0$')
ax_N.legend()
ax_N.set_xlim(E_min_dos, E_max_dos)

# Plot 4: g(E)
ax_g = axes[1, 1]
ax_g.plot(E_dos, g_E_dos, lw=2, color='darkorange')
ax_g.set_xlabel(r'$E - E_F$ (eV) = $\mu$ (Chemical Potential)')
ax_g.set_ylabel(r'$g(E)$ (States / eV / Unit Cell)')
ax_g.set_title(f'Density of States $g(E)$ (Hexagonal BZ, σ={sigma_dos:.4f} eV)')
ax_g.grid(True, ls=':')
ax_g.set_ylim(bottom=0)
ax_g.axvline(0, color='r', ls='--', lw=1, label='$E_F = 0$')
ax_g.legend()
ax_g.set_xlim(E_min_dos, E_max_dos)

plt.tight_layout()
plt.savefig('DOS_hexagonal_BZ.png', dpi=150, bbox_inches='tight')
plt.show()

# Example: Find N(0), the number of states below E_F
N_at_EF = N_per_cell(0.0)
print(f"\nN(E=0) = {N_at_EF:.4f} (states per cell below E_F)")

print("\n" + "="*70)
print("SUMMARY OF HEXAGONAL BZ DOS CALCULATION:")
print("="*70)
print(f"Brillouin Zone: HEXAGONAL (exact 1st BZ)")
print(f"  K-points in BZ: {N_k_in_BZ}")
print(f"  Theoretical BZ area: {A_1BZ_theory:.4f} Å⁻²")
print(f"  Sampled area: {A_sampled:.4f} Å⁻²")
print(f"  Coverage ratio: {A_sampled/A_1BZ_theory:.4f}")
print("-"*70)
print(f"DOS Energy Range: [{E_min_dos:.6f}, {E_max_dos:.6f}] eV")
print(f"Number of DOS Points: {num_dos_points}")
print(f"Energy Resolution: {(E_max_dos - E_min_dos)/(num_dos_points-1):.6f} eV per point")
print(f"Gaussian Broadening: σ = {sigma_dos:.6f} eV")
print(f"First μ point in CSV: {E_range[0]:.6f} eV")
print(f"Last μ point in CSV: {E_range[-1]:.6f} eV")
print("="*70)
print("\nOutput files:")
print(f"  - {n_e_filename} : N(E) data")
print(f"  - {dos_filename} : DOS data")
print(f"  - DOS_hexagonal_BZ.png : Visualization")
print("="*70)
