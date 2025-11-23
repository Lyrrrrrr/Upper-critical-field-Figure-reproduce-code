#------------------------------------ import ------------------------------------
import numpy as np
from scipy.constants import k as k_B, e


# ------------------------------------ parameters ------------------------------------
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

# ------------------------------------ Core code ------------------------------------
# energy dispersion, corresponding to H_{kin} term
def eps_Mo(kx, ky, mu):
    return (2*t1*(np.cos(ky*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(0.5*ky*a))
            +2*t2*(np.cos(np.sqrt(3)*kx*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(1.5*ky*a))
            +2*t3*(np.cos(2*ky*a) + 2*np.cos(np.sqrt(3)*kx*a)*np.cos(ky*a))-mu)

# g_z(k) for Zeeman-type SOI
def core_g(kx, ky):
    # sin(ky a) - 2 cos(√3/2 kx a) sin(ky a/2)
    return np.sin(ky*a) - 2*np.cos(np.sqrt(3)/2*kx*a)*np.sin(0.5*ky*a)

def f_k(kx, ky):
    # f(k) = | core_g(k) |
    return np.abs(core_g(kx, ky))

def F_k(kx, ky, fK, beta):
    # F(k) = beta * tanh[ f(K) - f(k) ] - 1
    return beta * np.tanh(fK - f_k(kx, ky)) - 1.0

def gzz(kx, ky, fK, beta):
    return F_k(kx, ky, fK, beta) * core_g(kx, ky)

# g_R(k) for Rashba-type SOI
def gR_core(kx, ky):
    gx_core = -np.sin(ky*a) - np.cos(np.sqrt(3)/2*kx*a)*np.sin(0.5*ky*a)
    gy_core = np.sqrt(3) * np.sin(np.sqrt(3)/2*kx*a)*np.cos(0.5*ky*a)
    return gx_core, gy_core

def gR_vec(kx, ky, fK, beta):
    Fk = F_k(kx, ky, fK, beta)
    gx_core, gy_core = gR_core(kx, ky)
    return Fk * gx_core, Fk * gy_core

def gR_mag(kx, ky, fK, beta):
    gx, gy = gR_vec(kx, ky, fK, beta)
    return np.sqrt(gx*gx + gy*gy)

# High symmetry points
Gamma = np.array([0.0, 0.0])
K = np.array([0.0, 4*np.pi/(3*a)])
M = np.array([np.pi/(np.sqrt(3)*a), np.pi/a])

# Path, Γ←K (neg s) and K→M (pos s), near K
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
Nk = 121
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
    kxK,  kyK  = kK

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


