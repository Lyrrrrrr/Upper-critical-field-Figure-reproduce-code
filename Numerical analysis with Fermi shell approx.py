import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm
from scipy.constants import k as k_B, e

# ------------------------ All parameters ------------------------
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

# ------------------------ TB & SOI process ------------------------
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
mu_guess = 0.10
E_K_raw  = eps_Mo(K[0], K[1], mu_guess)
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
def k_patch(center, rad=0.20, Nk=91):
    qx = np.linspace(-rad, rad, Nk)
    qy = np.linspace(-rad, rad, Nk)
    Qx, Qy = np.meshgrid(qx, qy, indexing='xy')
    mask = (Qx**2 + Qy**2 <= rad**2)
    kx = center[0] + Qx[mask]
    ky = center[1] + Qy[mask]
    return kx, ky

# Build Fermi surface shell samples around K and -K
def build_FS_shell(mu_guess, EF, rad=0.20, Nk=91, n_keep=1200):
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
def chi_singlet_vectorized(T, H, alpha, gz_func, kK, kKm, mu_guess, EF, Nw):
    kxK,  kyK  = kK
    kxKm, kyKm = kKm
    Nk = kxK.size
    xiK   = eps_Mo(kxK,  kyK,  mu_guess) - EF
    lamK  = alpha * gz_func(kxK,  kyK)
    xiKm  = eps_Mo(kxKm, kyKm, mu_guess) - EF
    lamKm = alpha * gz_func(kxKm, kyKm)

    wn = (2*np.arange(Nw)+1) * np.pi * k_B_eV * T
    A   = 1j*wn[:,None] - xiK[None,:]
    Bx  = -mu_B_eVT * H
    Bz  = -lamK[None,:]

    denom  = (A*A - Bx*Bx - Bz*Bz)
    Guu    = (A - Bz)/denom
    Gdd    = (A + Bz)/denom
    Gud    = (-Bx)/denom
    Gdu    = (-Bx)/denom

    A2   = -1j*wn[:,None] - xiKm[None,:]
    Bz2  = -lamKm[None,:]
    denom2 = (A2*A2 - Bx*Bx - Bz2*Bz2)
    Guu_m  = (A2 - Bz2)/denom2
    Gdd_m  = (A2 + Bz2)/denom2
    Gud_m  = (-Bx)/denom2
    Gdu_m  = (-Bx)/denom2

    term = Guu*Gdd_m - Gdu*Gud_m
    chi  = (k_B_eV*T) * np.real(term.sum()) / Nk
    return float(chi)

# ------------------------ Find Hc2 ----------------------------
# Determine V from Tc 
def determine_V(Tc, alpha, gz_func, kK, kKm, mu_guess, EF, Nw):
    chi_Tc0 = chi_singlet_vectorized(Tc, 0.0, alpha, gz_func, kK, kKm, mu_guess, EF, Nw)
    return 1.0 / chi_Tc0

# Find Hc2 at given T using bisection method searching in [0, Hmax]
def find_Hc2(T, V, alpha, gz_func, kK, kKm, mu_guess, EF, Nw, Hmax=100.0):
    if V*chi_singlet_vectorized(T, 0.0, alpha, gz_func, kK, kKm, mu_guess, EF, Nw) < 1.0:
        return 0.0
    lo, hi = 0.0, Hmax
    for _ in range(24):
        mid = 0.5*(lo+hi)
        v   = V*chi_singlet_vectorized(T, mid, alpha, gz_func, kK, kKm, mu_guess, EF, Nw)
        if v >= 1.0: lo = mid
        else:        hi = mid
    return 0.5*(lo+hi)

# ------------------------ two curves ----------------------------
# Fix alpha = 3, Do beta recalibration for each curve to make Δ(k_F)=13/3 meV
kK, kKm = build_FS_shell(mu_guess, E_F, n_keep=1200)
print("FS sample per valley:", kK[0].size)

Tc = 6.5    # critical temperature in K
Ts = np.linspace(0.1, 7.0, 100)
Nw = 800    # how many Matsubara frequencies considered
targets = [13.0, 3.0]
labels  = [r'$\Delta_Z(k_F)=13\,\mathrm{meV}$', r'$\Delta_Z(k_F)=3\,\mathrm{meV}$']

plt.figure(figsize=(6,4.6))
# Loop over target splittings
for targ, lab in zip(targets, labels):
    beta   = beta_from_target(targ)
    gz_fun = lambda kx,ky, fK=fK, b=beta: gzz(kx,ky,fK,b)

    # use Tc at H=0 to calibrate V
    V = determine_V(Tc, alpha, gz_fun, kK, kKm, mu_guess, E_F, Nw)

    # progress bar over temperatures
    Hs = [find_Hc2(T, V, alpha, gz_fun, kK, kKm, mu_guess, E_F,
                   Nw, Hmax=100.0)
          for T in tqdm(Ts, desc=f'Hc2 {lab}')]
    
    # plot with different colors
    if targ == 13.0:
        plt.plot(Ts, Hs, lw=2.2, color='red', label=lab)
    else:
        plt.plot(Ts, Hs, lw=2.2, color='black', label=lab)

plt.xlabel('T (K)')
plt.ylabel(r'$\mu_0 H_{c2}$ (T)')
plt.xlim(0, 7.0); plt.ylim(0, 80)
plt.grid(True, ls=':'); plt.legend()
plt.tight_layout()
plt.title('Temperature dependence of upper critical field')
plt.show()
