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
mu_guess = 0.10  # eV
E_K_raw = eps_Mo(K[0], K[1], mu_guess)
E_F = E_K_raw + 0.15   # set valley near -0.15 eV

def E(kx, ky):
    return eps_Mo(kx, ky, mu_guess) - E_F

# Calibrate alpha (splitting at K ≈ 3 meV)
fK = f_k(K[0], K[1])
coreK = abs(core_g(K[0], K[1]))
alpha = (3e-3) / (2.0 * coreK)

# Find k_F on right side where E crosses 0
mask_R = s >= 0
Ei = E(kx[mask_R], ky[mask_R])
kF_idx = np.argmin(np.abs(Ei))
kF_kx, kF_ky = kx[mask_R][kF_idx], ky[mask_R][kF_idx]

# Calibrate beta (splitting at k_F ≈ 13 meV)
betas = np.linspace(0.5, 200.0, 20001)
spls  = np.array([2*alpha*abs(gzz(kF_kx, kF_ky, fK, b)) for b in betas])
beta  = float(betas[np.argmin(np.abs(spls - 13e-3))])

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
def calculate_chi_formula_vec(
    T, H,
    kx_vals, ky_vals,
    eps_func, gz_func,
    mu, alpha,
    mu_B_eVT=mu_B_eVT, k_B_eV=k_B_eV,
    Nw=600, show_progress=False,
    eps_grid=None, gz_grid=None,
):
    # Nw = number of Matsubara frequencies, show_progress = whether to show progress bar

    wn  = (2*np.arange(Nw)+1) * np.pi * k_B_eV * T      # Matsubara frequencies ω_n = (2n+1)πT
    chi = 0.0 + 0.0j                                    # initialize χ_sc^0

    # Evaluate or reuse k-grids
    kxg, kyg = np.meshgrid(kx_vals, ky_vals, indexing='xy')
    epsg = eps_func(kxg, kyg, mu) if eps_grid is None else eps_grid
    gzg  = gz_func(kxg, kyg)      if gz_grid  is None else gz_grid

    muBH = mu_B_eVT * H
    Nk_tot = epsg.size

    # tqdm progress bars
    w_iter = tqdm(wn, desc="χ", leave=False) if show_progress else wn

    # double frequency loop
    for w in w_iter:
        iw = 1j*w

        # H(k)
        A = iw - (epsg + alpha*gzg)
        D = iw - (epsg - alpha*gzg)
        detk = A*D - (muBH**2)

        # H(-k)
        Am = -iw - (epsg - alpha*gzg)
        Dm = -iw - (epsg + alpha*gzg)
        detm = Am*Dm - (muBH**2)

        integrand = (D*Am - (muBH**2)) / (detk*detm)
        chi += np.sum(integrand)

    # Normalisation and Return
    chi = (2.0 * T / Nk_tot) * chi
    return float(np.real(chi))


# ------------------------ Find Hc2 ----------------------------
# Calibrate beta for target splitting at k_F
def calibrate_beta_for_target_split_at_kF(target_meV, alpha, kF_kx, kF_ky, fK):
    betas = np.linspace(0.5, 200.0, 20001)  # extensive, dense arrays of beta values
    spls  = np.array([2*alpha*abs(gzz(kF_kx, kF_ky, fK, b)) for b in betas])  # eV
    target = target_meV * 1e-3

    # Return beta that gives splitting closest to target
    return float(betas[np.argmin(np.abs(spls - target))])

# Determine V from Tc 
def determine_V(Tc, kx_vals, ky_vals, eps_func, gz_func, mu, alpha, Nw,
                eps_grid=None, gz_grid=None):
    chi_Tc_0 = calculate_chi_formula_vec(
        Tc, 0.0, kx_vals, ky_vals, eps_func, gz_func, mu, alpha,
        mu_B_eVT=mu_B_eVT, k_B_eV=k_B_eV, Nw=Nw, show_progress=False,
        eps_grid=eps_grid, gz_grid=gz_grid
    )
    return 1.0 / chi_Tc_0

# Find Hc2 at given T using bisection method
def find_Hc2_at_T(T, V, kx_vals, ky_vals, eps_func, gz_func, mu, alpha,
                  Nw, Hmax=150.0, max_iter=32, eps_grid=None, gz_grid=None):
    chi_T0 = V * calculate_chi_formula_vec(
        T, 0.0, kx_vals, ky_vals, eps_func, gz_func, mu, alpha,
        mu_B_eVT=mu_B_eVT, k_B_eV=k_B_eV, Nw=Nw, show_progress=False,
        eps_grid=eps_grid, gz_grid=gz_grid
    )
    if chi_T0 < 1.0:
        return 0.0

    lo, hi = 0.0, Hmax
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        val = V * calculate_chi_formula_vec(
            T, mid, kx_vals, ky_vals, eps_func, gz_func, mu, alpha,
            mu_B_eVT=mu_B_eVT, k_B_eV=k_B_eV, Nw=Nw, show_progress=False,
            eps_grid=eps_grid, gz_grid=gz_grid
        )
        if val >= 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

# ------------------------ Run: two curves -------------------------
# Fix alpha = 3, Do beta recalibration for each curve to make Δ(k_F)=13/3 meV
Tc = 6.5
Nw = 1200 # how many Songyuan frequencies considered
Nw = 600  # override to speed up
Ts = np.linspace(1.0, 7.0, 22)  

targets = [(13.0, 'red',   r'$\Delta_Z(k_F)=13\,\mathrm{meV}$'),
           ( 3.0, 'black', r'$\Delta_Z(k_F)=3\,\mathrm{meV}$')]

plt.figure(figsize=(6,4.6))

# Loop over target splittings
for target_meV, col, lab in targets:
    beta_target = calibrate_beta_for_target_split_at_kF(target_meV, alpha, kF_kx, kF_ky, fK)
    gz_func = lambda kx_, ky_, fK=fK, b=beta_target: gzz(kx_, ky_, fK, b)

    gz_grid = gz_func(kx_grid, ky_grid)
    V = determine_V(Tc, kx_vals, ky_vals, eps_Mo, gz_func, mu, alpha, Nw,
                    eps_grid=eps_grid_global, gz_grid=gz_grid)

    Hc2 = []
    for T in tqdm(Ts, desc=f'Hc2: {lab}'):
        Hc2.append(find_Hc2_at_T(
            T, V, kx_vals, ky_vals, eps_Mo, gz_func, mu, alpha, Nw,
            Hmax=150.0, max_iter=32, eps_grid=eps_grid_global, gz_grid=gz_grid
        ))
    Hc2 = np.array(Hc2)

    plt.plot(Ts, Hc2, lw=2.0, color=col, label=lab)

plt.xlabel('T (K)')
plt.ylabel(r'$\mu_0 H_{c2}$ (T)')
plt.xlim(1.0, 8.0)
plt.ylim(0, 120)
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.title('Numerical Calculation of Upper Critical Field Hc2')
plt.show()
# ------------------------------------------------------------------
