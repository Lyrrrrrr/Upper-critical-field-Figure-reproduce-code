import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B, e

# ===== TB parameters =====
a = 3.18
t1 = 146e-3
t2 = -0.40 * t1
t3 = 0.25 * t1
mu = 0.0

# ===== Physical =====
mu_B_eVT = 5.7883818012e-5
k_B_eV = k_B / e

# ===== High symmetry points =====
Gamma = np.array([0.0, 0.0])
K = np.array([0.0, 4*np.pi/(3*a)])
M = np.array([np.pi/(np.sqrt(3)*a), np.pi/a])

# ===== Energy dispersion =====
def eps_Mo(kx, ky, mu):
    return (2*t1*(np.cos(ky*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(0.5*ky*a))
            + 2*t2*(np.cos(np.sqrt(3)*kx*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(1.5*ky*a))
            + 2*t3*(np.cos(2*ky*a) + 2*np.cos(np.sqrt(3)*kx*a)*np.cos(ky*a)) - mu)

# ===== Path around K =====
def path_around_K(smax=0.2, N=801):
    v_GK = K - Gamma
    v_KM = M - K
    L_GK = np.linalg.norm(v_GK)
    L_KM = np.linalg.norm(v_KM)
    u1 = np.linspace(1 - smax/L_GK, 1.0, N)
    u2 = np.linspace(0.0, smax/L_KM, N)
    k_GK = Gamma + u1[:,None]*v_GK
    k_KM = K     + u2[:,None]*v_KM
    s_GK = -(1.0 - u1) * L_GK
    s_KM =  (u2      ) * L_KM
    kx = np.concatenate([k_GK[:,0], k_KM[:,0]])
    ky = np.concatenate([k_GK[:,1], k_KM[:,1]])
    s  = np.concatenate([s_GK,       s_KM])
    return kx, ky, s

# ===== g_z(k): out-of-plane Zeeman-type =====
def core_g(kx, ky):
    return np.sin(ky*a) - 2*np.cos(np.sqrt(3)/2*kx*a)*np.sin(0.5*ky*a)

def f_k(kx, ky):
    return np.abs(core_g(kx, ky))

def F_k(kx, ky, fK, beta):
    return beta * np.tanh(fK - f_k(kx, ky)) - 1.0

def gzz(kx, ky, fK, beta):
    return F_k(kx, ky, fK, beta) * core_g(kx, ky)

# ===== g_R(k): in-plane Rashba-type =====
def gR_vec(kx, ky):
    gx = np.sin(0.5*ky*a) * np.cos(np.sqrt(3)*0.5*kx*a)
    gy = -np.sin(np.sqrt(3)*0.5*kx*a) * np.cos(0.5*ky*a)
    return gx, gy

def gR_mag(kx, ky):
    gx, gy = gR_vec(kx, ky)
    return np.sqrt(gx*gx + gy*gy)

# ===== Path & Fermi alignment =====
kx, ky, s = path_around_K(smax=0.2, N=801)
mu_guess = 0.10
E_K_raw = eps_Mo(K[0], K[1], mu_guess)
E_F = E_K_raw + 0.15
def Ebare(kx, ky):
    return eps_Mo(kx, ky, mu_guess) - E_F

# ===== α_Z calibration: ΔE(K)=3 meV =====
fK = f_k(K[0], K[1])
coreK = abs(core_g(K[0], K[1]))
alpha_Z = (3e-3) / (2.0 * coreK)

# ===== Find k_F and calibrate β =====
def find_kF_on_right(kx, ky, s, Ebare):
    mask = s >= 0
    E = Ebare(kx[mask], ky[mask])
    idx = np.argmin(np.abs(E))
    return float(kx[mask][idx]), float(ky[mask][idx])

kF_kx, kF_ky = find_kF_on_right(kx, ky, s, Ebare)

def deltaE_total_at(kx1, ky1, alpha_Z, alpha_R, fK, beta):
    gz = gzz(kx1, ky1, fK, beta)
    gr = gR_mag(kx1, ky1)
    return 2.0*np.sqrt((alpha_Z*np.abs(gz))**2 + (alpha_R*gr)**2)

def calibrate_beta_for_eta(eta, target_meV, alpha_Z, kF_kx, kF_ky, fK):
    alpha_R = eta * alpha_Z
    target = target_meV * 1e-3
    betas = np.linspace(0.5, 200.0, 20001)
    vals = np.empty_like(betas)
    for i, b in enumerate(betas):
        vals[i] = deltaE_total_at(kF_kx, kF_ky, alpha_Z, alpha_R, fK, b)
    return float(betas[np.argmin(np.abs(vals - target))])

# ===== Multi-η loop =====
# eta_list = [0.00, 0.02, 0.06, 0.10]
eta_list = [0.00]
beta_book = {}
gr = gR_mag(kx, ky)

plt.figure(figsize=(7,5))
for eta in eta_list:
    beta = calibrate_beta_for_eta(eta, 13.0, alpha_Z, kF_kx, kF_ky, fK)
    beta_book[eta] = beta
    alpha_R = eta * alpha_Z
    gz = gzz(kx, ky, fK, beta)
    Gabs = np.sqrt((alpha_Z*gz)**2 + (alpha_R*gr)**2)

    # === 原能量表达式 ===
    E_plus  = Ebare(kx, ky) + Gabs
    E_minus = Ebare(kx, ky) - Gabs

    plt.plot(s, E_plus*1e3,  lw=1.2, color='red', label=f"η={eta:.2f} E+")
    plt.plot(s, E_minus*1e3, lw=1.2, color='blue', label=f"η={eta:.2f} E−")

# ===== Plot setup =====
plt.axvline(0, lw=1.0, ls='--')
plt.xlabel("k path near K")
plt.ylabel("Energy (meV)")
plt.title(r"Band energies $E_\pm(k)=E_{\text{bare}}\pm|G(k)|$ for different $\eta$")
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()

# ===== Check calibration =====
idx_K  = np.argmin(np.abs(s - 0.0))
idx_kF = np.argmin(np.hypot(kx - kF_kx, ky - kF_ky))
print("Calibration check (meV):")
for eta in eta_list:
    alpha_R = eta * alpha_Z
    beta = beta_book[eta]
    gz = gzz(kx, ky, fK, beta)
    dE = 2.0*np.sqrt((alpha_Z*gz)**2 + (alpha_R*gr)**2)*1e3
    print(f"η={eta:.2f}: ΔE(K)≈{dE[idx_K]:.3f}, ΔE(kF)≈{dE[idx_kF]:.3f}, β≈{beta:.3f}")
