# -*- coding: utf-8 -*-
"""
Multi-η sweep without changing your final energy expression.
- Calibrate α_Z from ΔE(K) ≈ 3 meV.
- For each η = α_R / α_Z, auto-calibrate β such that ΔE(k_F) ≈ 13 meV.
- Plot ΔE(k) = 2|G(k)| along Γ–K–M near K.
If your original plot is bands E± = E0 ± |G|, just uncomment the lines near the end.
"""

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

# ===== g_z(k): out-of-plane Zeeman-type form =====
def core_g(kx, ky):
    return np.sin(ky*a) - 2*np.cos(np.sqrt(3)/2*kx*a)*np.sin(0.5*ky*a)

def f_k(kx, ky):
    return np.abs(core_g(kx, ky))

def F_k(kx, ky, fK, beta):
    return beta * np.tanh(fK - f_k(kx, ky)) - 1.0

def gzz(kx, ky, fK, beta):
    return F_k(kx, ky, fK, beta) * core_g(kx, ky)

# ===== g_R(k): in-plane Rashba-type form =====
def gR_vec(kx, ky):
    gx = np.sin(0.5*ky*a) * np.cos(np.sqrt(3)*0.5*kx*a)
    gy = -np.sin(np.sqrt(3)*0.5*kx*a) * np.cos(0.5*ky*a)
    return gx, gy

def gR_mag(kx, ky):
    gx, gy = gR_vec(kx, ky)
    return np.sqrt(gx*gx + gy*gy)

# ===== Build path and align E_F (doping by EF shift) =====
kx, ky, s = path_around_K(smax=0.2, N=801)

mu_guess = 0.10  # eV
E_K_raw = eps_Mo(K[0], K[1], mu_guess)
E_F = E_K_raw + 0.15   # valley ≈ -0.15 eV (shifted)
def Ebare(kx, ky):
    return eps_Mo(kx, ky, mu_guess) - E_F

# ===== α_Z from ΔE(K) ≈ 3 meV =====
fK = f_k(K[0], K[1])
coreK = abs(core_g(K[0], K[1]))
alpha_Z = (3e-3) / (2.0 * coreK)   # eV

# ===== Utilities for k_F and β calibration =====
def find_kF_on_right(kx, ky, s, Ebare):
    mask_R = (s >= 0)
    Ei = Ebare(kx[mask_R], ky[mask_R])
    j = int(np.argmin(np.abs(Ei)))
    return float(kx[mask_R][j]), float(ky[mask_R][j])

kF_kx, kF_ky = find_kF_on_right(kx, ky, s, Ebare)

def deltaE_total_at(kx1, ky1, alpha_Z, alpha_R, fK, beta):
    gz = gzz(kx1, ky1, fK, beta)
    gr = gR_mag(kx1, ky1)
    return 2.0 * np.sqrt((alpha_Z*np.abs(gz))**2 + (alpha_R*gr)**2)  # eV

def calibrate_beta_for_eta(eta, target_meV, alpha_Z, kF_kx, kF_ky, fK):
    alpha_R = eta * alpha_Z
    target = target_meV * 1e-3
    betas = np.linspace(0.5, 200.0, 20001)
    vals = np.empty_like(betas)
    for i, b in enumerate(betas):
        vals[i] = deltaE_total_at(kF_kx, kF_ky, alpha_Z, alpha_R, fK, b)
    return float(betas[int(np.argmin(np.abs(vals - target)))])

# ===== Sweep η, without changing your final energy expression =====
# eta_list = [0.00, 0.02, 0.06, 0.08, 0.10]
eta_list = [0.00, 0.02, 0.6, 0.8, 1.0]
beta_book = {}
gr = gR_mag(kx, ky)

plt.figure(figsize=(7,5))
for eta in eta_list:
    # 1) β calibration for this η
    beta = calibrate_beta_for_eta(
        eta=eta, target_meV=13.0, alpha_Z=alpha_Z,
        kF_kx=kF_kx, kF_ky=kF_ky, fK=fK
    )
    beta_book[eta] = beta

    # 2) Parameters for this η (α_R, β)
    alpha_R = eta * alpha_Z

    # 3) Your original energy expression starts here:
    #    Keep as-is; only parameters (alpha_R, beta) change by η.
    gz = gzz(kx, ky, fK, beta)
    Gabs = np.sqrt((alpha_Z*gz)**2 + (alpha_R*gr)**2)

    # ---- If you plot splitting (default here): ΔE(k) = 2|G(k)| ----
    plt.plot(s, 2.0*Gabs*1e3, lw=1.6, label=f"η = {eta:.2f}")

    # ---- If your original plot is bands, uncomment the following two lines ----
    # E_plus  = Ebare(kx, ky) + Gabs
    # E_minus = Ebare(kx, ky) - Gabs
    # plt.plot(s, E_plus*1e3,  lw=1.2, label=f"η = {eta:.2f} (E+)")
    # plt.plot(s, E_minus*1e3, lw=1.2, label=f"η = {eta:.2f} (E−)")

# ===== Decorations =====
plt.axvline(0, lw=1.0, ls='--')
plt.xlabel("k path near K")
plt.ylabel("ΔE(k) (meV)")  # 若画能带，把 ylabel 改为 "Energy (meV)"
plt.title(r"Total splitting $\Delta E(k)=2|G(k)|$ for multiple $\eta$")
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()

# ===== Calibration checks =====
idx_K  = int(np.argmin(np.abs(s - 0.0)))
idx_kF = int(np.argmin(np.hypot(kx - kF_kx, ky - kF_ky)))
print("Calibration check (meV):")
for eta in eta_list:
    alpha_R = eta * alpha_Z
    beta = beta_book[eta]
    gz = gzz(kx, ky, fK, beta)
    dE = 2.0*np.sqrt((alpha_Z*gz)**2 + (alpha_R*gr)**2)*1e3
    print(f"  eta={eta:.2f}: ΔE(K)≈{dE[idx_K]:.3f}, ΔE(k_F)≈{dE[idx_kF]:.3f}, beta≈{beta:.3f}")
