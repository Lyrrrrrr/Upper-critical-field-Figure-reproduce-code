import numpy as np
import matplotlib.pyplot as plt

# TB parameters
a = 3.18
t1 = 146e-3;
t2 = -0.40*t1;
t3 = 0.25*t1;
mu = 0

# Zeeman-type spin-splitting parameters
alpha = 8e-3   # spin-splitting strength
beta  = 1.0    # F(k)=beta*tanh[f(K)-f(k)]-1 

# energy dispersion, corresponding to H_{kin} term
def eps_Mo(kx, ky, mu):
    return (2*t1*(np.cos(ky*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(0.5*ky*a))
            +2*t2*(np.cos(np.sqrt(3)*kx*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(1.5*ky*a))
            +2*t3*(np.cos(2*ky*a) + 2*np.cos(np.sqrt(3)*kx*a)*np.cos(ky*a))-mu)

# g_z(k) for Zeeman-type SOC
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

# Compute spin-split bands
E = E(kx, ky)
gz = gzz(kx, ky, fK, beta)

spin_splitting = alpha * np.abs(gz)

E_plus = E + spin_splitting
E_minus = E - spin_splitting


# Plot
plt.figure(figsize=(7,5))
plt.plot(s, E_plus,  lw=1.8, color='red', label=r'$E_+(k)$')
plt.plot(s, E_minus, lw=1.8, color='blue', label=r'$E_-(k)$')
plt.axvline(0, color='k', lw=1.0, ls='--')   # K
plt.axhline(0, color='k', lw=0.8, ls=':')    # EF
plt.xlim(-0.2, 0.2)
plt.ylim(-0.20, 0.05)
plt.xlabel(r'$k$ (Å$^{-1}$)')
plt.ylabel(r'$E - E_F$ (eV)')
plt.title(r'Fig. 4(b) reproduction (Zeeman-type only)')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.show()
