import numpy as np
import matplotlib.pyplot as plt

# TB parameters
a = 3.18
t1 = 146e-3;
t2 = -0.40*t1;
t3 = 0.25*t1;
mu = 100

# energy dispersion, corresponding to H_{kin} term
def eps_Mo(kx, ky):
    return (2*t1*(np.cos(ky*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(0.5*ky*a))
            +2*t2*(np.cos(np.sqrt(3)*kx*a) + 2*np.cos(np.sqrt(3)/2*kx*a)*np.cos(1.5*ky*a))
            +2*t3*(np.cos(2*ky*a) + 2*np.cos(np.sqrt(3)*kx*a)*np.cos(ky*a))-mu)


# High symmetry points
Gamma = np.array([0.0, 0.0])
K     = np.array([0.0, 4*np.pi/(3*a)])
M     = np.array([np.pi/(np.sqrt(3)*a), np.pi/a])

# 2D direction vector, vKM[0] is step on M, vKM[1] is step on Gamma
vKM = (M - K) / np.linalg.norm(M - K)

dk  = 0.2                                # Å^-1
N   = 601
s   = np.linspace(-dk, dk, N)
kx  = K[0] + s * vKM[0]
ky  = K[1] + s * vKM[1]

# 计算能量并以 E(K) 为零点
E   = eps_Mo(kx, ky)
E  -= eps_Mo(K[0], K[1])

# plot
plt.figure(figsize=(5,6))
plt.plot(s, E, color='black', lw=1.8)
plt.xlabel(r'$k - K$ (Å$^{-1}$)')
plt.ylabel(r'$E - E(K)$ (eV)')
plt.title('Unsplit band near $K$  (1D slice along K→M)')
plt.grid(True, ls=':')
plt.tight_layout()
plt.show()
