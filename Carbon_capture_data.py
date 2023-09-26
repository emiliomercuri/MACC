import numpy as np
import math
from scipy.optimize import fsolve
import matplotlib as plt
plt.rcParams["font.family"] = "Times New Roman"
# Defining Ergun Equation
def Ergun(u):
    return (DeltaP / (rho_g * u ** 2)) * (d_p / L) * (eps_bed ** 3 / (1 - eps_bed)) -\
        150 * ((1 - eps_bed) / (d_p * rho_g * u / mu_g)) - (7 / 4)
# Feed data
y_CO2 = 0.14
y_N2 = 1 - y_CO2
T = 298.15  # [K]
p_abs = 1  # [atm]
p_in_rel = 100 / 101325  # input em Pa e transformar para atm [atm]
p_in_abs = p_in_rel + p_abs  # [atm]
M_CO2 = 44.01  # [g mol-1]
M_N2 = 28.01  # [g mol-1]
M_g = y_CO2 * M_CO2 + y_N2 * M_N2  # [g mol-1]
rho_CO2 = (p_in_abs * 101325) * (M_CO2 / 1000) / (8.314 * T)  # [kg m-3]
rho_N2 = (p_in_abs * 101325) * (M_N2 / 1000) / (8.314 * T)  # [kg m-3]
rho_g = (y_CO2 / M_CO2) / (y_CO2 / M_CO2 + y_N2 / M_N2) * rho_CO2 + \
        (y_N2 / M_N2) / (y_CO2 / M_CO2 + y_N2 / M_N2) * rho_N2  # [kg m-3]

# Packed bed data
eps_bed = 0.5
d_p = 0.0025                                                                                                # [m]
D_bed = 0.1                                                                                                 # [m]
L = 0.3                                                                                                     # [m]
A_bed = math.pi * (D_bed / 2)**2                                                                            # [m2]

# Particle data
tau = 2.2 #tortuosity
d_pore = 2.46                                                                                               # [nm]

# Lennard Jones data
sig_CO2 = 3.9996                                                                                            # [A]
sig_N2 = 3.667                                                                                              # [A]
ek_CO2 = 190
ek_N2 = 99.8
kTe_CO2 = 1 / ek_CO2 * T
kTe_N2 = 1 / ek_N2 * T
omg_mu_CO2 = 1.16145 / kTe_CO2**0.14874 + 0.52487 / \
    math.exp(0.77320 * kTe_CO2) + 2.16178 / math.exp(2.43787 * kTe_CO2)
omg_mu_N2 = 1.16145 / kTe_N2**0.14874 + 0.52487 / \
    math.exp(0.77320 * kTe_N2) + 2.16178 / math.exp(2.43787 * kTe_N2)

# Viscosity estimation (Lennard Jones equation)
mu_CO2 = 2.6693e-5 * math.sqrt(M_CO2 * T) / (sig_CO2**2 * omg_mu_CO2)
mu_N2 = 2.6693e-5 * math.sqrt(M_N2 * T) / (sig_N2**2 * omg_mu_N2)

# Viscosity binary mixture estimation (Wilke Equation)
phi_aa = 1 / math.sqrt(8) * (1 + M_CO2 / M_CO2)**(-0.5) * \
    (1 + (mu_CO2 / mu_CO2)**0.5 * (M_CO2 / M_CO2)**0.25)**2
phi_ab = 1 / math.sqrt(8) * (1 + M_CO2 / M_N2)**(-0.5) * \
    (1 + (mu_CO2 / mu_N2)**0.5 * (M_CO2 / M_N2)**0.25)**2
phi_ba = 1 / math.sqrt(8) * (1 + M_N2 / M_CO2)**(-0.5) * \
    (1 + (mu_N2 / mu_CO2)**0.5 * (M_N2 / M_CO2)**0.25)**2
phi_bb = 1 / math.sqrt(8) * (1 + M_N2 / M_N2)**(-0.5) * \
    (1 + (mu_N2 / mu_N2)**0.5 * (M_N2 / M_N2)**0.25)**2
mu_g = y_CO2 * mu_CO2 / (y_CO2 * phi_aa + y_N2 * phi_ab) + \
    y_N2 * mu_N2 / (y_N2 * phi_bb + y_CO2 * phi_ba)  # [g cm-1 s-1]

# Molecular diffusivity estimation (Chapman-Enskog Equation)
sig_ab = 0.5 * (sig_CO2 + sig_N2)
eab_k = math.sqrt(ek_CO2 * ek_N2)
kTe_ab = 1 / eab_k * T
omg_Dab = 1.06036 / kTe_ab**0.15610 + 0.19300 / math.exp(0.47635 * kTe_ab) + \
    1.03587 / math.exp(1.52996 * kTe_ab) + 1.76474 / math.exp(3.894117 * kTe_ab)
Dab = 1.8583e-3 * math.sqrt(T**3 * (1 / M_CO2 + 1 / M_N2)) * 1 / \
    (p_in_abs * sig_ab**2 * omg_Dab) * 1e-4                                                                        # [m2 s-1]

# Effective diffusivity estimation
lam_ab = 8.3144e7 * T / (6.023e23 * math.sqrt(2) * math.pi * (sig_ab * 1e-8)**2 * (p_in_abs * 1.01325e6)) * 1e7    # [nm]
Dk = 4.85e3 * d_pore * 1e-7 * (T / (y_CO2 * M_CO2 + y_N2 * M_N2))**0.5 * 1e-4                               # [m2 s-1]
Deff = (tau * (1 / Dab + 1 / Dk))**(-1)                                                                     # [m2 s-1]

# Velocity estimation by Ergun Equation
DeltaP = p_in_rel * 101325  # Considerando P_out_abs = P_abs e transformando a unidade de atm para Pa
u0 = 0.001
u_avg = fsolve(Ergun, u0)
u_avg = u_avg.item() #Transformar u_avg de numpy.ndarray para float

# Dimensionless numbers estimation
Re_d = d_p*u_avg * rho_g / mu_g
Pe = 0.508 * Re_d**0.020 * (L / d_p)
Sc = (mu_g * 10) / (rho_g * Dab)
Sh = 1.09 * Re_d**0.27 * Sc**(1 / 3)
DL = u_avg * L / Pe                                                                                         # [m2 s-1]
kf = Sh * Dab / d_p                                                                                         # [m s-1]

# Permeability Kozeny-Carman equation
phi_sph = 1  # [sphericity] of the particles in the packed bed = 1 for spherical particles
k_bed = phi_sph**2 * eps_bed**3 * d_p**2 / (180 * (1 - eps_bed)**2)

# Clearing variables that are no longer needed
del ek_N2, ek_CO2, M_CO2, M_N2, phi_sph, sig_ab, sig_N2, sig_CO2, tau

print(f"P: {p_in_abs:.4f} atm")
print(f"T: {T:.2f} K")
print(f"mu_g: {mu_g/10:.4e} Pa s")
print(f"u_avg: {u_avg:.4f} m s-1")
print(f"Dab: {Dab:.4e} m2 s-1")
print(f"Dk: {Dk:.4e} m2 s-1")
print(f"Deff: {Deff:.4e} m2 s-1")
print(f"DL: {DL:.4e} m2 s-1")
print(f"kf: {kf:.4e} m2 s-1")
print(f"kappa_bed: {k_bed:.4e} m2")