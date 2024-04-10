import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

L = 0.015
mu_b = 1.7093e-05
rho_g = 1.2242

def sseval(x, mu_b, rho_g, L, udata, PLdata):
    d_p = x[0]
    eps_bed = x[1]
    PLclc = L*(150*mu_b*(1-eps_bed)**2*udata/(d_p**2*eps_bed**3)+1.75*rho_g*(1-eps_bed)*udata**2/(d_p*eps_bed**3))
    sse = np.sum((PLdata - PLclc)**2)
    return sse

def rsquare(y, f, c=True):
    if c:
        r2 = max(0, 1 - np.sum((y - f)**2) / np.sum((y - np.mean(y))**2))
    else:
        r2 = 1 - np.sum((y - f)**2) / np.sum(y**2)
        if r2 < 0:
            print("Consider adding a constant term to your model")
            r2 = 0
    rmse = np.sqrt(np.mean((y - f)**2))
    return r2, rmse

df = pd.read_excel("velocity_pressure_data.xlsx")
udata = df['velocity'].values
PLdata = df['PLdata'].values

x0 = np.random.rand(2)

# Define bounds for eps_bed
bounds = [(1e-07, 1e-01), (0.8, 0.95)]

# Perform optimization
res = minimize(lambda x: sseval(x, mu_b, rho_g, L, udata, PLdata), x0, bounds=bounds)

bestx = res.x
d_p, eps_bed = bestx

PLclc = lambda u: L*(150*mu_b*(1-eps_bed)**2*u/(d_p**2*eps_bed**3)+1.75*rho_g*(1-eps_bed)*u**2/(d_p*eps_bed**3))
PLclc_2 = L*(150*mu_b*(1-eps_bed)**2*udata/(d_p**2*eps_bed**3)+1.75*rho_g*(1-eps_bed)*udata**2/(d_p*eps_bed**3))

R_2, _ = rsquare(PLdata, PLclc_2)

plt.plot(udata, PLdata, 'k*')
plt.plot(udata, PLclc(udata), 'k', label='Modelo simplificado')
plt.xlim([0, udata[-1]+0.02])
plt.ylim([0, PLdata[-1]+2000])
plt.xlabel('Velocidade(m s-1)')
plt.ylabel('C/C0')
plt.legend()
plt.show()

print(f"R_2: {R_2:.4f}")
print(f"d_p: {bestx[0]:.4e} m")
print(f"eps_bed: {bestx[1]:.4e} [ ]")