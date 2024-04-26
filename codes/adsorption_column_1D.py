import numpy as np
import matplotlib.pyplot as plt

rho_p = 850
M_CO2 = 0.04401
dp = 100e-06
L = 15e-02
eps = 0.5
Dax = 4.7e-08
qm = 0.01
KL = 2.5
c0 = 0.625
u0 = 4e-4
kLDF = 0.008

dt = 0.1
tf = 5000 + dt
t = np.arange(0, tf, dt)

Nx = 50
x = np.linspace(0, L, Nx+1)
dx = x[1] - x[0]

c = np.zeros(Nx)
q = np.zeros(Nx)
qe = np.zeros(Nx)
q_old = np.zeros(Nx)
dqdt = np.zeros(Nx)
time = []
time.append(0)
ct = []
ct.append(0)


for i in t:
    for j in range(0,Nx):
        if j == 0:
            c[j] = (c0-Dax/(2*u0*dx)*c[j +2] + 4*Dax*c[j + 1]/2/dx/u0)/(1+3*Dax/2/dx/u0)
        elif j == Nx-1:
            c[j] = c[j - 1]
        else:
            qe[j] = qm * KL * c[j] / (1 + KL * c[j])
            q_old[j] = q[j]
            q[j] = q[j] + dt * kLDF * (qe[j] - q[j])
            dqdt[j] = (q[j] - q_old[j]) / dt
            ae = (Dax*dt/dx**2) / (1 + u0*dt/dx + 2*Dax*dt/dx**2)
            a0 = 1 / (1 + u0*dt/dx + 2*Dax*dt/dx**2)
            aw = (u0*dt/dx + Dax*dt/dx**2) / (1 + u0*dt/dx + 2*Dax*dt/dx**2)
            f = (1-eps)/eps * rho_p * dqdt[j]
            c[j] = a0 * c[j] + ae * c[j + 1] + aw * c[j - 1] - f*dt

    if i == 0:
        continue
    elif i % 60 == 0:
        time.append(i)
        ct.append(c[-1])

c_c0 = []
for i in ct:
    c_c0.append(i / c0)

plt.plot(time,c_c0)
plt.xlabel('time (s)')
plt.ylabel('c/c$_{0}$')
plt.show()