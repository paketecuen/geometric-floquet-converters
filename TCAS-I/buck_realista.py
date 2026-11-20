import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq
from clifford import Cl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# ============ buck REALISTA: modo corriente de pico con rampa ============
Vin = 12.0
L, RL   = 100e-6, 0.060      # bobina y su resistencia serie
C, ESR  = 100e-6, 0.025      # condensador y su ESR
R       = 3.0                # carga
Ron     = 0.040              # MOSFET
Vd, Rd  = 0.45, 0.020        # diodo: caida y resistencia
fs      = 100e3
T       = 1.0/fs
Iref    = 3.35               # apunta a D ~ 0.8

al = R/(R+ESR)               # divisor carga/ESR
def matA(Rsw):
    return np.array([[-(Rsw+RL+al*ESR)/L, -al/L],
                     [al/C, -al/(R*C)]])
A_on,  b_on  = matA(Ron), np.array([Vin/L, 0.0])
A_off, b_off = matA(Rd),  np.array([-Vd/L, 0.0])

def aug(A, b):
    M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = b
    return M

def flw(A, b, x, t):
    E = expm(aug(A, b)*t)
    return E[:2, :2]@x + E[:2, 2]

def punto_de_trabajo(ma):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T-ton)
        return [xc[0]-(Iref-ma*ton), x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, [Iref-0.35, 8.6, 0.8*T], full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    xc = flw(A_on, b_on, x0, ton)
    fm = A_on@xc + b_on; fp = A_off@xc + b_off
    m1, m2 = fm[0], -fp[0]
    den = m1 + ma
    u = (fp-fm)/den
    S = np.eye(2) + np.outer(u, np.array([1.0, 0.0]))
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    return x0, ton, xc, m1, m2, u, S, M

# ---- 1) verificacion del diccionario con A_on != A_off ----
layout, blades = Cl(2, 2)
E4 = [blades['e%d' % k] for k in (1, 2, 3, 4)]
w = [(E4[0]+E4[2])/2, (E4[1]+E4[3])/2]
v = [(E4[0]-E4[2])/2, (E4[1]-E4[3])/2]

def biv(a):
    B = 0*E4[0]
    for i in range(2):
        for j in range(2):
            if a[i, j] != 0:
                B = B + 2*a[i, j]*(w[i] ^ v[j])
    return B

def mvexp(B, N=120):
    tot, term = 1 + 0*B, 1 + 0*B
    for k in range(1, N):
        term = term*B/k
        tot = tot + term
        if max(abs(term.value)) < 1e-18:
            break
    return tot

def inv(W):
    c = float((W*~W)(0)); return ~W/c

def accion(W):
    Wi = inv(W)
    return np.column_stack([np.array([2*float((v[i] | (W*w[k]*Wi))(0))
                                      for i in range(2)]) for k in range(2)])

ma0 = 45e3
x0, ton, xc, m1, m2, u, S, M = punto_de_trabajo(ma0)
s = u[0]
g = np.zeros((2, 2)); g[0, 0] = np.log(-(1+s))
W_M = (mvexp(biv(A_off*(T-ton))/2) * (E4[0]*E4[2]) * mvexp(biv(g)/2)
       * mvexp(biv(A_on*ton)/2))
print("Verificacion con parasitos (A_on != A_off), ma = 45 kA/s:")
print(f"  D = {ton/T:.4f}   m1 = {m1:.0f}  m2 = {m2:.0f} A/s   1+s = {1+s:.4f}")
print(f"  |accion(W_M) - M| = {np.max(np.abs(accion(W_M)-M)):.2e}")
print(f"  mu = {np.sort(np.linalg.eigvals(M).real)}")

# ---- 2) frontera predicha: mu = -1 exacto, y regla clasica corregida ----
def mu_min(ma):
    return np.min(np.linalg.eigvals(punto_de_trabajo(ma)[7]).real)

ma_exacta = brentq(lambda ma: mu_min(ma) + 1.0, 20e3, 45e3, xtol=1.0)
_, _, _, m1b, m2b, _, _, _ = punto_de_trabajo(ma_exacta)
ma_clasica = (m2b - m1b)/2
print(f"\nFrontera del subarmonico:")
print(f"  ma* exacta (Floquet/versores) = {ma_exacta/1e3:.3f} kA/s")
print(f"  regla clasica (m2-m1)/2      = {ma_clasica/1e3:.3f} kA/s  (pendientes con parasitos)")

# ---- 3) simulacion temporal por eventos, barrido de ma (vectorizada) ----
nma = 140
mas = np.linspace(5e3, 60e3, nma)
dt = T/1000.0
Eon  = expm(aug(A_on, b_on)*dt)
Eoff = expm(aug(A_off, b_off)*dt)
X = np.tile(np.array([Iref-0.35, 8.6, 1.0]), (nma, 1))   # estado aumentado
N_trans, N_keep = 320, 80
muestras = np.zeros((nma, N_keep))

for ciclo in range(N_trans + N_keep):
    on = np.ones(nma, bool)
    tloc = 0.0
    h0 = X[:, 0] - Iref                       # umbral en t=0
    on[h0 >= 0] = False                       # dispara de inmediato
    for k in range(1000):
        Xn = np.where(on[:, None], X @ Eon.T, X @ Eoff.T)
        tloc += dt
        umbral = Iref - mas*tloc
        cruza = on & (Xn[:, 0] >= umbral)
        if np.any(cruza):
            # refino lineal del instante de cruce dentro del paso
            h_prev = X[cruza, 0] - (Iref - mas[cruza]*(tloc-dt))
            h_new  = Xn[cruza, 0] - umbral[cruza]
            frac = (h_prev/(h_prev - h_new))[:, None]
            Xn[cruza] = X[cruza] + frac*(Xn[cruza] - X[cruza])
            resto = (1.0 - frac.ravel())*dt
            # completa el subpaso restante ya en OFF
            for idx, tr in zip(np.where(cruza)[0], resto):
                Xn[idx] = expm(aug(A_off, b_off)*tr) @ Xn[idx]
            on[cruza] = False
        X = Xn
    if ciclo >= N_trans:
        muestras[:, ciclo-N_trans] = X[:, 0]

# ---- 4) figura ----
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
for j in range(nma):
    ax.plot(np.full(N_keep, mas[j]/1e3), muestras[j], '.', ms=1.2,
            color='#1a5276', alpha=0.55)
ax.axvline(ma_exacta/1e3, color='#c0392b', lw=1.8, ls='--',
           label=f'frontera predicha (versores): {ma_exacta/1e3:.1f} kA/s')
ax.axvline(ma_clasica/1e3, color='#e67e22', lw=1.4, ls=':',
           label=f'regla clasica $(m_2\\!-\\!m_1)/2$: {ma_clasica/1e3:.1f} kA/s')
ax.set_xlabel('pendiente de rampa  $m_a$  [kA/s]')
ax.set_ylabel('$i_L$ muestreada al inicio de ciclo  [A]')
ax.set_title('Buck realista (RL, ESR, Ron, diodo) - bifurcacion vs rampa, D~0.8')
ax.legend(loc='upper left', fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/bifurcacion_buck_realista.png')
print("\nFigura guardada.")
