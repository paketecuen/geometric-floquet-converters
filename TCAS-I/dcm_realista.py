import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq
from clifford import Cl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# ---------- buck realista, carga ligera ----------
Vin = 12.0
L, RL  = 100e-6, 0.060
C, ESR = 100e-6, 0.025
R      = 15.0
Ron    = 0.040
Vd, Rd = 0.45, 0.020
fs = 100e3; T = 1.0/fs
al = R/(R+ESR)

def matA(Rsw):
    return np.array([[-(Rsw+RL+al*ESR)/L, -al/L],
                     [al/C, -al/(R*C)]])
A_on,  b_on  = matA(Ron), np.array([Vin/L, 0.0])
A_off, b_off = matA(Rd),  np.array([-Vd/L, 0.0])
A_idl, b_idl = np.array([[0.0, 0.0], [0.0, -al/(R*C)]]), np.zeros(2)

def flw(A, b, x, t):
    M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = b
    E = expm(M*t)
    return E[:2, :2]@x + E[:2, 2]

def orbita_ccm(Iref, guess):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T-ton)
        return [xc[0]-Iref, x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, guess, full_output=True)
    return (z, ier == 1)

def orbita_dcm(Iref, guess):
    def res(z):
        v0, ton, tz = z
        x0 = np.array([0.0, v0])
        xpk = flw(A_on, b_on, x0, ton)
        xz  = flw(A_off, b_off, xpk, tz-ton)
        xT  = flw(A_idl, b_idl, xz, T-tz)
        return [xpk[0]-Iref, xz[0], xT[1]-v0]
    z, info, ier, msg = fsolve(res, guess, full_output=True)
    return (z, ier == 1)

def salt_pico(xc, ma=0.0):
    fm = A_on@xc + b_on; fp = A_off@xc + b_off
    den = fm[0] + ma
    u = (fp-fm)/den
    return np.eye(2) + np.outer(u, [1.0, 0.0]), u[0], fm[0], -fp[0]

def salt_cero(xz):
    fm = A_off@xz + b_off; fp = A_idl@xz + b_idl
    u = (fp-fm)/fm[0]
    return np.eye(2) + np.outer(u, [1.0, 0.0]), u[0]

# ---------- barrido con continuacion ----------
Irefs = np.linspace(0.06, 0.83, 160)
g_dcm, g_ccm = [0.35, 0.5e-6, 8e-6], None
modo, mu1, mu2, lam_pk, Ds = [], [], [], [], []
frontera = None
for Iref in Irefs:
    z, ok = orbita_dcm(Iref, g_dcm)
    es_dcm = ok and (z[2] < T) and (z[1] < z[2])
    if es_dcm:
        v0, ton, tz = z; g_dcm = list(z)
        xpk = flw(A_on, b_on, np.array([0.0, v0]), ton)
        xz  = flw(A_off, b_off, xpk, tz-ton)
        S_pk, s_pk, m1, m2 = salt_pico(xpk)
        S_0, s_0 = salt_cero(xz)
        M = expm(A_idl*(T-tz)) @ S_0 @ expm(A_off*(tz-ton)) @ S_pk @ expm(A_on*ton)
        if g_ccm is None:
            g_ccm = [0.0, v0, ton]
        modo.append('DCM'); Ds.append(ton/T)
    else:
        if g_ccm is None:
            g_ccm = [Iref-0.1, 7.5*Iref, 0.4*T]
        z, ok = orbita_ccm(Iref, g_ccm)
        if not ok:
            print(f"(barrido truncado en Iref = {Iref:.4f}: orbita al limite del grazing)")
            Irefs = Irefs[:len(modo)]
            break
        x0, ton = z[:2], z[2]; g_ccm = list(z)
        if frontera is None and modo and modo[-1] == 'DCM':
            frontera = Iref
        xc = flw(A_on, b_on, x0, ton)
        S_pk, s_pk, m1, m2 = salt_pico(xc)
        M = expm(A_off*(T-ton)) @ S_pk @ expm(A_on*ton)
        modo.append('CCM'); Ds.append(ton/T)
    mu = np.sort(np.linalg.eigvals(M).real)
    mu1.append(mu[0]); mu2.append(mu[-1])
    lam_pk.append(np.log(abs(1+s_pk)))

mu1, mu2, lam_pk = map(np.array, (mu1, mu2, lam_pk))
fis = np.array(Ds) < 0.99
Irefs, mu1, mu2, lam_pk = Irefs[fis], mu1[fis], mu2[fis], lam_pk[fis]
modo = [m for m, f in zip(modo, fis) if f]
Ds = [d for d, f in zip(Ds, fis) if f]
i_front = modo.index('CCM')
Iref_front = (Irefs[i_front-1] + Irefs[i_front])/2
i_flip = np.argmax(mu1 < -1) if np.any(mu1 < -1) else None
print(f"Frontera DCM->CCM:  Iref* = {Iref_front:.4f} A")
print(f"  mu_flip justo en CCM: {mu1[i_front]:.4f}   en DCM: {mu1[i_front-1]:.4f} (exactamente 0)")
if i_flip:
    print(f"Inestabilidad flip (mu=-1) en Iref = {Irefs[i_flip]:.3f} A  (D = {Ds[i_flip]:.3f})")
print(f"Rapidez del pico: {lam_pk[0]:+.2f} (DCM bajo) -> {lam_pk[-1]:+.2f} (D = {Ds[-1]:.3f}, hacia grazing)")

# ---------- verificacion versorial en un punto de cada modo ----------
layout, blades = Cl(2, 2)
E4 = [blades['e%d' % k] for k in (1, 2, 3, 4)]
w = [(E4[0]+E4[2])/2, (E4[1]+E4[3])/2]
v = [(E4[0]-E4[2])/2, (E4[1]-E4[3])/2]
def biv(a):
    B = 0*E4[0]
    for i in range(2):
        for j in range(2):
            if a[i, j] != 0: B = B + 2*a[i, j]*(w[i] ^ v[j])
    return B
def mvexp(B, N=140):
    tot, term = 1 + 0*B, 1 + 0*B
    for k in range(1, N):
        term = term*B/k; tot = tot + term
        if max(abs(term.value)) < 1e-18: break
    return tot
def inv(W): c = float((W*~W)(0)); return ~W/c
def accion(W):
    Wi = inv(W)
    return np.column_stack([np.array([2*float((v[i] | (W*w[k]*Wi))(0))
                                      for i in range(2)]) for k in range(2)])

# CCM medio
Iref = 0.40
z, _ = orbita_ccm(Iref, [0.15, 3.0, 0.35*T]); x0, ton = z[:2], z[2]
xc = flw(A_on, b_on, x0, ton)
S_pk, s_pk, _, _ = salt_pico(xc)
g = np.zeros((2, 2)); g[0, 0] = np.log(-(1+s_pk))
W = mvexp(biv(A_off*(T-ton))/2) * (E4[0]*E4[2]) * mvexp(biv(g)/2) * mvexp(biv(A_on*ton)/2)
Mref = expm(A_off*(T-ton)) @ S_pk @ expm(A_on*ton)
print(f"\nVerificacion CCM (Iref=0.40): |accion-M| = {np.max(np.abs(accion(W)-Mref)):.2e}")

# DCM: los versores llevan los tramos invertibles; S0 queda como proyector frontera
Iref = 0.10
z, _ = orbita_dcm(Iref, [0.5, 0.6e-6, 7e-6]); v0, ton, tz = z
xpk = flw(A_on, b_on, np.array([0.0, v0]), ton)
xz  = flw(A_off, b_off, xpk, tz-ton)
S_pk, s_pk, _, _ = salt_pico(xpk)
S_0, s_0 = salt_cero(xz)
g = np.zeros((2, 2)); g[0, 0] = np.log(-(1+s_pk))
tramo1 = accion(mvexp(biv(A_on*ton)/2))
tramo2 = accion(mvexp(biv(A_off*(tz-ton))/2) * (E4[0]*E4[2]) * mvexp(biv(g)/2))
tramo3 = accion(mvexp(biv(A_idl*(T-tz))/2))
M_mix = tramo3 @ S_0 @ tramo2 @ tramo1
M_dcm = expm(A_idl*(T-tz)) @ S_0 @ expm(A_off*(tz-ton)) @ S_pk @ expm(A_on*ton)
print(f"Verificacion DCM (Iref=0.10): |mixto-M| = {np.max(np.abs(M_mix-M_dcm)):.2e}"
      f"   1+s_cero = {1+s_0:.2e}   mu = {np.sort(np.linalg.eigvals(M_dcm).real)}")

# ---------- figura ----------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.6), dpi=150, sharex=True)
dcm_mask = np.array([m == 'DCM' for m in modo])
for ax in (a1, a2):
    ax.axvspan(Irefs[0], Iref_front, color='#d5e8d4', alpha=0.55)
a1.plot(Irefs, mu1, '.-', ms=3, lw=0.8, color='#c0392b', label='$\\mu_{flip}$')
a1.plot(Irefs, mu2, '.-', ms=3, lw=0.8, color='#1a5276', label='$\\mu_{tension}$')
a1.axhline(-1, color='0.4', lw=0.8, ls='--')
a1.axhline(0, color='0.85', lw=0.7)
if i_flip:
    a1.axvline(Irefs[i_flip], color='#7d3c98', lw=1.2, ls=':',
               label=f'flip $\\mu=-1$ (Iref={Irefs[i_flip]:.2f})')
a1.annotate('DCM: $\\mu_{flip}=0$ exacto\n(proyector)', xy=(0.09, 0.06),
            fontsize=8, color='#1e8449')
a1.annotate(f'salto en la frontera\nIref*={Iref_front:.3f} A', xy=(Iref_front, -0.25),
            fontsize=8, ha='left')
a1.set_ylabel('multiplicadores (parte real)')
a1.set_ylim(-1.6, 1.15); a1.legend(fontsize=8, loc='center right'); a1.grid(alpha=0.25)
a1.set_title('Buck realista, carga ligera (R=15$\\Omega$): la vida completa en un barrido')

a2.plot(Irefs, lam_pk, '.-', ms=3, lw=0.8, color='#b7791f',
        label='rapidez del pico  $\\lambda=\\ln|1+s|$')
a2.axhline(0, color='0.4', lw=0.8, ls='--')
a2.annotate('$\\lambda_{cero}=-\\infty$ (DCM)', xy=(0.085, -1.6), fontsize=8,
            color='#1e8449')
a2.annotate('$\\lambda\\to+\\infty$\n(grazing, $D\\to1$)', xy=(0.76, 2.0), fontsize=8)
a2.set_xlabel('$I_{ref}$  [A]'); a2.set_ylabel('rapidez  $\\lambda$')
a2.legend(fontsize=8, loc='upper left'); a2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/recta_rapidez_ccm_dcm_grazing.png')
print("\nFigura guardada.")
