"""Fig. 3 — Recta de rapidez CCM-DCM-grazing (versión publicación).
Cálculo idéntico a ../dcm_realista.py; solo cambia la presentación."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from estilo_ieee import plt, COLW, comprueba_solapes

import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve

# ---------- buck realista, carga ligera (identico a dcm_realista.py) ----------
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
            print(f"(barrido truncado en Iref = {Iref:.4f}: grazing)")
            Irefs = Irefs[:len(modo)]
            break
        x0, ton = z[:2], z[2]; g_ccm = list(z)
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
i_flip = np.argmax(mu1 < -1) if np.any(mu1 < -1) else None

# --- raices EXACTAS para las anotaciones -------------------------------------
# Los indices de arriba son nodos de la rejilla del barrido; anotarlos daria
# valores que no coinciden con los que cita el manuscrito. Se localizan por
# biseccion sobre la propia monodromia versorial.
from scipy.optimize import brentq

def ccm_exacta(Iref, guess):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T-ton)
        return [xc[0]-Iref, x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, guess, full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    xc = flw(A_on, b_on, x0, ton)
    S, s, _, _ = salt_pico(xc)
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    return x0[0], ton/T, np.sort(np.linalg.eigvals(M).real)

# frontera DCM->CCM: valle de corriente nulo
Iref_front = brentq(lambda I: ccm_exacta(I, [0.10, 4.5, 0.30*T])[0],
                    0.10, 0.20, xtol=1e-12)
_, D_front, mu_front = ccm_exacta(Iref_front, [0.10, 4.5, 0.30*T])
# cruce exacto de mu = -1
Iref_flip = brentq(lambda I: ccm_exacta(I, [0.35, 4.6, 0.45*T])[2][0] + 1.0,
                   0.45, 0.65, xtol=1e-12)
_, D_flip, _ = ccm_exacta(Iref_flip, [0.35, 4.6, 0.45*T])

print(f"Frontera DCM->CCM exacta:  Iref* = {Iref_front:.5f} A  "
      f"(D = {D_front:.5f})   mu_flip = {mu_front[0]:+.4f}")
print(f"flip mu=-1 exacto:  Iref = {Iref_flip:.5f} A  (D = {D_flip:.5f})")
print(f"rapidez: {lam_pk[0]:+.2f} -> {lam_pk[-1]:+.2f} (D = {Ds[-1]:.3f})")

# ---------- figura ----------
# Reglas de colocacion: cada rotulo va en una banda horizontal libre y su
# flecha es corta y vertical/oblicua sin cruzar la leyenda ni otra curva.
fig, (a1, a2) = plt.subplots(2, 1, figsize=(COLW, 4.25), sharex=True,
                             gridspec_kw=dict(hspace=0.09))
for ax in (a1, a2):
    ax.axvspan(Irefs[0], Iref_front, color='#d5e8d4', alpha=0.65, lw=0)
    ax.tick_params(length=2, pad=1.5)
    ax.grid(alpha=0.20)
    ax.margins(x=0.015)

# ===== panel superior: multiplicadores =====
# mu_voltage es casi constante (~0.99) y mu_flip baja de 0 a -1.7, asi que la
# franja 0.15 < y < 0.85 esta libre para TODO x: ahi van los rotulos, en dos
# alturas distintas para que no choquen entre si.
a1.plot(Irefs, mu1, '-', lw=1.0, color='#c0392b',
        label=r'$\mu_{\mathrm{flip}}$')
a1.plot(Irefs, mu2, '-', lw=1.0, color='#1a5276',
        label=r'$\mu_{\mathrm{voltage}}$')
a1.axhline(-1, color='0.35', lw=0.6, ls='--')
a1.axhline(0, color='0.85', lw=0.5)
a1.set_ylabel(r'multipliers (real part)', labelpad=2)
a1.set_ylim(-1.55, 1.62)
a1.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
a1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.02), ncol=2,
          frameon=False)
a1.text(0.5*(Irefs[0]+Iref_front), -0.70, r'DCM', ha='center', va='center',
        fontsize=6.6, color='#1e6b33', rotation=90)
# Solo hay UNA banda horizontal libre en este panel (entre mu_flip y
# mu_voltage), y no caben dos rotulos largos en ella: el de la colision de
# borde sube a la esquina superior izquierda, que la leyenda no ocupa.
a1.annotate(rf'border collision: $I_{{\mathrm{{ref}}}}^{{*}}={Iref_front:.4f}$~A',
            xy=(Iref_front, mu_front[0]), xytext=(0.072, 1.34),
            fontsize=6.4, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', lw=0.5, color='0.30',
                            shrinkA=1, shrinkB=2))
if i_flip:
    a1.plot([Iref_flip], [-1.0], 'o', ms=2.8, color='#7d3c98', zorder=5)
    # A la IZQUIERDA del punto: hacia la derecha el rotulo se sale del eje y
    # obliga a bbox_inches='tight' a ensanchar la figura, comprimiendo el
    # area de trazado al escalar a \columnwidth.
    a1.annotate(rf'flip: $\mu=-1$ at $I_{{\mathrm{{ref}}}}={Iref_flip:.4f}$~A',
                xy=(Iref_flip, -1.0),
                xytext=(Iref_flip-0.020, 0.25),
                fontsize=6.4, color='#7d3c98', ha='right', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.5, color='#7d3c98',
                                shrinkA=1, shrinkB=2))

# ===== panel inferior: rapidez =====
# lambda crece monotona de -2.7 a +4.1: la franja alta de la izquierda y la
# baja de la derecha estan libres.
a2.plot(Irefs, lam_pk, '-', lw=1.0, color='#b7791f',
        label=r'peak-event rapidity $\lambda=\ln|1+\sigma|$')
a2.axhline(0, color='0.35', lw=0.6, ls='--')
a2.set_xlabel(r'$I_{\mathrm{ref}}$~[A]', labelpad=2)
a2.set_ylabel(r'rapidity $\lambda$', labelpad=2)
lo, hi = min(lam_pk), max(lam_pk)
a2.set_ylim(lo-1.5, hi+2.3)
a2.legend(loc='upper left', bbox_to_anchor=(0.015, 1.02), frameon=False)
a2.annotate(r'$\lambda_{\mathrm{zero}}=-\infty$ in DCM',
            xy=(Irefs[0]+0.006, lo-0.10), xytext=(Iref_front+0.045, lo-1.02),
            fontsize=6.4, color='#1e6b33', ha='left', va='center',
            arrowprops=dict(arrowstyle='->', lw=0.5, color='#1e6b33',
                            shrinkA=1, shrinkB=2))
# lambda cruza 0 en Iref ~ 0.54, asi que la franja y<0 con x>0.6 esta libre:
# ahi va el rotulo del grazing, con la flecha subiendo al final de la curva.
a2.annotate(r'$\lambda\to+\infty$ (grazing, $D\to1$)',
            xy=(Irefs[-1], lam_pk[-1]),
            xytext=(Irefs[-1]-0.012, lo+0.55),
            fontsize=6.4, ha='right', va='center',
            arrowprops=dict(arrowstyle='->', lw=0.5, color='0.30',
                            shrinkA=1, shrinkB=2,
                            connectionstyle='arc3,rad=-0.18'))
fig.align_ylabels((a1, a2))
fig.tight_layout(pad=0.25)
assert not comprueba_solapes(fig), 'hay textos solapados en la Fig. 3'
out = pathlib.Path(__file__).resolve().parent / 'recta_rapidez_ccm_dcm_grazing.pdf'
fig.savefig(out, bbox_inches='tight')
print(f"figura -> {out}")
