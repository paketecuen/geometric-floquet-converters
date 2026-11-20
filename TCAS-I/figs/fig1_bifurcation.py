"""Fig. 1 — Frontera subarmonica del buck realista (version publicacion).

Dos paneles:
 (a) cascada global: muestreo del atractor en un barrido amplio de m_a.
 (b) discriminacion cuantitativa de la frontera. El barrido grueso de (a)
     NO puede resolver el hueco exacto-vs-clasico (0.374 kA/s frente a un
     paso de rejilla de 0.396 kA/s): las dos lineas caen practicamente en la
     misma celda. Y cerca de mu=-1 la convergencia es marginal, asi que
     "donde colapsa visualmente" no es un observable fiable.
     En su lugar se mide la AMPLITUD del ciclo 2 en una ventana fina y se
     usa que, en una bifurcacion flip supercritica, amplitud^2 es lineal en
     el parametro: el ajuste por minimos cuadrados se extrapola a cero y da
     la frontera con incertidumbre, comparable con las dos candidatas.

El integrador de eventos es exacto: el instante de cruce se resuelve por
biseccion sobre el flujo en forma cerrada (autodescomposicion de A), no por
paso fijo, de modo que la medida no arrastra error de rejilla temporal.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from estilo_ieee import plt, COLW, comprueba_solapes

import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

# ---------- modelo realista (identico a buck_realista.py) ----------
Vin = 12.0
L, RL   = 100e-6, 0.060
C, ESR  = 100e-6, 0.025
R       = 3.0
Ron     = 0.040
Vd, Rd  = 0.45, 0.020
fs      = 100e3
T       = 1.0/fs
Iref    = 3.35

al = R/(R+ESR)
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

# ---------- frontera exacta por versores (mu = -1) ----------
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
    u = (fp-fm)/(m1+ma)
    S = np.eye(2) + np.outer(u, np.array([1.0, 0.0]))
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    return x0, ton, xc, m1, m2, u, S, M

def mu_min(ma):
    return np.min(np.linalg.eigvals(punto_de_trabajo(ma)[7]).real)

ma_exacta = brentq(lambda ma: mu_min(ma) + 1.0, 20e3, 45e3, xtol=1.0)
_, ton_b, _, m1b, m2b, _, _, _ = punto_de_trabajo(ma_exacta)
ma_clasica = (m2b - m1b)/2
print(f"ma* exacta = {ma_exacta/1e3:.4f} kA/s   clasica = {ma_clasica/1e3:.4f}"
      f" kA/s   hueco = {(ma_clasica-ma_exacta)/1e3:.4f} kA/s")
print(f"D en la frontera = {ton_b/T:.6f}")

# ---------- integrador de eventos exacto ----------
# flujo en forma cerrada: x(t) = e^{At}x0 + A^{-1}(e^{At}-I)b, via autovalores
def preparar(A, b):
    lam, V = np.linalg.eig(A)
    Vi = np.linalg.inv(V)
    Ainvb = np.linalg.solve(A, b)
    return lam, V, Vi, Ainvb

PON, POFF = preparar(A_on, b_on), preparar(A_off, b_off)

def flujo(P, x, t):
    lam, V, Vi, Ainvb = P
    E = (V * np.exp(lam*t)) @ Vi
    return np.real(E @ (x + Ainvb) - Ainvb)

def un_ciclo(x0, ma):
    """Un periodo: ON hasta cruzar la rampa, OFF el resto. Devuelve x(T) y
    el pico. Si no hay cruce en (0,T), el ciclo saturea en ON."""
    def h(t):
        return flujo(PON, x0, t)[0] - (Iref - ma*t)
    if h(0.0) >= 0.0:                      # ya por encima: apaga de inmediato
        return flujo(POFF, x0, T), x0[0]
    if h(T) < 0.0:                          # nunca cruza: duty saturado
        return flujo(PON, x0, T), flujo(PON, x0, T)[0]
    ton = brentq(h, 0.0, T, xtol=1e-14, rtol=1e-14)
    xc = flujo(PON, x0, ton)
    return flujo(POFF, xc, T-ton), xc[0]

def atractor(ma, n_trans, n_keep, x0=None):
    x = np.array([Iref-0.35, 8.6]) if x0 is None else x0.copy()
    for _ in range(n_trans):
        x, _ = un_ciclo(x, ma)
    pk = np.empty(n_keep)
    for j in range(n_keep):
        x, pk[j] = un_ciclo(x, ma)
    return pk, x


def es_ciclo2(pk, tol=2e-6):
    """True si el atractor muestreado es un 2-ciclo limpio: las dos
    subsecuencias alternas deben ser constantes. Sin este filtro, los puntos
    de periodo alto o caoticos de la izquierda del rango contaminan el ajuste
    de amplitud (fue exactamente el error de la primera version)."""
    a, b = pk[0::2], pk[1::2]
    return max(np.ptp(a), np.ptp(b)) < tol and np.ptp(pk) > 10*tol

# ---------- panel (a): cascada global ----------
mas_a = np.linspace(5e3, 60e3, 120)
print("simulando cascada global...")
casc = [atractor(ma, 400, 60)[0] for ma in mas_a]

# ---------- panel (b): amplitud del ciclo 2 en ventana fina ----------
# Ventana ESTRECHA por debajo de la frontera, la unica en la que el atractor
# es un 2-ciclo limpio (por debajo de ~33.3 kA/s hay periodo alto y caos).
# Se recorre con continuacion, arrastrando el estado convergido de un m_a al
# siguiente, para que el transitorio no domine el coste cerca de mu=-1.
mas_b = np.linspace(33.35e3, 34.02e3, 40)
print("midiendo amplitud del ciclo 2 (continuacion)...")
amp, ok, xc = [], [], None
for i, ma in enumerate(mas_b):
    pk, xc = atractor(ma, 2500 if i == 0 else 700, 60, xc)
    amp.append(np.ptp(pk)); ok.append(es_ciclo2(pk))
amp, ok = np.array(amp), np.array(ok)
print(f"  puntos con 2-ciclo limpio: {ok.sum()}/{len(mas_b)}")
mb, amp2 = mas_b[ok], amp[ok]**2
# ajuste lineal de amp^2 y su corte con cero
cf, cov = np.polyfit(mb, amp2, 1, cov=True)
ma_fit = -cf[1]/cf[0]
J = np.array([cf[1]/cf[0]**2, -1.0/cf[0]])      # propagacion de 1er orden
sig = float(np.sqrt(J @ cov @ J))
r2 = 1 - np.sum((amp2-np.polyval(cf, mb))**2)/np.sum((amp2-amp2.mean())**2)
print(f"ajuste amp^2 (R^2 = {r2:.5f}) -> frontera = "
      f"{ma_fit/1e3:.4f} +/- {sig/1e3:.4f} kA/s")
print(f"  |ajuste - exacta|  = {abs(ma_fit-ma_exacta)/1e3:.4f} kA/s "
      f"({abs(ma_fit-ma_exacta)/sig:.2f} sigma)")
print(f"  |ajuste - clasica| = {abs(ma_fit-ma_clasica)/1e3:.4f} kA/s "
      f"({abs(ma_fit-ma_clasica)/sig:.2f} sigma)")

# ---------- figura ----------
fig, (aa, ab) = plt.subplots(2, 1, figsize=(COLW, 4.45),
                             gridspec_kw=dict(hspace=0.52))

for j, ma in enumerate(mas_a):
    aa.plot(np.full(len(casc[j]), ma/1e3), casc[j], '.', ms=0.9,
            color='#1a5276', alpha=0.5, rasterized=True)
aa.axvline(ma_exacta/1e3, color='#c0392b', lw=1.0, ls='--')
aa.set_xlabel(r'compensation-ramp slope $m_a$~[kA/s]', labelpad=2)
aa.set_ylabel(r'peak $i_L$~[A]', labelpad=2)
aa.set_title(r'(a) global cascade', fontsize=7, pad=3)
aa.grid(alpha=0.25)
aa.margins(x=0.02)

ab.plot(mb/1e3, amp2*1e3, 'o', ms=2.4, mfc='none', mew=0.7,
        color='#1a5276', label=r'simulated $\Delta^{2}$ (event-driven)')
xr = np.linspace(mb[0], ma_clasica + 0.15e3, 60)
ab.plot(xr/1e3, np.polyval(cf, xr)*1e3, '-', lw=0.9, color='#1a5276',
        alpha=0.8, label=rf'fit $\to$ zero at ${ma_fit/1e3:.3f}\pm{sig/1e3:.3f}$')
ab.axhline(0, color='0.8', lw=0.5)
ab.axvline(ma_exacta/1e3, color='#c0392b', lw=1.0, ls='--',
           label=rf'versor prediction: {ma_exacta/1e3:.3f}')
ab.axvline(ma_clasica/1e3, color='#e67e22', lw=1.0, ls=':',
           label=rf'classical rule (boundary orbit): {ma_clasica/1e3:.3f}')
ab.set_xlabel(r'compensation-ramp slope $m_a$~[kA/s]', labelpad=2)
ab.set_ylabel(r'$\Delta^{2}$~[$10^{-3}\,$A$^2$]', labelpad=2)
ab.set_title(r'(b) period-2 amplitude, extrapolated', fontsize=7, pad=3)
ab.grid(alpha=0.25)
ab.set_xlim(mb[0]/1e3 - 0.10, ma_clasica/1e3 + 0.16)
ab.set_ylim(-3.6, 22.0)
# Leyenda FUERA del area de datos, bajo el rotulo del eje: es la unica banda
# que no cruzan ni las verticales candidatas ni la recta de ajuste.
ab.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=2,
          fontsize=5.8, frameon=False, handlelength=1.4,
          labelspacing=0.25, columnspacing=1.2, borderaxespad=0.0)

fig.align_ylabels((aa, ab))
fig.tight_layout(pad=0.25)
assert not comprueba_solapes(fig), 'hay textos solapados en la Fig. 1'
out = pathlib.Path(__file__).resolve().parent / 'bifurcacion_buck_realista.pdf'
fig.savefig(out, bbox_inches='tight')
print(f"figura -> {out}")
