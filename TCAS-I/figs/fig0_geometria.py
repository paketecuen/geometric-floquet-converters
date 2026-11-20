"""Fig. 1 — La geometria que el articulo vende, dibujada.

(a) Accion de una saltacion S = I + u n^T: fija el hiperplano ker n^T punto a
    punto y escala la direccion de cruce por 1+s. Las tres especies son los
    tres signos de 1+s, y se ven como tres imagenes del mismo vector u.

(b) La recta de rapidez NO es una recta: es un par de hojas, una por signo de
    det S (equivalentemente por valor de W~W), pegadas por el punto de
    clamping 1+s = 0, donde lambda = -infinito y el versor sale del grupo.
    Grazing es el extremo lambda -> +infinito de ambas hojas.
    Encima se dibuja la travesia real de la Tabla II: al subir la rampa de
    compensacion, el evento de pico recorre la hoja de reflexion hacia la
    izquierda, toca el punto de pegado justo en el deadbeat m_a = m_2, y sale
    por la hoja de pass-through. Los lambda se recalculan aqui del modelo, no
    se copian de la tabla.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from estilo_ieee import plt, COLW, comprueba_solapes

import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

# ================= travesia de especies (misma que la Tabla II) =============
Vin, L, RL = 12.0, 100e-6, 0.060
C, ESR, R = 100e-6, 0.025, 3.0
Ron, Vd, Rd = 0.040, 0.45, 0.020
T, Iref = 1.0/100e3, 3.35
al = R/(R+ESR)


def matA(Rsw):
    return np.array([[-(Rsw+RL+al*ESR)/L, -al/L], [al/C, -al/(R*C)]])


A_on, b_on = matA(Ron), np.array([Vin/L, 0.0])
A_off, b_off = matA(Rd), np.array([-Vd/L, 0.0])


def flw(A, b, x, t):
    M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = b
    E = expm(M*t)
    return E[:2, :2]@x + E[:2, 2]


def trabajo(ma):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T-ton)
        return [xc[0]-(Iref-ma*ton), x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, [Iref-0.35, 8.6, 0.6*T], full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    xc = flw(A_on, b_on, x0, ton)
    fm, fp = A_on@xc + b_on, A_off@xc + b_off
    m1, m2 = fm[0], -fp[0]
    return (ma - m2)/(m1 + ma), m2


ma_db = brentq(lambda ma: trabajo(ma)[1] - ma, 60e3, 140e3, xtol=1.0)
MAS = np.array([20e3, 34.038e3, 60e3, 80e3, 110e3, 130e3])
uno_s = np.array([trabajo(ma)[0] for ma in MAS])
lam = np.log(np.abs(uno_s))
print(f"deadbeat m_a = m_2 = {ma_db/1e3:.3f} kA/s")
for ma, us, lm in zip(MAS, uno_s, lam):
    print(f"  m_a = {ma/1e3:7.3f} kA/s   1+s = {us:+.6f}   lambda = {lm:+.4f}"
          f"   {'reflexion' if us < 0 else 'pass-through'}")

# ================= figura =================
# El panel (a) va en TRES vinetas, una por especie. Dibujar u_k y sus tres
# imagenes en un solo diagrama superpone la flecha de pass-through sobre la
# de u_k (misma direccion) y el resultado es ilegible.
fig = plt.figure(figsize=(COLW, 3.60))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.60,
                      wspace=0.06)
pas = [fig.add_subplot(gs[0, j]) for j in range(3)]
pb = fig.add_subplot(gs[1, :])

DIR = np.array([0.52, 0.86])
DIR = DIR/np.linalg.norm(DIR)
esp = [(1.55, '#1a5276', r'(i) pass-through', r'$1+\sigma>0$'),
       (-0.90, '#c0392b', r'(ii) reflection', r'$1+\sigma<0$'),
       (0.0, '#1e6b33', r'(iii) singular', r'$1+\sigma=0$')]

for pa, (fac, col, tit, cond) in zip(pas, esp):
    pa.set_aspect('equal')
    pa.axis('off')
    pa.plot([-1.05, 1.05], [0, 0], '-', lw=0.9, color='0.40')
    # S_k u_k: imagen, gruesa y en color. Va DEBAJO, porque en el caso (ii)
    # es colineal con u_k y mas larga: dibujada encima, borra la preimagen.
    if abs(fac) < 1e-9:
        pa.plot([0], [0], 'o', ms=5.0, mfc='none', mew=1.3, color=col,
                zorder=5)
    else:
        pa.annotate('', xy=tuple(fac*0.88*DIR), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', lw=1.25, color=col,
                                    mutation_scale=7), zorder=3)
    # u_k: preimagen, fina y discontinua, POR ENCIMA
    pa.annotate('', xy=tuple(0.88*DIR), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', lw=0.75, color='0.25',
                                ls=(0, (2.0, 1.3)), mutation_scale=5),
                zorder=7)
    pa.plot([0], [0], 'o', ms=1.7, color='0.15', zorder=6)
    pa.set_xlim(-1.12, 1.12)
    pa.set_ylim(-1.05, 1.62)
    pa.text(0.0, 1.60, tit, fontsize=6.0, color=col, ha='center', va='top')
    pa.text(0.0, 1.34, cond, fontsize=6.0, color=col, ha='center', va='top')

# rotulos comunes, solo en la primera vineta
pas[0].text(-1.05, -0.16, r'$\ker \bm{n}_k^{T}$', fontsize=6.2, ha='left',
            va='top', color='0.35')
pas[0].text(*(0.88*DIR + np.array([0.07, 0.0])), r'$\bm{u}_k$', fontsize=6.6,
            ha='left', va='center', color='0.40')
pas[1].text(*(-0.90*0.88*DIR + np.array([-0.07, 0.0])), r'$S_k\bm{u}_k$',
            fontsize=6.6, ha='right', va='center', color='#c0392b')
fig.text(0.5, 0.985,
         r'(a) $S_k=I+\bm{u}_k\bm{n}_k^{T}$: hyperplane fixed, one direction scaled '
         r'by $1+\sigma$', fontsize=7, ha='center', va='top')

# ---------- (b) la recta de rapidez: dos hojas pegadas ----------
def xi(l):                      # compactifica R -> (-1, 1)
    return l/(1.0 + np.abs(l))


LMAX = 5.0
pb.axis('off')
X0, X1 = -1.0, xi(LMAX)
YS = 0.62
# hojas
for sgn, col, etq in ((+1, '#1a5276', r'$\det S_k>0$,\ \ $W_{S_k}\rev{W}_{S_k}=+1$'),
                      (-1, '#c0392b', r'$\det S_k<0$,\ \ $W_{S_k}\rev{W}_{S_k}=-1$')):
    xs = np.linspace(-0.985, X1, 300)
    ys = sgn*YS*np.tanh((xs + 0.985)/0.16)
    pb.plot(xs, ys, '-', lw=1.15, color=col)
    pb.annotate('', xy=(X1+0.055, sgn*YS), xytext=(X1, sgn*YS),
                arrowprops=dict(arrowstyle='-|>', lw=1.15, color=col,
                                mutation_scale=7))
    # Separar el rotulo de su hoja: pegado a la linea, la tilde de \widetilde
    # queda tapada por el propio trazo de la hoja inferior.
    # ambas etiquetas por encima de su hoja: la inferior, en la banda libre
    # entre el eje lambda y la hoja; debajo chocaria con 'increasing m_a'.
    pb.text(X1+0.075, (YS if sgn > 0 else -YS) + 0.07,
            etq.replace(r'\rev{W}', r'\widetilde{W}'),
            fontsize=5.9, color=col, ha='right', va='bottom')
# Bandas libres de este panel: y in [0.10,0.45] y y in [-0.45,-0.10] para
# x>-0.6 (entre el eje lambda y las hojas), y todo y<-0.72 (bajo la hoja
# inferior). Cada rotulo va a una de ellas y a ninguna otra.
# punto de pegado, rotulado BAJO las dos hojas con flecha
pb.plot([-1.0], [0.0], 'o', ms=4.6, color='#1e6b33', zorder=6)
pb.annotate(r'singular: DCM, deadbeat',
            xy=(-1.0, -0.03), xytext=(-1.05, -1.02),
            fontsize=6.0, color='#1e6b33', ha='left', va='center',
            arrowprops=dict(arrowstyle='->', lw=0.5, color='#1e6b33',
                            shrinkA=1, shrinkB=2))
# extremo grazing, en la banda superior derecha
pb.text(X1, 0.24, r'grazing: $\lambda\to+\infty$', fontsize=6.2,
        ha='right', va='center', color='0.25')
# escala de lambda: pocos ticks, porque la compactificacion agolpa los
# grandes |lambda| (-3 y -2 caen a 0.08 uno del otro). Etiquetas DEBAJO.
for lv in (-3, -1, 0, 1, 3):
    x = xi(lv)
    pb.plot([x, x], [-0.05, 0.05], '-', lw=0.5, color='0.55')
    pb.text(x, -0.075, f'${lv}$', fontsize=5.6, ha='center', va='top',
            color='0.40')
pb.plot([xi(-3.7), xi(3.7)], [0, 0], '-', lw=0.5, color='0.55')
pb.text(xi(-3.7)-0.02, 0.0, r'$\lambda$', fontsize=6.6, ha='right',
        va='center', color='0.35')

# travesia de la Tabla II
xr = xi(lam[uno_s < 0])
xp = xi(lam[uno_s > 0])
pb.plot(xr, np.full(len(xr), -YS), 'o', ms=2.9, color='#c0392b', zorder=5)
pb.plot(xp, np.full(len(xp), +YS), 'o', ms=2.9, color='#1a5276', zorder=5)
pb.annotate('', xy=(xr.min()-0.03, -YS), xytext=(xr.max()+0.03, -YS),
            arrowprops=dict(arrowstyle='-|>', lw=0.6, color='#c0392b',
                            mutation_scale=6, alpha=0.75))
pb.annotate('', xy=(xp.max()+0.03, +YS), xytext=(xp.min()-0.03, +YS),
            arrowprops=dict(arrowstyle='-|>', lw=0.6, color='#1a5276',
                            mutation_scale=6, alpha=0.75))
pb.text(0.5*(xr.min()+xr.max()), -YS-0.10, r'increasing $m_a$', fontsize=5.8,
        color='#c0392b', ha='center', va='top')
pb.set_xlim(X0-0.12, X1+0.12)
pb.set_ylim(-1.20, 1.02)
pb.set_title(r'(b) the rapidity line is two sheets, glued at $1+\sigma=0$',
             fontsize=7, pad=1)

fig.tight_layout(pad=0.22)
sol = comprueba_solapes(fig); assert not sol, 'hay textos solapados en la Fig. 1: %s' % sol
out = pathlib.Path(__file__).resolve().parent / 'geometria_saltacion_rapidez.pdf'
fig.savefig(out, bbox_inches='tight')
print(f"figura -> {out}")
