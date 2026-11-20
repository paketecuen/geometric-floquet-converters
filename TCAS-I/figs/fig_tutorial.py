"""Fig. tutorial — un periodo del buck con control de corriente de pico.
(a) corriente de bobina de la orbita periodica y de una trayectoria perturbada:
    el instante de apagado se adelanta y las dos trayectorias difieren al final
    del periodo. Eso, linealizado, es la matriz de saltacion.
(b) la funcion de conmutacion h = i_L - (I_ref - m_a t) con dos pendientes de
    rampa: reflexion (m_a < m_2: toca el umbral y vuelve) y paso (m_a > m_2).
Modelo: la plataforma R = 3 ohm de la Sec. V-A (buck_realista.py)."""
import pathlib
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq
from estilo_ieee import plt, COLW, comprueba_solapes

Vin, L, RL, C, ESR, R, Ron, Vd, Rd, T, Iref = 12.0, 100e-6, 0.060, 100e-6, 0.025, 3.0, 0.040, 0.45, 0.020, 1e-5, 3.35
al = R/(R + ESR)
matA = lambda Rsw: np.array([[-(Rsw + RL + al*ESR)/L, -al/L], [al/C, -al/(R*C)]])
A_on, b_on, A_off, b_off = matA(Ron), np.array([Vin/L, 0.0]), matA(Rd), np.array([-Vd/L, 0.0])


def flw(A, b, x, t):
    Z = np.zeros((3, 3)); Z[:2, :2] = A; Z[:2, 2] = b
    E = expm(Z*t); return E[:2, :2]@x + E[:2, 2]


def orbita(ma):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton); x1 = flw(A_off, b_off, xc, T - ton)
        return [xc[0] - (Iref - ma*ton), x1[0] - x0[0], x1[1] - x0[1]]
    z, _, ier, msg = fsolve(res, [Iref - 0.35, 8.6, 0.8*T], full_output=True); assert ier == 1, msg
    return z[:2], z[2]


def trayectoria(x0, ma, nt=400):
    """(t, i_L, h) a lo largo de un periodo, apagando cuando h cruza cero."""
    h = lambda t: flw(A_on, b_on, x0, t)[0] - (Iref - ma*t)
    ton = brentq(h, 1e-9, T) if h(T) > 0 else T
    t1 = np.linspace(0, ton, nt); t2 = np.linspace(ton, T, nt)
    xc = flw(A_on, b_on, x0, ton)
    i1 = np.array([flw(A_on, b_on, x0, t)[0] for t in t1])
    i2 = np.array([flw(A_off, b_off, xc, t - ton)[0] for t in t2])
    t = np.concatenate([t1, t2]); i = np.concatenate([i1, i2])
    return t, i, i - (Iref - ma*t), ton


AZUL, ROJO, VERDE, GRIS = '#1a5276', '#c0392b', '#1e6b33', '0.35'
fig = plt.figure(figsize=(COLW, 4.35))
gs = fig.add_gridspec(3, 1, height_ratios=[0.92, 1.0, 1.0], hspace=0.62)
e = fig.add_subplot(gs[0]); a = fig.add_subplot(gs[1]); b = fig.add_subplot(gs[2])
us = 1e6


def esquema(ax):
    """(a) etapa de potencia y controlador de corriente de pico con latch."""
    c, lw = '0.2', 0.9
    ax.set_xlim(0, 16); ax.set_ylim(0.55, 6.0); ax.set_aspect('equal'); ax.axis('off')
    # fuente
    ax.add_patch(plt.Circle((1.0, 3.0), 0.5, fill=False, lw=lw, color=c))
    ax.text(1.0, 3.0, r'$V_{\mathrm{in}}$', fontsize=5.8, ha='center', va='center')
    ax.plot([1.0, 1.0], [3.5, 4.6], '-', color=c, lw=lw); ax.plot([1.0, 1.0], [2.5, 1.2], '-', color=c, lw=lw)
    # interruptor (abierto) y nodo
    ax.plot([1.0, 2.4], [4.6, 4.6], '-', color=c, lw=lw)
    ax.plot([2.4, 3.3], [4.6, 5.05], '-', color=c, lw=lw)
    ax.plot([2.4, 3.5], [4.6, 4.6], 'o', ms=1.8, color=c)
    ax.plot([3.5, 4.6], [4.6, 4.6], '-', color=c, lw=lw)
    ax.text(2.95, 5.35, 'switch', fontsize=5.8, ha='center', va='bottom', color=c)
    # diodo (catodo arriba)
    ax.plot([4.6, 4.6], [4.6, 3.5], '-', color=c, lw=lw); ax.plot([4.6, 4.6], [2.6, 1.2], '-', color=c, lw=lw)
    ax.add_patch(plt.Polygon([[4.25, 2.6], [4.95, 2.6], [4.6, 3.5]], fill=False, lw=lw, color=c))
    ax.plot([4.25, 4.95], [3.5, 3.5], '-', color=c, lw=lw)
    ax.text(4.1, 3.05, 'diode', fontsize=5.8, ha='right', va='center', color=c)
    # bobina
    ax.plot([4.6, 5.1], [4.6, 4.6], '-', color=c, lw=lw)
    th = np.linspace(0, np.pi, 30)
    for k in range(4):
        ax.plot(5.1 + 0.45*k + 0.225 + 0.225*np.cos(th[::-1]), 4.6 + 0.22*np.sin(th), '-', color=c, lw=lw)
    ax.plot([6.9, 8.4], [4.6, 4.6], '-', color=c, lw=lw)
    ax.text(6.0, 5.1, r'$L$', fontsize=5.8, ha='center', va='bottom')
    ax.annotate('', xy=(6.75, 4.25), xytext=(5.9, 4.25), arrowprops=dict(arrowstyle='->', lw=0.6, color=c))
    ax.text(6.3, 3.95, r'$i_L$', fontsize=5.8, ha='center', va='top')
    # condensador
    ax.plot([7.4, 7.4], [4.6, 3.15], '-', color=c, lw=lw); ax.plot([7.05, 7.75], [3.15, 3.15], '-', color=c, lw=lw)
    ax.plot([7.05, 7.75], [2.85, 2.85], '-', color=c, lw=lw); ax.plot([7.4, 7.4], [2.85, 1.2], '-', color=c, lw=lw)
    ax.text(7.4, 2.45, r'$C$', fontsize=5.8, ha='center', va='top')
    ax.text(7.0, 3.0, r'$v_C$', fontsize=5.8, ha='right', va='center')
    # carga
    ax.plot([8.4, 8.4], [4.6, 3.6], '-', color=c, lw=lw)
    ax.add_patch(plt.Rectangle((8.15, 2.4), 0.5, 1.2, fill=False, lw=lw, color=c))
    ax.plot([8.4, 8.4], [2.4, 1.2], '-', color=c, lw=lw)
    ax.text(8.85, 3.0, r'$R$', fontsize=5.8, ha='left', va='center')
    ax.plot([1.0, 8.4], [1.2, 1.2], '-', color=c, lw=lw)                     # masa
    # ---- controlador ----
    # comparador
    ax.add_patch(plt.Polygon([[10.6, 1.6], [10.6, 3.6], [12.1, 2.6]], fill=False, lw=lw, color=c))
    ax.text(10.75, 3.1, r'$+$', fontsize=5.8, ha='left', va='center'); ax.text(10.75, 2.1, r'$-$', fontsize=5.8, ha='left', va='center')
    ax.plot([9.3, 10.6], [3.1, 3.1], '-', color=c, lw=lw); ax.text(9.9, 3.3, r'$i_L$', fontsize=5.8, ha='center', va='bottom')
    ax.plot([9.3, 10.6], [2.1, 2.1], '-', color=c, lw=lw)
    ax.text(10.0, 1.1, r'ramp $I_{\mathrm{ref}}-m_at$', fontsize=5.8, ha='center', va='top')
    # latch RS
    ax.add_patch(plt.Rectangle((12.8, 1.6), 1.9, 2.0, fill=False, lw=lw, color=c))
    ax.plot([12.1, 12.8], [2.6, 2.6], '-', color=c, lw=lw)
    ax.text(12.95, 2.6, 'R', fontsize=5.8, ha='left', va='center'); ax.text(12.95, 3.25, 'S', fontsize=5.8, ha='left', va='center')
    ax.text(14.55, 2.95, 'Q', fontsize=5.8, ha='right', va='center'); ax.text(13.75, 1.3, 'latch', fontsize=5.8, ha='center', va='top', color=c)
    ax.plot([12.1, 12.8], [3.25, 3.25], '-', color=c, lw=lw); ax.text(12.45, 3.95, 'clock', fontsize=5.8, ha='center', va='bottom', color=c)
    ax.plot([11.3, 11.6, 11.6, 11.9, 11.9], [3.3, 3.3, 3.55, 3.55, 3.3], '-', color=c, lw=0.7)   # glifo de reloj
    # salida Q hacia el interruptor (discontinua)
    ax.plot([14.7, 15.3, 15.3, 2.95, 2.95], [2.95, 2.95, 5.7, 5.7, 5.15], '--', color=c, lw=0.7)
    ax.annotate('', xy=(2.95, 5.08), xytext=(2.95, 5.4), arrowprops=dict(arrowstyle='->', lw=0.6, color=c))
    ax.set_title('(a) power stage and peak-current controller', fontsize=7, pad=2)


esquema(e)

# ---------------- (a) orbita y trayectoria perturbada, m_a = 34 kA/s ----------------
ma = 34e3
x0, ton = orbita(ma)
t, i, h, _ = trayectoria(x0, ma)
di = 0.09
tp, ip, hp, tonp = trayectoria(x0 + np.array([di, 0.0]), ma)
a.plot(t*us, i, '-', color=AZUL, lw=1.1)
a.plot(tp*us, ip, '--', color=ROJO, lw=0.9)
a.plot(t*us, Iref - ma*t, ':', color=GRIS, lw=0.8)
a.axvline(ton*us, color=AZUL, lw=0.5, ls=':'); a.axvline(tonp*us, color=ROJO, lw=0.5, ls=':')
a.set_xlim(0, T*us); a.set_ylim(i.min() - 0.08, Iref + 0.09)
a.set_xlabel(r'time within the period $[\mu\mathrm{s}]$', labelpad=1.5)
a.set_ylabel(r'$i_L$ [A]', labelpad=1)
a.set_title(r'(b) one period: the switching instant moves with the state', fontsize=7, pad=2)
a.text(1.2, Iref - ma*1.2e-6 + 0.02, r'ramp $I_{\mathrm{ref}}-m_at$', fontsize=6.3, color=GRIS, ha='left', va='bottom')
a.text(0.3, x0[0] + di + 0.055, r'perturbed, $+\delta i$', fontsize=6.3, color=ROJO, ha='left', va='bottom')
a.text(0.3, x0[0] - 0.02, r'periodic orbit', fontsize=6.3, color=AZUL, ha='left', va='top')
a.annotate('', xy=(ton*us, i.min() + 0.02), xytext=(tonp*us, i.min() + 0.02),
           arrowprops=dict(arrowstyle='<->', lw=0.6, color='0.3', shrinkA=0, shrinkB=0))
a.text(0.5*(ton + tonp)*us, i.min() + 0.035, r'$\delta t$', fontsize=6.6, color='0.3', ha='center', va='bottom')
a.text(4.3, flw(A_on, b_on, x0, 4.3e-6)[0] - 0.035, r'slope $m_1$', fontsize=6.3, color=AZUL, ha='left', va='top')
a.text(ton*us + 0.9, Iref - 0.05, r'slope $-m_2$', fontsize=6.3, color=AZUL, ha='left', va='center')
a.annotate('', xy=(T*us - 0.15, ip[-1] - 0.03), xytext=(T*us - 0.15, i[-1] + 0.03),
           arrowprops=dict(arrowstyle='<->', lw=0.6, color='0.3', shrinkA=0, shrinkB=0))
a.text(T*us - 0.3, i[-1] + 0.08, r'$\delta\bm{x}(T)$', fontsize=6.6, color='0.3', ha='right', va='center')

# ---------------- (b) funcion de conmutacion: reflexion vs paso ----------------
for ma_b, col, etq in ((34e3, ROJO, r'$m_a<m_2$: reflection, $\det S<0$'),
                       (110e3, AZUL, r'$m_a>m_2$: pass-through, $\det S>0$')):
    x0b, tonb = orbita(ma_b)
    tb, ib, hb, _ = trayectoria(x0b, ma_b)
    b.plot(tb*us, hb, '-', color=col, lw=1.1)
    b.plot([tonb*us], [0.0], 'o', ms=3.0, mfc='w', mec=col, mew=0.9, zorder=5)
    b.text(9.75, -0.55 if col == AZUL else -0.80, etq, fontsize=6.3, color=col, ha='right', va='center')
b.axhline(0.0, color=GRIS, lw=0.5)
b.set_xlim(0, T*us); b.set_ylim(-1.05, 0.25)
b.set_xlabel(r'time within the period $[\mu\mathrm{s}]$', labelpad=1.5)
b.set_ylabel(r'$h=i_L-(I_{\mathrm{ref}}-m_at)$ [A]', labelpad=1)
b.set_title(r'(c) the switching function: bounce or cross', fontsize=7, pad=2)
b.text(2.0, 0.12, r'$\dot h^{-}=m_1+m_a>0$', fontsize=6.3, color=GRIS, ha='left', va='center')
b.text(8.3, -0.30, r'$\dot h^{+}=-(m_2-m_a)$', fontsize=6.3, color=GRIS, ha='left', va='center')

fig.align_ylabels((a, b))
fig.tight_layout(pad=0.25, h_pad=0.9)
sol = comprueba_solapes(fig)
assert not sol, 'hay textos solapados en la figura tutorial: %s' % sol
out = pathlib.Path(__file__).resolve().parent / 'tutorial_saltacion.pdf'
fig.savefig(out, bbox_inches='tight')
print('figura -> %s' % out)
