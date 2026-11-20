"""Fig. 6 -- La mitad par de la regla de paridad (Sec. V-D).

(a) Frecuencia del batido frente al desajuste de inductancia, 1 mA por
    encima de la frontera de estabilidad: forma cerrada, monodromia y
    simulacion temporal. El batido se apaga en |Delta mu| = 2 chi.
(b) Muestras estroboscopicas de i_L1: fases iguales (toro) y desajuste del
    2 % (doblamiento de periodo).

Los numeros NO se recalculan aqui: se leen de
validacion/exp11_par_flip/resultados/ (td3_umbral.py).
"""
import json
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
RES = AQUI.parent / 'validacion' / 'exp11_par_flip' / 'resultados'
from estilo_ieee import plt, COLW, comprueba_solapes

d = json.load(open(RES / 'td3_umbral.json'))
S = np.load(RES / 'td3_series.npz')
FS = 100e3
AZUL, ROJO, VERDE = '#1a5276', '#c0392b', '#1e6b33'

r = np.array([p['r'] for p in d])
q = np.array([p['q_cf'] for p in d])
hz = lambda th: np.asarray(th)*FS/(2*np.pi)
f_lin = hz([p['th_lin'] for p in d])
f_sim = hz([p['th_sim'] for p in d])
es_bat = np.array([p['tipo'] == 'batido' for p in d])

# curva continua de la forma cerrada: kappa* ~ 1/sqrt(r), q(r) interpolado
k0 = d[0]['th_cf']
rr = np.linspace(1.0, 1.026, 600)
qq = np.interp(rr, r, q)
ff = hz(k0/np.sqrt(rr)*np.sqrt(np.clip(1-qq**2, 0, None)))
r_um = float(np.interp(1.0, q, r))

# (a) va en el articulo (Fig. 5) y (b) en el suplemento (Fig. S2)
fig, a1 = plt.subplots(figsize=(COLW, 1.80))
fig2, a2 = plt.subplots(figsize=(COLW, 1.65))

x = lambda v: (np.asarray(v)-1)*100
a1.axvspan(x(r_um), x(rr[-1]), color='#fdedec', zorder=0)
a1.axvspan(0, x(r_um), color='#eaf2f8', zorder=0)
a1.plot(x(rr), ff, '-', color=AZUL, lw=1.0, label=r'closed form, $\sqrt{\chi^{2}-(\Delta\mu)^{2}/4}$')
a1.plot(x(r[es_bat]), f_lin[es_bat], 'x', color='0.25', ms=3.4, mew=0.7, label='monodromy')
a1.plot(x(r[es_bat]), f_sim[es_bat], 'o', ms=3.0, mfc='w', mec=ROJO, mew=0.8,
        label='time-domain simulation')
a1.plot(x(r[~es_bat]), np.zeros((~es_bat).sum()), 's', ms=2.8, mfc=ROJO, mec=ROJO,
        label='simulation: period 2')
a1.set_xlim(0, x(rr[-1]))
a1.set_ylim(-2.5, 44)
a1.set_xlabel(r'inductance mismatch $L_2/L_1-1$ \ [\%]', labelpad=1.5)
a1.set_ylabel('beat frequency [Hz]', labelpad=2)
a1.text(0.55, 14.0, 'complex pair:\ntorus', fontsize=6.3, color=AZUL, ha='center', va='center',
        linespacing=1.15)
a1.text(2.07, 37.5, 'two real multipliers:\nperiod doubling', fontsize=6.3, color=ROJO, ha='center',
        va='center', linespacing=1.15)
a1.annotate(r'$\lvert\Delta\mu\rvert=2\chi$', xy=(x(r_um), 0), xytext=(1.16, 5.2),
            fontsize=6.3, ha='center', va='center',
            arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3', shrinkA=1, shrinkB=1))
a1.legend(loc='center right', bbox_to_anchor=(0.995, 0.47), fontsize=5.6, frameon=True)

n = 6000
c = np.arange(n)
y0 = (S['r1.000'][-n:, 0] - S['r1.000'][-n:, 0].mean())*1e3
y2 = (S['r1.020'][-n:, 0] - S['r1.020'][-n:, 0].mean())*1e3
a2.plot(c, y0, '.', ms=0.7, color=AZUL, rasterized=True)
a2.plot(c, y2, '.', ms=0.7, color=ROJO, rasterized=True)
a2.set_xlim(0, n)
a2.set_ylim(-62, 62)
a2.set_xlabel('switching cycle', labelpad=1.5)
a2.set_ylabel(r'$i_{L1}-\bar{\imath}_{L1}$ at the clock [mA]', labelpad=1)
a2.text(3350, 52, 'equal phases', fontsize=6.3, color=AZUL, ha='center', va='center')
a2.text(1500, 43, r'$L_2/L_1=1.02$', fontsize=6.3, color=ROJO, ha='center', va='center')

ch = []
for f, nombre in ((fig, 'par_flip_dos_fases.pdf'), (fig2, 'par_flip_series.pdf')):
    f.tight_layout(pad=0.3)
    ch += comprueba_solapes(f)
    f.savefig(AQUI / nombre)
print(f'umbral (forma cerrada): L2/L1 = {r_um:.4f};  batido maximo {ff[0]:.1f} Hz;  solapes: {len(ch)}')
