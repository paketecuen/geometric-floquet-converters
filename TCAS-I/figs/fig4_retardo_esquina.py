"""Fig. 5 — Las dos fronteras de la regla de paridad.

(a) RETARDO DIGITAL. El muestreador es una saltacion mas, y la ganancia del
    lazo elige su especie: 1+sigma_Sigma cruza cero en K = 0. A la izquierda es
    hiperbolico y no fuerza nada; en K = 0 es un proyector exacto (la medida ya
    no llega al mando); a la derecha es de reflexion y la Prop. 2 fuerza un
    multiplicador real negativo desde ganancia infinitesimal. Superpuesto,
    max|mu| muestra que la orbita no pierde la estabilidad ahi sino mucho mas
    tarde, y por un par complejo: el canal esta ABIERTO, no TOMADO.

(b) ESQUINA SIMULTANEA. Dos fases acopladas y desiguales cuyos apagados
    colisionan en epsilon = 0. det M converge a una candidata por la izquierda
    y a la otra por la derecha: tiene un salto finito aunque todos los eventos
    sean transversales. Es la limitacion genuina del trabajo.

Los numeros NO se recalculan aqui: se leen de validacion/resultados/*.json,
que es donde la campana los valida.
"""
import json
import pathlib
import numpy as np
from estilo_ieee import plt, COLW, comprueba_solapes

RAIZ = pathlib.Path(__file__).resolve().parent.parent
RES = RAIZ / 'validacion' / 'resultados'
d5 = json.load(open(RES / 'exp5.json'))
d6 = json.load(open(RES / 'exp6.json'))

AZUL, ROJO, VERDE, MORADO = '#1a5276', '#c0392b', '#1e6b33', '#7d3c98'

# Desde la revision R1 (20-11-2026) son dos figuras: (a) va en el articulo
# (Fig. 7) y (b) en el suplemento (Fig. S1).
fig, a1 = plt.subplots(figsize=(COLW, 1.85))
fig2, a2 = plt.subplots(figsize=(COLW, 1.80))

# ============================ (a) retardo digital ==========================
fino = [r for r in d5['fino'] if r['K'] >= -0.10]        # K < -0.08: orbita ya inestable
K = np.array([r['K'] for r in fino])
dS = np.array([r['detSig'] for r in fino])*1e3            # milesimas
mx = np.array([r['mumax'] if r['K'] >= 0.0 else np.nan for r in fino])  # solo K >= 0
mn = np.array([np.nan if r['mu_neg'] is None else r['mu_neg']*1e3
               for r in fino])                            # multiplicador forzado
Kc = d5['K_critico']

a1.axhline(0.0, color='0.55', lw=0.5, ls='-')
a1.axvspan(K.min(), 0.0, color='#eaf2f8', zorder=0)
a1.axvspan(0.0, K.max(), color='#fdedec', zorder=0)
a1.plot(K, dS, '-', color=AZUL, lw=1.1)
a1.plot(K, mn, ':', color=ROJO, lw=1.0)
a1.text(0.30, -9.0, r'forced $\mu_-$ $[\times10^{-3}]$', fontsize=6.3,
        color=ROJO, ha='center', va='center')
a1.plot([0.0], [0.0], 'o', ms=3.2, mfc='w', mec=VERDE, mew=0.9, zorder=5)
a1.set_xlim(K.min(), K.max())
a1.set_ylim(-31, 16)
a1.set_ylabel(r'$\det\Sigma$ \ $[\times 10^{-3}]$', labelpad=1)
a1.set_xlabel(r'digital loop gain $K$', labelpad=1.5)

# especies: la franja alta esta libre porque det Sigma decrece
a1.text(-0.096, 12.2, 'hyperbolic', fontsize=6.3, color=AZUL, ha='left',
        va='center')
a1.text(-0.096, 8.2, r'$N_r=0$', fontsize=6.3, color='0.35', ha='left',
        va='center')
a1.text(0.47, 9.6, 'reflection', fontsize=6.3, color=ROJO, ha='right',
        va='center')
a1.text(0.47, 5.6, r'$N_r=1$: flip channel forced', fontsize=6.3,
        color='0.35', ha='right', va='center')
a1.annotate(r'projector, $\det\Sigma=0$', xy=(0.0, 0.0),
            xytext=(-0.096, -12.0), fontsize=6.3, color=VERDE,
            ha='left', va='center',
            arrowprops=dict(arrowstyle='->', lw=0.5, color=VERDE,
                            shrinkA=1, shrinkB=2))

# max|mu| en el eje derecho: la estabilidad se pierde donde NO manda la regla
b1 = a1.twinx()
b1.xaxis.set_visible(False)          # twinx redibuja los ticks de x encima
b1.plot(K, mx, '--', color=MORADO, lw=0.9)
b1.axhline(1.0, color=MORADO, lw=0.5, ls=':')
b1.set_ylim(0.965, 1.075)
b1.set_ylabel(r'$\max_k|\mu_k|$', color=MORADO, labelpad=1.5)
b1.tick_params(axis='y', colors=MORADO)
b1.spines['right'].set_color(MORADO)
b1.spines['right'].set_linewidth(0.6)
if Kc:
    b1.plot([Kc], [1.0], 's', ms=2.8, color=MORADO, zorder=5)
    b1.annotate(r'Neimark--Sacker, $K^{*}=%.3f$' % Kc, xy=(Kc, 1.0),
                xytext=(0.435, 1.030), fontsize=6.3, color=MORADO,
                ha='right', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.5, color=MORADO,
                                shrinkA=1, shrinkB=2))
# b1 se dibuja DESPUES de a1, asi que su curva queda encima de las bandas de
# color; lo unico que hay que quitar es su fondo opaco. (Subir el zorder de a1
# en su lugar hace que sus axvspan tapen el eje gemelo entero.)
b1.patch.set_visible(False)

# ============================ (b) esquina simultanea =======================
lat = d6['lateral']
e = np.array([r['eps'] for r in lat['fino']])
dm = np.array([r['det'] for r in lat['fino']])
izq, der = e < 0, e > 0
d_ab, d_ba = lat['det_ab'], lat['det_ba']

a2.axhline(d_ba, color='0.55', lw=0.5, ls='--')
a2.axhline(d_ab, color='0.55', lw=0.5, ls='--')
a2.axvline(0.0, color='0.35', lw=0.6, ls='-')
a2.plot(e[izq], dm[izq], '-', color=VERDE, lw=1.1)
a2.plot(e[der], dm[der], '-', color=ROJO, lw=1.1)
a2.plot([e[izq].max()], [dm[izq][np.argmax(e[izq])]], 'o', ms=3.0,
        mfc='w', mec=VERDE, mew=0.9, zorder=5)
a2.plot([e[der].min()], [dm[der][np.argmin(e[der])]], 'o', ms=3.0,
        mfc='w', mec=ROJO, mew=0.9, zorder=5)
a2.set_xscale('symlog', linthresh=2e-5, linscale=0.35)
a2.set_xlim(-4e-3, 4e-3)
a2.set_xticks([-1e-3, -1e-4, 0.0, 1e-4, 1e-3])
a2.set_xticklabels([r'$-10^{-3}$', r'$-10^{-4}$', r'$0$',
                    r'$10^{-4}$', r'$10^{-3}$'])
a2.set_ylim(2.0, 26.0)
a2.set_xlabel(r'phase mismatch $\epsilon$ '
              r'(events coincide at $\epsilon=0$)', labelpad=1.5)
a2.set_ylabel(r'$\det M$', labelpad=1)

# los dos limites laterales, rotulados en la mitad libre de cada rama
a2.text(-2.6e-3, d_ba - 2.6, r'$\det M_{ba}=%.3f$' % d_ba, fontsize=6.3,
        color=VERDE, ha='left', va='center')
a2.text(2.6e-3, d_ab + 2.6, r'$\det M_{ab}=%.3f$' % d_ab, fontsize=6.3,
        color=ROJO, ha='right', va='center')
a2.annotate('', xy=(0.0, d_ba), xytext=(0.0, d_ab),
            arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.30',
                            shrinkA=0, shrinkB=0))
a2.text(3.0e-5, 0.5*(d_ab + d_ba), 'finite jump', fontsize=6.3, color='0.30',
        ha='left', va='center')
a2.text(-3.6e-3, 24.0, r'phase 2 switches first', fontsize=6.3, color=VERDE,
        ha='left', va='top')
a2.text(3.6e-3, 3.0, r'phase 1 switches first', fontsize=6.3, color=ROJO,
        ha='right', va='bottom')

AQUI = pathlib.Path(__file__).resolve().parent
for f, nombre in ((fig, 'retardo_digital.pdf'), (fig2, 'esquina_simultanea.pdf')):
    f.tight_layout(pad=0.25)
    sol = comprueba_solapes(f)
    assert not sol, 'hay textos solapados en %s: %s' % (nombre, sol)
    f.savefig(AQUI / nombre, bbox_inches='tight')
    print('figura -> %s' % (AQUI / nombre))
print('  (a) det Sigma de %+.4e a %+.4e, cruce de especie en K = 0'
      % (dS[0]*1e-3, dS[-1]*1e-3))
print('  (b) limites laterales: %.4f (izq) y %.4f (der), salto relativo %.3f'
      % (d_ba, d_ab, abs(d_ba - d_ab)/d_ba))
