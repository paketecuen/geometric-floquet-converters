"""Exp. 2 — Fronteras de condicionamiento de las tres degeneraciones.

Exp. 1 se limita al regimen generico a proposito. Aqui se ataca justo lo que
alli se excluyo, y la pregunta no es "¿es cierta la teoria?" (lo es, por la
factorizacion) sino "¿hasta donde puede el coma flotante verificarla, y por
que ruta?". Tres barridos, uno por degeneracion de la Prop. 1:

  (A) CLAMPING, 1+sigma -> 0.  det S -> 0, lambda -> -infinito. El versor sale
      del grupo en el limite. Se mide donde el determinante de la monodromia
      ENSAMBLADA deja de resolver el signo, frente al determinante exacto de
      la factorizacion.

  (B) PARABOLICO, sigma -> 0.  Aqui det S -> 1: la MATRIZ esta perfectamente
      condicionada. Lo que degenera es la forma cerrada (ii), porque
      Bhat = B/sigma explota. La pregunta practica: ¿cuando hay que cambiar a
      la forma (i), W = 1 + B/2? Se comparan las dos ramas contra la matriz.

  (C) GRAZING, hdot^- -> 0.  Entonces u = (f+ - f-)/hdot^- -> infinito, luego
      sigma -> infinito y lambda -> +infinito. Se mide el crecimiento del
      error del versor con la rapidez.

Salida: resultados/exp2.json + resumen impreso.
"""
import json, pathlib, time
import numpy as np
from scipy.linalg import expm
import comun as C

SEMILLA = 20260727
RES = pathlib.Path(__file__).resolve().parent / 'resultados'
RES.mkdir(exist_ok=True)
REP = 8                       # instancias aleatorias por punto del barrido


def det_analitico(sigmas, flujos):
    tr = sum(float(np.trace(A))*tau for A, tau in flujos)
    p = 1.0
    for s in sigmas:
        p *= (1.0 + s)
    return np.exp(tr)*p


def caso(rng, n, sigmas, esc_A=0.35, tau_max=0.5):
    M, salts, flujos = C.monodromia(rng, n, sigmas, esc_A, tau_max)
    d_ana = det_analitico(sigmas, flujos)
    sgn, logabs = np.linalg.slogdet(M)
    return M, salts, flujos, d_ana, float(sgn), float(np.linalg.cond(M))


# =============== (A) clamping: 1+sigma -> 0 ===============
def barrido_clamping(n=4, K=3):
    print('=' * 72)
    print('(A) CLAMPING   1+sigma -> 0   (n=4, 3 eventos, uno degenerando)')
    print('=' * 72)
    print(f"{'|1+sigma|':>11} {'signo OK':>9} {'cond(M)':>10} "
          f"{'|det| exacto':>13} {'err versor':>11} {'|W~W|-1':>10}")
    filas = []
    for e in [10.0**(-k) for k in range(0, 15)]:
        rng = np.random.default_rng(SEMILLA + int(round(-np.log10(e))))
        ok, conds, dets, errs, wws = 0, [], [], [], []
        for _ in range(REP):
            sig = [-1.0 - e] + [rng.uniform(-4.0, -1.2),
                                rng.uniform(-0.8, 4.0)]
            M, salts, flujos, d_ana, sgn, cond = caso(rng, n, sig)
            ok += int(sgn == np.sign(d_ana))
            conds.append(cond); dets.append(abs(d_ana))
            if e >= 1e-9:                    # el algebra solo hasta donde vive
                G = C.algebra(n)
                W = C.versor_monodromia(salts, flujos, G)
                if W is not None:
                    A = C.accion(W, G)
                    errs.append(float(np.max(np.abs(A - M))
                                      / max(1.0, np.max(np.abs(M)))))
                    wws.append(abs(abs(float((W*~W)(0))) - 1.0))
        f = dict(eps=e, signo_ok=ok, de=REP, cond=float(np.median(conds)),
                 det=float(np.median(dets)),
                 err=float(np.median(errs)) if errs else None,
                 ww=float(np.median(wws)) if wws else None)
        filas.append(f)
        s_err = '%11.2e' % f['err'] if f['err'] is not None else '%11s' % '-'
        s_ww = '%10.2e' % f['ww'] if f['ww'] is not None else '%10s' % '-'
        print(f"{e:11.0e} {ok:>6}/{REP} {f['cond']:10.2e} {f['det']:13.2e} "
              f"{s_err} {s_ww}")
    return filas


# =============== (B) parabolico: sigma -> 0, rama (i) vs (ii) ===============
def barrido_parabolico(n=3):
    print()
    print('=' * 72)
    print('(B) PARABOLICO   sigma -> 0   forma (i) 1+B/2  frente a (ii) exp')
    print('=' * 72)
    print(f"{'|sigma|':>10} {'err rama (i)':>13} {'(ii) con log':>14} "
          f"{'(ii) con log1p':>16}")
    filas = []
    for e in [10.0**(-k) for k in range(0, 16)]:
        rng = np.random.default_rng(SEMILLA + 500 + int(round(-np.log10(e))))
        e1, e2, e3 = [], [], []
        for _ in range(REP):
            S, u, nv = C.salt_desde_sigma(rng, n, e)
            G = C.algebra(n)
            U = sum(float(u[i])*G['w'][i] for i in range(n))
            V = sum(float(nv[j])*G['v'][j] for j in range(n))
            B = 2*(U ^ V)
            s = float(nv @ u)
            # rama (i): translator exacto
            W1 = G['uno'] + B/2
            e1.append(float(np.max(np.abs(C.accion(W1, G) - S))))
            # rama (ii): exp(ln(1+s) Bhat/2), que divide por s. Se comparan
            # las dos formas de evaluar el logaritmo, porque la cancelacion
            # al formar 1+s es justo lo que decide si hace falta ramificar.
            if abs(s) > 0:
                W2 = C.mvexp(np.log(1.0 + s)*(B/s)/2, G['uno'])
                e2.append(float(np.max(np.abs(C.accion(W2, G) - S))))
                W3 = C.mvexp(np.log1p(s)*(B/s)/2, G['uno'])
                e3.append(float(np.max(np.abs(C.accion(W3, G) - S))))
        m1 = float(np.median(e1)); m2 = float(np.median(e2)) if e2 else np.nan
        m3 = float(np.median(e3)) if e3 else np.nan
        filas.append(dict(eps=e, err_i=m1, err_ii=m2, err_ii_log1p=m3))
        print(f"{e:10.0e} {m1:13.2e} {m2:14.2e} {m3:16.2e}")
    return filas


# =============== (C) grazing: sigma -> infinito ===============
def barrido_grazing(n=3):
    print()
    print('=' * 72)
    print('(C) GRAZING   hdot^- -> 0, luego sigma -> infinito')
    print('=' * 72)
    print(f"{'sigma':>11} {'rapidez':>9} {'err versor':>11} {'|W~W|-1':>10}")
    filas = []
    for s in [-(10.0**k) for k in range(1, 13)]:
        rng = np.random.default_rng(SEMILLA + 900 + int(round(np.log10(-s))))
        errs, wws = [], []
        for _ in range(REP):
            S, u, nv = C.salt_desde_sigma(rng, n, s)
            G = C.algebra(n)
            W, esp = C.versor_salt(u, nv, G)
            if W is None:
                continue
            errs.append(float(np.max(np.abs(C.accion(W, G) - S))
                              / max(1.0, np.max(np.abs(S)))))
            wws.append(abs(abs(float((W*~W)(0))) - 1.0))
        lam = float(np.log(abs(1.0 + s)))
        filas.append(dict(sigma=s, lam=lam, err=float(np.median(errs)),
                          ww=float(np.median(wws))))
        print(f"{s:11.0e} {lam:9.3f} {np.median(errs):11.2e} "
              f"{np.median(wws):10.2e}")
    return filas


if __name__ == '__main__':
    t0 = time.time()
    A = barrido_clamping()
    B = barrido_parabolico()
    Cg = barrido_grazing()

    # fronteras
    fr_signo = next((f['eps'] for f in A if f['signo_ok'] < f['de']), None)
    fr_vers = next((f['eps'] for f in A
                    if f['err'] is not None and f['err'] > 1e-6), None)
    cruce = next((f['eps'] for f in B
                  if np.isfinite(f['err_ii']) and f['err_ii'] > f['err_i']*10),
                 None)
    cruce1p = next((f['eps'] for f in B
                    if np.isfinite(f['err_ii_log1p'])
                    and f['err_ii_log1p'] > f['err_i']*10), None)
    print()
    print('=' * 72)
    print('FRONTERAS')
    print('=' * 72)
    print(f"  clamping: el signo del producto ensamblado empieza a fallar en "
          f"|1+sigma| ~ {fr_signo if fr_signo else '<1e-14'}")
    print(f"  clamping: el versor pasa de 1e-6 de error en "
          f"|1+sigma| ~ {fr_vers if fr_vers else 'nunca en el rango medido'}")
    print(f"  parabolico: con log(1+s), la rama (ii) pierde 10x frente a la "
          f"(i) en |sigma| ~ {cruce if cruce else 'no cruza en el rango'}")
    print(f"  parabolico: con log1p(s), en "
          f"|sigma| ~ {cruce1p if cruce1p else 'NO CRUZA: no hace falta ramificar'}")
    print(f"  grazing: error del versor a sigma=-1e12: "
          f"{Cg[-1]['err']:.2e}  (rapidez {Cg[-1]['lam']:.1f})")
    print(f"\n  {round(time.time()-t0,1)} s")
    (RES / 'exp2.json').write_text(json.dumps(
        dict(semilla=SEMILLA, repeticiones=REP, clamping=A, parabolico=B,
             grazing=Cg,
             frontera_signo=fr_signo, frontera_versor=fr_vers,
             cruce_parabolico=cruce, cruce_parabolico_log1p=cruce1p),
        indent=1))
