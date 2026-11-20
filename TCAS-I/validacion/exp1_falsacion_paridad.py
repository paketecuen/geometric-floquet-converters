"""Exp. 1 — Falsacion de la regla de paridad en el regimen GENERICO.

Leccion de la primera pasada: sembrar sigma a 1e-9 de la frontera de clamping
y luego pedirle el signo a det M calculado del producto ensamblado no falsa
nada, solo mide underflow. Con seis eventos casi singulares det M ~ 1e-54.
Ese estudio es legitimo, pero es OTRO experimento: esta en exp2.

Aqui se ataca el regimen generico, |1+sigma| >= 0.05, y se usa como verdad de
terreno el determinante EXACTO que da la propia factorizacion,

    det M = exp( sum_i tr(A_i) tau_i ) * prod_k (1 + sigma_k),

de modo que hay tres caminos independientes que comparar:

  (1) sign(det M) analitico            -> debe ser (-1)^{N_r} EXACTAMENTE
  (2) det M del producto matricial     -> debe coincidir con el analitico
  (3) paridad del numero de multiplicadores reales negativos
  (4) N_r impar => al menos un multiplicador real negativo

y, en el algebra, la identidad versorial accion(W) = M con W~W = sign(det M).

Se cuenta ademas el caso que el manuscrito declara PERMITIDO pero no forzado:
N_r par con dos o mas multiplicadores reales negativos. Que aparezca confirma
que el enunciado corregido no es una cautela vacia.

Salida: resultados/exp1.json. Codigo != 0 si hay contraejemplo.
"""
import json, pathlib, sys, time
import numpy as np
import comun as C

SEMILLA = 20260726
MARGEN = 0.05          # |1+sigma| >= MARGEN: regimen generico
COND_MAX = 1e10        # por encima, el signo de det M no es representable
RES = pathlib.Path(__file__).resolve().parent / 'resultados'
RES.mkdir(exist_ok=True)


def sigma_generico(rng):
    """Especies genericas, lejos de la frontera de clamping. El caso
    parabolico (sigma = 0) SI se incluye: det S = 1, esta bien condicionado."""
    r = rng.random()
    if r < 0.10:
        return 0.0                                  # parabolico exacto
    if r < 0.55:
        return rng.uniform(-6.0, -1.0 - MARGEN)     # reflexion
    return float(rng.choice([-1.0, 1.0])) * rng.uniform(MARGEN, 6.0) \
        if False else rng.uniform(-1.0 + MARGEN, 6.0)   # hiperbolico


def det_analitico(sigmas, flujos):
    tr = sum(float(np.trace(A))*tau for A, tau in flujos)
    p = 1.0
    for s in sigmas:
        p *= (1.0 + s)
    return np.exp(tr)*p


def negativos(M):
    """Real y negativo, con criterio RELATIVO a cada autovalor. Escalar la
    tolerancia con max|mu| clasifica mal los autovalores pequenos cuando el
    espectro abarca decadas, que es lo normal en una monodromia rigida."""
    mu = np.linalg.eigvals(M)
    real = np.abs(mu.imag) <= 1e-8*np.maximum(np.abs(mu), 1e-300)
    return int(np.sum(real & (mu.real < 0.0))), mu


# ======================= bloque A: matrices, n = 2..8 ======================
def bloque_A(n_ensayos=20000):
    rng = np.random.default_rng(SEMILLA)
    f_paridad, f_num, f_eig, f_impar = [], [], [], []
    permitidos, peor_num, mal_cond = 0, 0.0, 0
    t0 = time.time()
    for it in range(n_ensayos):
        n = int(rng.integers(2, 9))
        K = int(rng.integers(1, 7))
        sig = [sigma_generico(rng) for _ in range(K)]
        M, salts, flujos = C.monodromia(rng, n, sig)
        nr = C.N_r(sig)
        d_ana = det_analitico(sig, flujos)
        # slogdet es mas estable que det, y cond(M) dice si el signo del
        # producto ensamblado es siquiera representable en doble precision.
        sgn, logabs = np.linalg.slogdet(M)
        d_num = float(sgn)*float(np.exp(logabs)) if np.isfinite(logabs) else 0.0
        cond = float(np.linalg.cond(M))

        # (1) la regla, sobre el determinante exacto
        if np.sign(d_ana) != (-1)**nr:
            f_paridad.append(dict(it=it, n=n, K=K, det_ana=d_ana, Nr=nr,
                                  sigmas=list(map(float, sig))))
            continue
        # (2) el producto ensamblado reproduce el determinante exacto
        rel = abs(d_num - d_ana)/max(abs(d_ana), 1e-300)
        peor_num = max(peor_num, rel if cond < COND_MAX else 0.0)
        if cond >= COND_MAX:
            mal_cond += 1
            continue                     # el signo no es representable: no cuenta
        if rel > 1e-6:
            f_num.append(dict(it=it, n=n, K=K, rel=rel, cond=cond,
                              det_ana=d_ana, det_num=d_num,
                              sigmas=list(map(float, sig))))
            continue
        # (3) paridad del numero de multiplicadores reales negativos
        neg, mu = negativos(M)
        if (neg % 2) != (0 if d_ana > 0 else 1):
            f_eig.append(dict(it=it, n=n, K=K, det_ana=d_ana, neg=neg,
                              cond=cond,
                              mu=[complex(z).__repr__() for z in mu]))
            continue
        # (4) impar => al menos uno
        if nr % 2 == 1 and neg < 1:
            f_impar.append(dict(it=it, n=n, K=K, det_ana=d_ana, neg=neg))
            continue
        if nr % 2 == 0 and neg >= 2:
            permitidos += 1
    return dict(ensayos=n_ensayos, margen=MARGEN, cond_max=COND_MAX,
                descartados_mal_condicionados=mal_cond,
                fallos_paridad=f_paridad, fallos_numericos=f_num,
                fallos_autovalores=f_eig, fallos_impar_sin_negativo=f_impar,
                peor_error_relativo_det=peor_num,
                permitidos_par_con_2neg=permitidos,
                segundos=round(time.time()-t0, 1))


# ==================== bloque B: identidad versorial ====================
def bloque_B(n_ensayos=200):
    rng = np.random.default_rng(SEMILLA + 1)
    fallos, err_max, ww_peor, ok = [], 0.0, 0.0, 0
    t0 = time.time()
    for it in range(n_ensayos):
        n = int(rng.integers(2, 5))
        K = int(rng.integers(1, 4))
        sig = [sigma_generico(rng) for _ in range(K)]
        M, salts, flujos = C.monodromia(rng, n, sig, esc_A=0.5, tau_max=0.6)
        G = C.algebra(n)
        W = C.versor_monodromia(salts, flujos, G)
        if W is None:
            continue
        A = C.accion(W, G)
        err = float(np.max(np.abs(A - M))/max(1.0, np.max(np.abs(M))))
        ww = float((W*~W)(0))
        d_ana = det_analitico(sig, flujos)
        err_max = max(err_max, err)
        ww_peor = max(ww_peor, abs(abs(ww) - 1.0))
        if err > 1e-9:
            fallos.append(dict(it=it, tipo='accion(W) != M', n=n, K=K, err=err,
                               sigmas=list(map(float, sig))))
            continue
        if np.sign(ww) != np.sign(d_ana):
            fallos.append(dict(it=it, tipo='W~W != sign(det M)', n=n, K=K,
                               ww=ww, det_ana=d_ana))
            continue
        ok += 1
    return dict(ensayos=n_ensayos, validos=ok, fallos=fallos,
                err_relativo_max=err_max, desvio_max_absWW=ww_peor,
                segundos=round(time.time()-t0, 1))


if __name__ == '__main__':
    print('=' * 72)
    print(f'BLOQUE A  regla de paridad, n = 2..8, |1+sigma| >= {MARGEN}')
    print('=' * 72)
    A = bloque_A()
    print(f"  ensayos: {A['ensayos']}   ({A['segundos']} s)")
    print(f"  (1) sign(det M) analitico != (-1)^Nr .......... "
          f"{len(A['fallos_paridad'])}")
    print(f"  (2) producto ensamblado != det analitico ...... "
          f"{len(A['fallos_numericos'])}   "
          f"(peor error relativo {A['peor_error_relativo_det']:.2e})")
    print(f"  (3) paridad de multiplicadores negativos ...... "
          f"{len(A['fallos_autovalores'])}")
    print(f"  (4) Nr impar sin multiplicador real negativo .. "
          f"{len(A['fallos_impar_sin_negativo'])}")
    for k in ('fallos_paridad', 'fallos_numericos', 'fallos_autovalores',
              'fallos_impar_sin_negativo'):
        for f in A[k][:3]:
            print(f"     !! {k}: {f}")
    print(f"  descartados por cond(M) >= {A['cond_max']:.0e}: "
          f"{A['descartados_mal_condicionados']}")
    print(f"  casos PERMITIDOS (Nr par, >=2 multiplicadores reales negativos):"
          f" {A['permitidos_par_con_2neg']}")

    print()
    print('=' * 72)
    print('BLOQUE B  identidad versorial y W~W en G(n,n), n = 2..4')
    print('=' * 72)
    B = bloque_B()
    print(f"  ensayos: {B['ensayos']}  validos: {B['validos']}  "
          f"({B['segundos']} s)")
    print(f"  max |accion(W) - M| relativo: {B['err_relativo_max']:.2e}")
    print(f"  max desvio de |W~W| respecto de 1: {B['desvio_max_absWW']:.2e}")
    print(f"  contraejemplos: {len(B['fallos'])}")
    for f in B['fallos'][:5]:
        print(f"     !! {f}")

    tot = (len(A['fallos_paridad']) + len(A['fallos_numericos'])
           + len(A['fallos_autovalores']) + len(A['fallos_impar_sin_negativo'])
           + len(B['fallos']))
    (RES / 'exp1.json').write_text(json.dumps(
        dict(semilla=SEMILLA, margen=MARGEN, bloque_A=A, bloque_B=B), indent=1))
    print()
    print('=' * 72)
    print(f"VEREDICTO: {tot} contraejemplos en "
          f"{A['ensayos'] + B['ensayos']} ensayos del regimen generico")
    print('=' * 72)
    sys.exit(1 if tot else 0)
