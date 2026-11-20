"""Exp. 5 — Retardo de calculo del control digital.

La seccion de Discusion declara el retardo de control digital como una
limitacion. Este experimento averigua si lo es de verdad.

Planta: el buck con perdidas y ESR de la Sec. V-A (12 V, 100 kHz, L = 100 uH,
C = 100 uF, R_L = 60 mOhm, R_on = 40 mOhm, ESR = 25 mOhm, V_d = 0.45 V,
R_d = 20 mOhm, R = 3 Ohm), regulado a 5 V. Dos controladores sobre la MISMA
planta:

  (a) ANALOGICO, corriente de pico con latch. El instante de conmutacion lo
      fija el estado, h = i_L - (I_ref - m_a t). Evento de REFLEXION.

  (b) DIGITAL, modo tension con un ciclo de retardo de calculo. Se muestrea
      v_C al principio del ciclo, se calcula, y el ciclo devuelve
          d[k+1] = D_0 + K (V_ref - v_C[k]).
      El estado se aumenta con el registro d. Dentro del ciclo la
      conmutacion ocurre en t = d T, disparada por TIEMPO, luego

          h(z,t) = t - d T,  n = [0, 0, -T]^T,  dh/dt = 1,
          hpunto^- = n^T f^- + dh/dt = 1  (porque d se mantiene),
          u = f^+ - f^-  con tercera componente nula,
          sigma = n^T u = 0   EXACTAMENTE.

      Es decir: el retardo vuelve PARABOLICA la saltacion, por construccion y
      no por ajuste. N_r = 0 por esa via.

Y aqui esta la pregunta que importa. Si el muestreador quedara fuera del
formalismo, la regla de paridad no diria nada sobre (b). Pero el muestreador
sustituye la ultima fila de la monodromia, y eso es una actualizacion de
RANGO UNO:

    M = Sigma . Phi_off . S . Phi_on,     Sigma = I + u_S n_S^T,
    1 + sigma_S = det Sigma = K e_v^T P_x^{-1} P_d,

con P_x, P_d los bloques de la monodromia de flujo+saltacion. O sea: el
muestreador es un factor del mismo tipo algebraico que una saltacion, tiene
sus cuatro especies, y la regla de paridad se extiende contandolo.

Se comprueba numericamente: (1) sigma = 0 exacto en el evento temporal,
(2) Sigma tiene rango uno, (3) la forma cerrada de det Sigma, (4) la regla de
paridad ampliada sobre un barrido de la ganancia, (5) que la especie del
muestreador cambia con K y que el canal de flip se abre y se cierra con ella,
(6) el contraste con el controlador analogico sobre la misma planta.

Salida: resultados/exp5.json
"""
import json, pathlib
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq
import comun as C

RES = pathlib.Path(__file__).resolve().parent / 'resultados'
RES.mkdir(exist_ok=True)

# ---- planta: el buck con perdidas y ESR de la Sec. V-A ----
Vin = 12.0
L, RL = 100e-6, 0.060
Cout, ESR, R = 100e-6, 0.025, 3.0
Ron, Vd, Rd = 0.040, 0.45, 0.020
T = 1.0/100e3
Vref = 5.0                                     # consigna del lazo digital
D0 = 0.45                                      # duty nominal del lazo digital

al = R/(R + ESR)                               # divisor carga/ESR, como en buck_realista.py


def matA(Rsw):
    return np.array([[-(Rsw + RL + al*ESR)/L, -al/L],
                     [al/Cout, -al/(R*Cout)]])


A_on, b_on = matA(Ron), np.array([Vin/L, 0.0])
A_off, b_off = matA(Rd), np.array([-Vd/L, 0.0])
EV = np.array([0.0, 1.0])                      # muestrea v_C (tension del condensador)


def flujo(A, b, t):
    """(Phi, beta) del flujo afin durante t."""
    Z = np.zeros((3, 3)); Z[:2, :2] = A; Z[:2, 2] = b
    E = expm(Z*t)
    return E[:2, :2], E[:2, 2]


def cicloP(x, d):
    """x -> P(x,d) del ciclo con duty mantenido, y sus jacobianos."""
    tau = d*T
    Pon, bon = flujo(A_on, b_on, tau)
    Pof, bof = flujo(A_off, b_off, T - tau)
    xc = Pon@x + bon
    xT = Pof@xc + bof
    Px = Pof@Pon
    fm = A_on@xc + b_on                        # f^- en la conmutacion
    fp = A_off@xc + b_off                      # f^+
    Pd = T*(Pof@(fm - fp))                     # = -T Phi_off (f^+ - f^-)
    return xT, Px, Pd, xc, fm, fp, Pon, Pof, tau


def punto_fijo(K):
    def res(z):
        x, d = z[:2], z[2]
        xT = cicloP(x, d)[0]
        return [xT[0]-x[0], xT[1]-x[1], d - (D0 + K*(Vref - x[1]))]
    z, _, ier, msg = fsolve(res, [Vref/R, Vref, D0], full_output=True)
    assert ier == 1, msg
    return z[:2], float(z[2])


def monodromia_digital(K):
    """M aumentada 3x3 con z = [i_L, v_C, d] y un ciclo de retardo."""
    x, d = punto_fijo(K)
    xT, Px, Pd, xc, fm, fp, Pon, Pof, tau = cicloP(x, d)
    M = np.zeros((3, 3))
    M[:2, :2] = Px
    M[:2, 2] = Pd
    M[2, :2] = -K*EV                           # d[k+1] = D0 + K(Vref - v_C[k])
    M[2, 2] = 0.0
    # flujo + saltacion, sin muestreador
    P = np.eye(3); P[:2, :2] = Px; P[:2, 2] = Pd; P[2, 2] = 1.0
    Sig = M @ np.linalg.inv(P)
    return dict(x=x, d=d, tau=tau, M=M, P=P, Sig=Sig, Px=Px, Pd=Pd,
                xc=xc, fm=fm, fp=fp, Pon=Pon, Pof=Pof)


# ======================= (1) la saltacion temporal =======================
print('=' * 74)
print('(1) EL EVENTO DISPARADO POR TIEMPO ES PARABOLICO POR CONSTRUCCION')
print('=' * 74)
K0 = 0.02
q = monodromia_digital(K0)
nv = np.array([0.0, 0.0, -T])                  # n = grad_z (t - d T)
u = np.concatenate([q['fp'] - q['fm'], [0.0]])  # hpunto^- = 1
sig_S = float(nv @ u)
S_aug = np.eye(3) + np.outer(u, nv)
print(f"  punto de trabajo: D = {q['d']:.6f}   "
      f"x* = [{q['x'][0]:.5f} A, {q['x'][1]:.5f} V]")
print(f"  u = [{u[0]:.4e}, {u[1]:.4e}, {u[2]:.1f}]   n = [0, 0, {-T:.2e}]")
print(f"  sigma = n^T u = {sig_S:.3e}      det S = {1+sig_S:.16f}")
print(f"  especie: {C.versor_salt(u, nv, C.algebra(3))[1]}   -> N_r = 0 "
      f"por esta via")
Fon = np.eye(3); Fon[:2, :2] = q['Pon']        # variacional: la b no entra
Fof = np.eye(3); Fof[:2, :2] = q['Pof']
err_P = float(np.max(np.abs(Fof @ S_aug @ Fon - q['P'])))
print(f"  |Phi_off S Phi_on - P| = {err_P:.2e}   -> la factorizacion del "
      f"manuscrito\n    sigue valiendo en el espacio aumentado con el "
      f"registro")

# ======================= (2)-(3) el muestreador ==========================
print()
print('=' * 74)
print('(2) EL MUESTREADOR ES UNA ACTUALIZACION DE RANGO UNO')
print('=' * 74)
Sm1 = q['Sig'] - np.eye(3)
sv = np.linalg.svd(Sm1, compute_uv=False)
print(f"  valores singulares de (Sigma - I): "
      f"{np.array2string(sv, precision=3, suppress_small=False)}")
print(f"  rango numerico = {int(np.sum(sv > 1e-12*max(sv[0], 1.0)))}"
      f"   -> Sigma = I + u_S n_S^T")
sig_Sig = float(np.trace(Sm1))                 # sigma = n^T u = tr(u n^T)
cerrada = K0*float(EV @ np.linalg.solve(q['Px'], q['Pd']))
print(f"  1 + sigma_Sigma = det Sigma = {np.linalg.det(q['Sig']):+.10f}")
print(f"  forma cerrada  K e_v^T P_x^-1 P_d = {cerrada:+.10f}   "
      f"dif = {abs(np.linalg.det(q['Sig'])-cerrada):.2e}")
print(f"  traza de (Sigma - I) = {sig_Sig:+.10f}  "
      f"(coincide con sigma: {abs(1+sig_Sig-cerrada):.2e})")

# versor del muestreador
u_S = Sm1[:, np.argmax(np.linalg.norm(Sm1, axis=0))].copy()
n_S = Sm1.T @ u_S / (u_S @ u_S)
G3 = C.algebra(3)
W_S, esp_S = C.versor_salt(u_S, n_S, G3)
err_vs = (float(np.max(np.abs(C.accion(W_S, G3) - q['Sig'])))
          if W_S is not None else np.nan)
print(f"  especie del muestreador a K = {K0}: {esp_S}   "
      f"|accion(W_Sigma) - Sigma| = {err_vs:.2e}")

# ======================= (4)-(5) barrido de ganancia =====================
print()
print('=' * 74)
print('(4) REGLA DE PARIDAD AMPLIADA A LO LARGO DEL BARRIDO DE GANANCIA')
print('=' * 74)
print(f"{'K':>9} {'1+sig_Sig':>12} {'especie':>12} {'N_r':>4} "
      f"{'pred':>5} {'sign detM':>10} {'mu real neg':>12} {'|mu|max':>9}")
filas, fallos = [], 0
for K in [-0.05, -0.02, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05, 0.10,
          0.20, 0.40]:
    try:
        w = monodromia_digital(K)
    except AssertionError:
        print(f"{K:9.3f}   (sin punto fijo)")
        continue
    dS = float(np.linalg.det(w['Sig']))
    esp = ('clamping' if abs(dS) < 1e-12 else
           'reflexion' if dS < 0 else 'hiperbolico')
    nr = 1 if dS < -1e-12 else 0                # la saltacion aporta 0
    mu = np.linalg.eigvals(w['M'])
    det = float(np.linalg.det(w['M']))
    neg = C.n_reales_negativos(w['M'])
    pred = 0 if abs(dS) < 1e-12 else (-1)**nr
    ok = (np.sign(det) == pred if pred != 0 else abs(det) < 1e-12)
    ok = ok and (nr % 2 == 0 or neg >= 1)
    fallos += (not ok)
    filas.append(dict(K=K, detSig=dS, especie=esp, N_r=nr, det=det,
                      neg=neg, mumax=float(np.max(np.abs(mu))),
                      mu=[[float(z.real), float(z.imag)] for z in mu]))
    print(f"{K:9.3f} {dS:+12.6f} {esp:>12} {nr:4d} {pred:+5d} "
          f"{np.sign(det):+10.0f} {neg:12d} {np.max(np.abs(mu)):9.5f}"
          f"{'   <-- FALLO' if not ok else ''}")

print()
print('=' * 74)
print('(5) DONDE SE ABRE EL CANAL DE FLIP Y DONDE CRUZA')
print('=' * 74)


def espectro(K):
    w = monodromia_digital(K)
    if not (0.02 < w['d'] < 0.98):
        raise ValueError('duty fuera de rango: %.4f' % w['d'])
    mu = np.linalg.eigvals(w['M'])
    r = mu[np.abs(mu.imag) < 1e-9*np.maximum(np.abs(mu), 1.0)].real
    return w, mu, (float(np.min(r[r < 0])) if np.any(r < 0) else np.nan), \
        (float(np.max(r[r > 0])) if np.any(r > 0) else np.nan), \
        float(np.max(np.abs(mu)))


print(f"  para TODO K > 0 el muestreador es de reflexion, luego la regla")
print(f"  fuerza un multiplicador real negativo desde ganancia infinitesimal.")
print(f"  Pero abierto no es lo mismo que cruzado. Donde esta cada uno:")
print()
print(f"{'K':>8} {'1+sig_Sig':>11} {'mu neg':>11} {'mu pos':>10} "
      f"{'|mu|max':>9} {'duty':>8}")
K_est, cruces = None, []
rejilla = [0.01*1.6**k for k in range(22)]
prev = None
for K in rejilla:
    try:
        w, mu, mn, mp, mx = espectro(K)
    except (AssertionError, ValueError) as e:
        print(f"{K:8.4f}   {str(e)[:44]}")
        break
    print(f"{K:8.4f} {np.linalg.det(w['Sig']):+11.6f} "
          f"{mn:11.6f} {mp:10.6f} {mx:9.5f} {w['d']:8.5f}")
    if prev is not None and prev[0] < 1.0 <= mx:
        K_est = K
    prev = (mx,)
if K_est is not None:
    Kc = brentq(lambda K: espectro(K)[4] - 1.0, K_est/1.6, K_est, xtol=1e-10)
    w, mu, mn, mp, mx = espectro(Kc)
    z = mu[np.argmax(np.abs(mu))]
    cual = ('Neimark-Sacker (par complejo)' if abs(z.imag) > 1e-9 else
            'flip (mu = -1)' if z.real < 0 else 'silla-nodo (mu = +1)')
    cruces.append((float(Kc), cual))
    print()
    print(f"  primera perdida de estabilidad: K* = {Kc:.6f}   "
          f"por {cruces[0][1]}")
    print(f"  multiplicadores alli: "
          f"{np.array2string(np.sort_complex(mu), precision=6)}")
    print(f"  1+sigma_Sigma = {np.linalg.det(w['Sig']):+.6f}, "
          f"o sea el canal de flip llevaba abierto\n  desde K = 0+, con el "
          f"multiplicador negativo en {mn:.2e}: la regla dice que\n  EXISTE, "
          f"no que sea el que cruza. Aqui cruza el otro.")
else:
    Kc = None
    print("\n  no se pierde la estabilidad en la rejilla explorada")

# ======================= (6) contraste analogico =========================
print()
print('=' * 74)
print('(6) LA MISMA PLANTA CON CORRIENTE DE PICO ANALOGICA')
print('=' * 74)
ma = 0.0
NVa = np.array([1.0, 0.0])


def punto_analogico(Iref):
    def res(z):
        x0, ton = z[:2], z[2]
        Pon, bon = flujo(A_on, b_on, ton)
        xc = Pon@x0 + bon
        Pof, bof = flujo(A_off, b_off, T - ton)
        x1 = Pof@xc + bof
        return [x1[0]-x0[0], x1[1]-x0[1], xc[0] - (Iref - ma*ton)]
    z, _, ier, msg = fsolve(res, [1.7, 5.0, 0.45*T], full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    Pon, bon = flujo(A_on, b_on, ton)
    xc = Pon@x0 + bon
    Pof, _ = flujo(A_off, b_off, T - ton)
    fm, fp = A_on@xc + b_on, A_off@xc + b_off
    hm, hp = float(NVa@fm + ma), float(NVa@fp + ma)
    u = (fp - fm)/hm
    S = np.eye(2) + np.outer(u, NVa)
    M = Pof @ S @ Pon
    return dict(D=ton/T, sigma=float(NVa@u), hm=hm, hp=hp, S=S, M=M,
                mu=np.linalg.eigvals(M), vC=float(x0[1]))


# la misma salida regulada que el lazo digital: I_ref tal que v_C* = V_ref
Iref = brentq(lambda I: punto_analogico(I)['vC'] - Vref, 1.2, 3.0, xtol=1e-9)
a = punto_analogico(Iref)
print(f"  I_ref = {Iref:.5f} A (v_C* = {a['vC']:.4f} V, misma salida que el lazo digital)")
print(f"  D = {a['D']:.5f}   hpunto^- = {a['hm']:+.4e} A/s   "
      f"hpunto^+ = {a['hp']:+.4e} A/s")
print(f"  1 + sigma = hpunto^+/hpunto^- = {1+a['sigma']:+.6f}   "
      f"especie: reflexion   N_r = 1")
print(f"  det M = {np.linalg.det(a['M']):+.6f}   "
      f"multiplicadores: {np.array2string(a['mu'], precision=5)}")
print(f"  multiplicadores reales negativos: {C.n_reales_negativos(a['M'])}")
print()
print("  Misma planta, dos controladores, dos rutas distintas al mismo canal:")
print("  el analogico lo abre por la SALTACION (el estado dispara el evento),")
print("  el digital por el MUESTREADOR (el evento es temporal y la saltacion")
print("  es parabolica). La regla los cuenta a los dos igual.")

print()
print('=' * 74)
print(f"VEREDICTO: {fallos} fallos de la regla ampliada en "
      f"{len(filas)} ganancias")
print('=' * 74)

# barrido fino, solo para la figura: los mismos observables que la tabla de
# arriba pero con resolucion suficiente para dibujar la frontera de especie.
fino = []
for K in np.linspace(-0.25, 0.50, 151):
    try:
        w, mu, mn, mp, mx = espectro(float(K))
    except (AssertionError, ValueError):
        continue
    fino.append(dict(K=float(K), detSig=float(np.linalg.det(w['Sig'])),
                     det=float(np.linalg.det(w['M'])),
                     mu_neg=(None if np.isnan(mn) else mn), mumax=mx))
print(f"barrido fino para la figura: {len(fino)} puntos")

json.dump(dict(
    sigma_saltacion_temporal=sig_S,
    det_S_temporal=1+sig_S,
    rango_Sigma=int(np.sum(sv > 1e-12*max(sv[0], 1.0))),
    sv_Sigma=[float(s) for s in sv],
    det_Sigma_K0=float(np.linalg.det(q['Sig'])),
    det_Sigma_cerrada=float(cerrada),
    especie_muestreador_K0=esp_S, err_versor_muestreador=float(err_vs),
    barrido=filas, K_critico=(float(Kc) if Kc else None),
    tipo_cruce=(cruces[0][1] if cruces else None), fino=fino,
    analogico=dict(Iref=float(Iref), vC=a['vC'], D=float(a['D']), uno_sigma=float(1+a['sigma']),
                   det=float(np.linalg.det(a['M'])),
                   neg=C.n_reales_negativos(a['M']),
                   mu=[[float(z.real), float(z.imag)] for z in a['mu']]),
    fallos=fallos),
    open(RES / 'exp5.json', 'w'), indent=1)
print('resultados/exp5.json escrito')
raise SystemExit(1 if fallos else 0)
