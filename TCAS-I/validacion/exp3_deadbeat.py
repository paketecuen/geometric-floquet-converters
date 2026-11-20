"""Exp. 3 — La prediccion deadbeat, atacada desde fuera del marco.

La afirmacion nueva del manuscrito es que la cuarta especie (clamping,
det S = 0) cae exactamente en m_a = m_2, y que ese punto es la condicion de
deadbeat clasica del lazo de corriente. Eso es falsable sin tocar el algebra
geometrica, y es lo que se hace aqui:

  (1) La frontera de especie. Localizar por biseccion el m_a donde
      det S = 1+sigma cruza cero, y comprobar que coincide con m_a = m_2
      resuelto de forma autoconsistente (m_2 depende de m_a porque la orbita
      se desplaza).

  (2) La estructura de proyector. En ese punto S debe ser idempotente de
      rango n-1, y M debe perder un rango: un multiplicador exactamente nulo.

  (3) EL ENSAYO INDEPENDIENTE: asentamiento en un ciclo. Se perturba la orbita
      periodica y se integra UN periodo con un integrador de eventos exacto
      (biseccion sobre el flujo en forma cerrada, sin paso fijo). Si el punto
      es deadbeat, la componente de corriente de la perturbacion debe morir en
      un solo ciclo. Esto no usa versores, ni monodromia, ni autovalores: es
      dominio del tiempo.

  (4) El barrido alrededor. Numero de ciclos hasta que la perturbacion baja de
      1e-9, en funcion de m_a. Debe tener un minimo en el punto deadbeat.

Salida: resultados/exp3.json
"""
import json, pathlib
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

RES = pathlib.Path(__file__).resolve().parent / 'resultados'
RES.mkdir(exist_ok=True)

# ---- plataforma de la Sec. V-A del manuscrito (R = 3 ohm) ----
Vin, L, RL = 12.0, 100e-6, 0.060
Cc, ESR, R = 100e-6, 0.025, 3.0
Ron, Vd, Rd = 0.040, 0.45, 0.020
T, Iref = 1.0/100e3, 3.35
al = R/(R+ESR)


def matA(Rsw):
    return np.array([[-(Rsw+RL+al*ESR)/L, -al/L], [al/Cc, -al/(R*Cc)]])


A_on, b_on = matA(Ron), np.array([Vin/L, 0.0])
A_off, b_off = matA(Rd), np.array([-Vd/L, 0.0])


def prep(A, b):
    lam, V = np.linalg.eig(A)
    return lam, V, np.linalg.inv(V), np.linalg.solve(A, b)


PON, POFF = prep(A_on, b_on), prep(A_off, b_off)


def flujo(P, x, t):
    lam, V, Vi, Ainvb = P
    return np.real((V*np.exp(lam*t)) @ Vi @ (x + Ainvb) - Ainvb)


def orbita(ma):
    """Orbita periodica, pendientes y saltacion en el evento de pico."""
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flujo(PON, x0, ton)
        x1 = flujo(POFF, xc, T-ton)
        return [xc[0]-(Iref-ma*ton), x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, [Iref-0.35, 8.6, 0.6*T], full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    xc = flujo(PON, x0, ton)
    fm, fp = A_on@xc + b_on, A_off@xc + b_off
    m1, m2 = fm[0], -fp[0]
    nv = np.array([1.0, 0.0])
    u = (fp - fm)/(m1 + ma)                 # hdot^- = m1 + ma
    S = np.eye(2) + np.outer(u, nv)
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    return dict(x0=x0, ton=ton, xc=xc, m1=m1, m2=m2, u=u, S=S, M=M,
                sigma=float(nv @ u))


def un_ciclo(x0, ma):
    """Un periodo con el evento resuelto por biseccion. Si no hay cruce, el
    ciclo satura en ON."""
    def h(t):
        return flujo(PON, x0, t)[0] - (Iref - ma*t)
    if h(0.0) >= 0.0:
        return flujo(POFF, x0, T)
    if h(T) < 0.0:
        return flujo(PON, x0, T)
    ton = brentq(h, 0.0, T, xtol=1e-15, rtol=8.9e-16)
    return flujo(POFF, flujo(PON, x0, ton), T - ton)


# ===================== (1) frontera de especie =====================
print('=' * 72)
print('(1) FRONTERA DE ESPECIE')
print('=' * 72)
ma_det0 = brentq(lambda ma: 1.0 + orbita(ma)['sigma'], 60e3, 140e3, xtol=1e-4)
ma_m2 = brentq(lambda ma: orbita(ma)['m2'] - ma, 60e3, 140e3, xtol=1e-4)
p = orbita(ma_det0)
print(f"  m_a con det S = 0 (biseccion) : {ma_det0/1e3:.6f} kA/s")
print(f"  m_a con m_a = m_2 (autoconsist.): {ma_m2/1e3:.6f} kA/s")
print(f"  |diferencia| = {abs(ma_det0-ma_m2):.3e} A/s   -> "
      f"{'COINCIDEN' if abs(ma_det0-ma_m2) < 1.0 else 'DIFIEREN'}")
print(f"  en ese punto: m_1 = {p['m1']/1e3:.4f}  m_2 = {p['m2']/1e3:.4f} kA/s"
      f"   1+sigma = {1+p['sigma']:+.3e}")

# ===================== (2) estructura de proyector =====================
print()
print('=' * 72)
print('(2) ESTRUCTURA DE PROYECTOR')
print('=' * 72)
S, M = p['S'], p['M']
print(f"  |S^2 - S| = {np.max(np.abs(S@S - S)):.3e}   (idempotente)")
print(f"  det S = {np.linalg.det(S):+.3e}   rango S = "
      f"{np.linalg.matrix_rank(S, tol=1e-9)}  de 2")
mu = np.sort(np.linalg.eigvals(M).real)
print(f"  det M = {np.linalg.det(M):+.3e}   rango M = "
      f"{np.linalg.matrix_rank(M, tol=1e-9)}  de 2")
print(f"  multiplicadores: {mu}")
print(f"  |multiplicador nulo| = {abs(mu[0]):.3e}")

# ============ (3) ensayo independiente: asentamiento en un ciclo ============
print()
print('=' * 72)
print('(3) ASENTAMIENTO EN UN CICLO  (dominio del tiempo, sin algebra)')
print('=' * 72)
# OJO: el observable NO es la componente de corriente al cerrar el ciclo. El
# proyector anula la perturbacion en el INSTANTE de conmutacion, y el flujo
# OFF posterior la vuelve a mezclar antes de completar el periodo. Lo que
# muere en un ciclo es la componente sobre la direccion propia NULA de M, que
# es la afirmacion con contenido: rango M = n-1, asi que existe un subespacio
# de perturbaciones que se extingue completamente en un periodo.
val, vec = np.linalg.eig(p['M'])
i0 = int(np.argmin(np.abs(val)))
v0 = np.real(vec[:, i0]); v0 = v0/np.linalg.norm(v0)
v1 = np.real(vec[:, 1-i0]); v1 = v1/np.linalg.norm(v1)
print(f"  direccion propia nula de M: v0 = [{v0[0]:+.6f}, {v0[1]:+.6f}]"
      f"   mu = {val[i0].real:+.3e}")
print(f"  la otra:                    v1 = [{v1[0]:+.6f}, {v1[1]:+.6f}]"
      f"   mu = {val[1-i0].real:+.3e}")
print()
print(f"{'perturbacion':>34} {'residuo tras 1 ciclo':>24} {'factor':>11}")
res3 = []
base = un_ciclo(p['x0'], ma_det0)
for nom, d in (('a lo largo de v0 (nula)', 1e-3*v0),
               ('a lo largo de v0, 10x',   1e-2*v0),
               ('a lo largo de v1',        1e-3*v1),
               ('generica',                np.array([5e-3, 5e-3]))):
    dx1 = un_ciclo(p['x0'] + d, ma_det0) - base
    fac = np.linalg.norm(dx1)/np.linalg.norm(d)
    res3.append(dict(nombre=nom, d=list(map(float, d)),
                     dx1=list(map(float, dx1)), factor=float(fac)))
    print(f"  {nom:>32}  [{dx1[0]:+.2e}, {dx1[1]:+.2e}]  {fac:11.2e}")
print("  la de v1 sobrevive con el factor |mu| que le corresponde.")

# El residuo sobre v0 NO es redondeo: es el termino de segundo orden del mapa
# de un ciclo, que es no lineal. Si la parte lineal se anula exactamente, el
# residuo debe escalar como delta^2. Se mide la pendiente en log-log.
print()
print("  escalado del residuo sobre la direccion nula (debe ser cuadratico):")
print(f"{'delta':>12} {'residuo':>13} {'residuo/delta^2':>18}")
esc = []
for dd in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
    r = float(np.linalg.norm(un_ciclo(p['x0'] + dd*v0, ma_det0) - base))
    esc.append(dict(delta=dd, residuo=r, r_sobre_d2=r/dd**2))
    print(f"{dd:12.0e} {r:13.3e} {r/dd**2:18.4f}")
ld = np.log([e['delta'] for e in esc]); lr = np.log([e['residuo'] for e in esc])
pend = float(np.polyfit(ld, lr, 1)[0])
print(f"  pendiente log-log = {pend:.4f}  "
      f"(2 = cuadratico exacto -> la parte lineal se anula)")

# ===================== (4) barrido alrededor =====================
print()
print('=' * 72)
print('(4) EL MODO RAPIDO alrededor del punto deadbeat')
print('=' * 72)
# El asentamiento TOTAL lo gobierna el multiplicador lento (el de tension,
# ~0.96), que es ajeno al deadbeat: medirlo da >40 ciclos en todos los puntos
# y no dice nada. Lo que el deadbeat toca es el modo RAPIDO, el del lazo de
# corriente. Se mide su decaimiento medido en el tiempo, ciclo a ciclo.
print(f"{'m_a [kA/s]':>11} {'1+sigma':>11} {'mu_rapido':>11} "
      f"{'factor medido':>14} {'ciclos a 1e-9':>14}")
res4 = []
for ma in (60e3, 75e3, 82e3, 85e3, ma_det0, 88e3, 95e3, 110e3, 130e3):
    q = orbita(ma)
    vv, VV = np.linalg.eig(q['M'])
    j = int(np.argmin(np.abs(vv)))
    vr = np.real(VV[:, j]); vr = vr/np.linalg.norm(vr)
    mu_r = float(vv[j].real)
    b = un_ciclo(q['x0'], ma)
    d = 1e-3*vr
    fac = float(np.linalg.norm(un_ciclo(q['x0'] + d, ma) - b)
                / np.linalg.norm(d))
    ciclos = (1 if fac < 1e-9
              else int(np.ceil(np.log(1e-9)/np.log(max(fac, 1e-300)))))
    marca = '  <-- deadbeat' if abs(ma - ma_det0) < 1.0 else ''
    res4.append(dict(ma=float(ma), uno_sigma=float(1+q['sigma']),
                     mu_rapido=mu_r, factor=fac, ciclos=ciclos))
    print(f"{ma/1e3:11.3f} {1+q['sigma']:+11.4f} {mu_r:+11.6f} "
          f"{fac:14.3e} {ciclos:>14}{marca}")

(RES / 'exp3.json').write_text(json.dumps(dict(
    ma_det_cero=float(ma_det0), ma_m2=float(ma_m2),
    diferencia=float(abs(ma_det0-ma_m2)),
    m1=float(p['m1']), m2=float(p['m2']), uno_sigma=float(1+p['sigma']),
    idempotencia=float(np.max(np.abs(S@S - S))),
    det_M=float(np.linalg.det(M)), multiplicadores=list(map(float, mu)),
    ensayo_un_ciclo=res3, escalado=esc, pendiente_loglog=pend,
    barrido=res4), indent=1))
print()
print('=' * 72)
print('resultados/exp3.json escrito')
print('=' * 72)
