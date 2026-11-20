"""Simulador temporal por eventos del buck entrelazado de dos fases.

Independiente de la construccion de Floquet: no usa matrices de saltacion.
Entre eventos el flujo es exacto (autovalores de A por configuracion); el
cruce del comparador se localiza con Newton protegido por biseccion; la
logica es la del circuito: el reloj de cada fase cierra el interruptor y
reinicia su rampa, el comparador lo abre y el latch lo mantiene abierto
hasta el reloj siguiente.

Uso como modulo: prep(), simula(), jacobiano(), orbita_td().
"""
import numpy as np
import numba as nb
from par_flip_interleaved import Conv, T

# --------------------------------------------------------------- nucleo ----

def prep(cv):
    """Descomposicion espectral del flujo afin en las 4 configuraciones."""
    V = np.empty((4, 3, 3), complex)
    lam = np.empty((4, 3), complex)
    Vi = np.empty((4, 3, 3), complex)
    xeq = np.empty((4, 3))
    A = np.empty((4, 3, 3))
    b = np.empty((4, 3))
    for cfg in range(4):
        u = (cfg & 1, (cfg >> 1) & 1)
        A[cfg], b[cfg] = cv.A(u), cv.b(u)
        lam[cfg], V[cfg] = np.linalg.eig(A[cfg])
        Vi[cfg] = np.linalg.inv(V[cfg])
        xeq[cfg] = -np.linalg.solve(A[cfg], b[cfg])
    return V, lam, Vi, xeq, A, b


@nb.njit(cache=True)
def _flujo(V, lam, Vi, xeq, x, t):
    c = np.zeros(3, np.complex128)
    for i in range(3):
        for j in range(3):
            c[i] += Vi[i, j]*(x[j] - xeq[j])
        c[i] *= np.exp(lam[i]*t)
    y = np.empty(3)
    for i in range(3):
        s = 0j
        for j in range(3):
            s += V[i, j]*c[j]
        y[i] = s.real + xeq[i]
    return y


@nb.njit(cache=True)
def _cruce(V, lam, Vi, xeq, A, b, x, k, ref0, ma, tmax):
    """Primer instante s en [0, tmax] con x_k(s) = ref0 - ma*s; -1 si no hay."""
    h0 = x[k] - ref0
    if h0 >= 0.0:
        return 0.0
    xe = _flujo(V, lam, Vi, xeq, x, tmax)
    if xe[k] - (ref0 - ma*tmax) < 0.0:
        return -1.0
    lo, hi = 0.0, tmax
    s = 0.0
    hs = h0
    xs = x
    for _ in range(80):
        dh = b[k] + ma
        for j in range(3):
            dh += A[k, j]*xs[j]
        sn = s - hs/dh if dh != 0.0 else 0.5*(lo+hi)
        if not (lo < sn < hi):
            sn = 0.5*(lo+hi)
        xs = _flujo(V, lam, Vi, xeq, x, sn)
        hs = xs[k] - (ref0 - ma*sn)
        if hs < 0.0:
            lo = sn
        else:
            hi = sn
        if abs(sn - s) < 1e-19 or hi - lo < 1e-19:
            s = sn
            break
        s = sn
    return s


@nb.njit(cache=True)
def simula(V, lam, Vi, xeq, A, b, Iref, ma, x0, on0, trel0, ncic, T):
    """Devuelve el estado al inicio de cada ciclo (reloj de la fase 1)."""
    x = x0.copy()
    on = on0.copy()
    trel = trel0.copy()
    out = np.empty((ncic, 3))
    for c in range(ncic):
        out[c] = x
        for half in range(2):
            on[half] = True
            trel[half] = 0.0
            trem = 0.5*T
            while trem > 1e-18:
                cfg = (1 if on[0] else 0) + (2 if on[1] else 0)
                best = trem
                kb = -1
                for k in range(2):
                    if on[k]:
                        s = _cruce(V[cfg], lam[cfg], Vi[cfg], xeq[cfg], A[cfg], b[cfg],
                                   x, k, Iref[k] - ma[k]*trel[k], ma[k], trem)
                        if s >= 0.0 and s < best:
                            best = s
                            kb = k
                if best > 0.0:
                    x = _flujo(V[cfg], lam[cfg], Vi[cfg], xeq[cfg], x, best)
                trel[0] += best
                trel[1] += best
                trem -= best
                if kb >= 0:
                    on[kb] = False
                else:
                    break
    return out, x, on, trel


# ------------------------------------------------------------ utilidades ----

def estado_inicial(tcs):
    """Estado discreto en t = 0 coherente con la orbita (fase 2 encendida si
    su evento esta envuelto, D > 0.5)."""
    envuelto = tcs[1] < T/2
    return np.array([False, envuelto]), np.array([0.0, 0.5*T if envuelto else 0.0])


def corre(cv, x0, tcs, ncic, P=None):
    P = prep(cv) if P is None else P
    on0, trel0 = estado_inicial(tcs)
    Iref = np.array([cv.Iref, cv.Iref], float) if np.ndim(cv.Iref) == 0 else np.array(cv.Iref, float)
    ma = np.array([cv.ma, cv.ma], float) if np.ndim(cv.ma) == 0 else np.array(cv.ma, float)
    return simula(*P, Iref, ma, np.array(x0, float), on0, trel0, ncic, T)


def mapa(cv, x, tcs, P=None):
    """Un ciclo: estado en t = T partiendo de x en t = 0."""
    return corre(cv, x, tcs, 1, P)[1]


def jacobiano(cv, xs, tcs, d=1e-6, P=None):
    """Jacobiano del mapa estroboscopico por diferencias centradas sobre la
    simulacion temporal."""
    P = prep(cv) if P is None else P
    J = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = d
        J[:, j] = (mapa(cv, xs+e, tcs, P) - mapa(cv, xs-e, tcs, P))/(2*d)
    return J


def orbita_td(cv, xg, tcs, P=None, it=30):
    """Punto fijo del mapa temporal por Newton (no usa el solver de la ruta
    de Floquet salvo como punto de partida)."""
    P = prep(cv) if P is None else P
    x = np.array(xg, float)
    for _ in range(it):
        r = mapa(cv, x, tcs, P) - x
        if np.max(np.abs(r)) < 1e-13:
            break
        x = x - np.linalg.solve(jacobiano(cv, x, tcs, 1e-6, P) - np.eye(3), r)
    return x
