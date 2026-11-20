"""Paso 1: el simulador temporal reproduce la monodromia.

A) Plataforma de la Tabla III (ma = 0, D = 0.40): jacobiano del mapa
   temporal frente a la monodromia por saltaciones.
B) Cerca de la frontera de estabilidad (ma = 10 kA/s, D = 0.58, 5 mA por
   debajo de Iref*): el par flip sale complejo o real en la simulacion
   temporal donde dice el criterio |Delta| < 2 kappa.
"""
import numpy as np
from scipy.optimize import brentq
from par_flip_interleaved import Conv, T, monodromia, analiza
from sim_dosfases import prep, mapa, jacobiano, orbita_td

np.set_printoptions(precision=7, suppress=True, linewidth=150)
MA = 10e3


def compara(nombre, cv, tcs0, x0g):
    x0, tcs, M, dets = monodromia(cv, tcs0, x0g)
    P = prep(cv)
    xs = orbita_td(cv, x0, tcs, P)
    J = jacobiano(cv, xs, tcs, 1e-6, P)
    neg_m, Delta, k2 = analiza(M)
    neg_j, _, _ = analiza(J)
    k = np.sqrt(max(k2, 0.0))
    tipo = lambda ng: "complejo" if abs(ng[0].imag) > 1e-7 else "real    "
    print(f"  {nombre:30s} |x*_td - x*_F| = {np.max(np.abs(xs-x0)):.1e}  |J - M| = {np.max(np.abs(J-M)):.1e}"
          f"  |Delta|/2k = {abs(Delta)/(2*k):6.3f}")
    print(f"      Floquet : {neg_m[0]:.7f}, {neg_m[1]:.7f}  ({tipo(neg_m)})")
    print(f"      temporal: {neg_j[0]:.7f}, {neg_j[1]:.7f}  ({tipo(neg_j)})")
    return x0, tcs, M, J


def frontera(Ls, RLs, ma=MA):
    """Iref* con |mu| = 1 y una semilla valida para continuar desde alli."""
    g = ([0.58*T, 0.08*T], [2.10, 2.36, 6.75])
    pts = []
    for Iref in np.arange(2.40, 2.62, 0.005):
        try:
            x0, tcs, M, dets = monodromia(Conv(Ls, RLs, Iref, ma), *g)
        except AssertionError:
            continue
        if not (0 < tcs[1] < T/2 < tcs[0] < T):
            continue
        g = (list(tcs), list(x0))
        pts.append((Iref, max(abs(m) for m in analiza(M)[0]), g))
    for a, b in zip(pts[:-1], pts[1:]):
        if (a[1]-1)*(b[1]-1) <= 0:
            f = lambda I: max(abs(m) for m in analiza(monodromia(Conv(Ls, RLs, I, ma), *a[2])[2])[0]) - 1
            return brentq(f, a[0], b[0], xtol=1e-13), a[2]
    raise RuntimeError("sin frontera en el barrido")


CASOS = [("emparejado", [100e-6, 100e-6], [0.060, 0.060]),
         ("L2/L1 = 1.005", [100e-6, 100.5e-6], [0.060, 0.060]),
         ("L2/L1 = 1.010", [100e-6, 101e-6], [0.060, 0.060]),
         ("L2/L1 = 1.020", [100e-6, 102e-6], [0.060, 0.060]),
         ("L2/L1 = 1.050", [100e-6, 105e-6], [0.060, 0.060]),
         ("RL 60/56 mOhm", [100e-6, 100e-6], [0.060, 0.056]),
         ("RL 60/52 mOhm", [100e-6, 100e-6], [0.060, 0.052]),
         ("articulo: L 100/106, RL 60/52", [100e-6, 106e-6], [0.060, 0.052])]

if __name__ == "__main__":
    print("A) Tabla III del articulo (Iref = 1.60 A, ma = 0)")
    compara("articulo: L 100/106, RL 60/52", Conv([100e-6, 106e-6], [0.060, 0.052], 1.60),
            [4.0e-6, 9.0e-6], [1.30, 1.45, 4.2])
    compara("emparejado", Conv([100e-6, 100e-6], [0.060, 0.060], 1.60),
            [4.0e-6, 9.0e-6], [1.30, 1.45, 4.2])

    print("\nB) 5 mA por debajo de la frontera (ma = 10 kA/s)")
    for nombre, Ls, RLs in CASOS:
        Is, g = frontera(Ls, RLs)
        compara(f"{nombre} (Iref* = {Is:.5f})", Conv(Ls, RLs, Is - 0.005, MA), *g)
