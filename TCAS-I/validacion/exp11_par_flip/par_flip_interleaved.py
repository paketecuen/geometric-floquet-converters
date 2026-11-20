"""Sonda (19-11-2026): la mitad PAR de la regla de paridad.

En el interleaved de dos fases N_r = 2 y la regla de paridad solo permite,
sin forzarlo, que el par flip sea real. Aqui se contrasta un criterio:

    el par flip es complejo   <=>   |Delta| < 2*kappa

Delta y kappa salen del bloque 2x2 de corrientes de M una vez eliminada la
tension (complemento de Schur en la media del par flip):
    Delta   = Me[0,0] - Me[1,1]     desajuste entre las dos reflexiones
    kappa^2 = -Me[0,1]*Me[1,0]      interferencia entre las dos fases
y entonces Im(mu) = sqrt(kappa^2 - Delta^2/4).

Solo matrices. Mismo modelo que TCAS-I/interleaved_realista.py, con rampa
de compensacion opcional y con el evento de la fase 2 envuelto al periodo
siguiente cuando D > 0.5.

Lo que NO comprueba: el atractor mas alla de la frontera (simulacion
temporal), ni da kappa en forma cerrada.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

Vin, C, ESR, R = 12.0, 100e-6, 0.025, 1.5
Ron, Vd, Rd = 0.040, 0.45, 0.020
fs = 100e3
T = 1.0/fs
al = R/(R+ESR)


class Conv:
    def __init__(self, Ls, RLs, Iref, ma=0.0):
        self.N = len(Ls)
        self.Ls, self.RLs, self.Iref, self.ma = list(Ls), list(RLs), Iref, ma

    def A(self, u):
        N = self.N
        A = np.zeros((N+1, N+1))
        for k in range(N):
            Rk = self.RLs[k] + (Ron if u[k] else Rd)
            A[k, :N] = -al*ESR/self.Ls[k]
            A[k, k] -= Rk/self.Ls[k]
            A[k, N] = -al/self.Ls[k]
            A[N, k] = al/C
        A[N, N] = -al/(R*C)
        return A

    def b(self, u):
        return np.array([(Vin if u[k] else -Vd)/self.Ls[k] for k in range(self.N)] + [0.0])

    def flw(self, u, x, t):
        Mx = np.zeros((self.N+2,)*2)
        Mx[:-1, :-1] = self.A(u)
        Mx[:-1, -1] = self.b(u)
        E = expm(Mx*t)
        return E[:-1, :-1] @ x + E[:-1, -1]


def u_en(cv, tcs, t):
    u = []
    for k in range(cv.N):
        ton = k*T/cv.N
        if tcs[k] > ton:
            u.append(1 if ton <= t < tcs[k] else 0)
        else:                                    # evento envuelto (D > 0.5)
            u.append(1 if (t >= ton or t < tcs[k]) else 0)
    return tuple(u)


def tray(cv, x0, tcs, tfin):
    cortes = sorted(set([k*T/cv.N for k in range(cv.N)] + list(tcs) + [tfin, T]))
    t, x = 0.0, np.array(x0, float)
    for tc in cortes:
        if tc <= t:
            continue
        tc = min(tc, tfin)
        x = cv.flw(u_en(cv, tcs, (t+tc)/2), x, tc-t)
        t = tc
        if t >= tfin:
            break
    return x


def monodromia(cv, tcs0, x0g):
    N = cv.N

    def res(z):
        x0, tcs = z[:N+1], z[N+1:]
        r = []
        for k in range(N):
            ton = k*T/N
            dt = tcs[k]-ton if tcs[k] > ton else tcs[k]+T-ton
            r.append(tray(cv, x0, tcs, tcs[k])[k] - (cv.Iref - cv.ma*dt))
        return r + list(tray(cv, x0, tcs, T) - x0)

    z, _, ier, msg = fsolve(res, list(x0g)+list(tcs0), full_output=True, xtol=1e-13)
    assert ier == 1, msg
    x0, tcs = z[:N+1], z[N+1:]
    M = np.eye(N+1)
    t, x, dets = 0.0, x0.copy(), [None]*N
    for te in sorted([k*T/N for k in range(1, N)] + list(tcs) + [T]):
        u = u_en(cv, tcs, (t+te)/2)
        M = expm(cv.A(u)*(te-t)) @ M
        x = cv.flw(u, x, te-t)
        t = te
        for k in range(N):
            if abs(te - tcs[k]) < 1e-15:
                um, up = list(u), list(u)
                um[k], up[k] = 1, 0
                fm = cv.A(tuple(um)) @ x + cv.b(tuple(um))
                fp = cv.A(tuple(up)) @ x + cv.b(tuple(up))
                nk = np.zeros(N+1)
                nk[k] = 1.0
                uk = (fp-fm)/(nk @ fm + cv.ma)
                dets[k] = 1.0 + nk @ uk
                M = (np.eye(N+1) + np.outer(uk, nk)) @ M
    return x0, tcs, M, dets


def analiza(M):
    mu = np.linalg.eigvals(M)
    neg = sorted([m for m in mu if m.real < 0], key=lambda z: z.imag)
    mu0 = np.mean([m.real for m in neg])
    Me = M[:2, :2] + M[:2, 2:] @ M[2:, :2]/(mu0 - M[2, 2])
    Delta = Me[0, 0] - Me[1, 1]
    k2 = -Me[0, 1]*Me[1, 0]
    return neg, Delta, k2


def fila(nombre, cv, tcs0, x0g):
    x0, tcs, M, dets = monodromia(cv, tcs0, x0g)
    neg, Delta, k2 = analiza(M)
    k = np.sqrt(max(k2, 0.0))
    imp = np.sqrt(max(k2 - Delta**2/4, 0.0))
    print(f"  {nombre:30s} |dif det S| = {abs(dets[0]-dets[1]):.2e}  |Delta|/2k = {abs(Delta)/(2*k):6.3f}"
          f"  Im pred. {imp:.3e}  Im real {abs(neg[0].imag):.3e}   par: {neg[0]:.5f}, {neg[1]:.5f}")
    return x0, tcs, M, dets


G0 = ([4.0e-6, 9.0e-6], [1.30, 1.45, 4.2])            # D < 0.5 (Tabla III)

if __name__ == "__main__":
    print("1. Punto de trabajo de la Tabla III (Iref = 1.60 A, ma = 0, D = 0.40)")
    fila("articulo: L 100/106, RL 60/52", Conv([100e-6, 106e-6], [0.060, 0.052], 1.60), *G0)
    fila("emparejado", Conv([100e-6, 100e-6], [0.060, 0.060], 1.60), *G0)

    print("\n2. Desajuste de RL (L 100/100, ma = 0)")
    for rl2 in (56, 52, 48, 46, 42, 30):
        fila(f"RL 60/{rl2} mOhm", Conv([100e-6, 100e-6], [0.060, rl2*1e-3], 1.60), *G0)

    def disc(rl2):
        _, Delta, k2 = analiza(monodromia(Conv([100e-6, 100e-6], [0.060, rl2], 1.60), *G0)[2])
        return Delta**2 - 4*k2
    def compl(rl2):
        return abs(analiza(monodromia(Conv([100e-6, 100e-6], [0.060, rl2], 1.60), *G0)[2])[0][0].imag) > 1e-9
    lo, hi = 0.040, 0.052
    for _ in range(45):
        mid = 0.5*(lo+hi)
        lo, hi = (lo, mid) if compl(mid) else (mid, hi)
    print(f"  criterio |Delta| = 2k:      RL2 = {brentq(disc, 0.040, 0.052, xtol=1e-10)*1e3:.3f} mOhm")
    print(f"  el par se hace real en:     RL2 = {0.5*(lo+hi)*1e3:.3f} mOhm")

    print("\n3. Desajuste de L con y sin rampa (RL 60/60, Iref = 1.60 A)")
    for ma in (0.0, 15e3):
        print(f"  ma = {ma/1e3:.0f} kA/s")
        for r in (1.01, 1.02, 1.06, 1.10):
            fila(f"L2/L1 = {r:.2f}", Conv([100e-6, 100e-6*r], [0.060, 0.060], 1.60, ma), *G0)

    print("\n4. En la frontera de estabilidad (|mu| = 1) con ma = 10 kA/s, D = 0.58")
    def frontera(nombre, Ls, RLs):
        g = ([0.58*T, 0.08*T], [2.10, 2.36, 6.75])
        pts = []
        for Iref in np.arange(2.40, 2.62, 0.005):
            try:
                x0, tcs, M, dets = monodromia(Conv(Ls, RLs, Iref, 10e3), *g)
            except AssertionError:
                continue
            if not (0 < tcs[1] < T/2 < tcs[0] < T):
                continue
            g = (list(tcs), list(x0))
            pts.append((Iref, max(abs(m) for m in analiza(M)[0]), g))
        for a, b in zip(pts[:-1], pts[1:]):
            if (a[1]-1)*(b[1]-1) <= 0:
                f = lambda I: max(abs(m) for m in analiza(monodromia(Conv(Ls, RLs, I, 10e3), *a[2])[2])[0]) - 1
                Is = brentq(f, a[0], b[0], xtol=1e-12)
                x0, tcs, M, dets = monodromia(Conv(Ls, RLs, Is, 10e3), *a[2])
                neg, Delta, k2 = analiza(M)
                if abs(neg[0].imag) > 1e-9:
                    th = np.pi - abs(np.angle(neg[1]))
                    tipo = f"par complejo (Neimark-Sacker); batido cada {2*np.pi/th:.0f} ciclos"
                else:
                    tipo = f"reales: -1 y {max(m.real for m in neg):.4f} (flip)"
                print(f"  {nombre:30s} |dif det S| = {abs(dets[0]-dets[1])/abs(dets[0])*100:4.2f}%"
                      f"  |Delta|/2k = {abs(Delta)/(2*np.sqrt(k2)):6.3f}  -> {tipo}")
                return
        print(f"  {nombre:30s} sin cruce en el barrido")
    frontera("emparejado", [100e-6, 100e-6], [0.060, 0.060])
    for r in (1.005, 1.01, 1.02, 1.05):
        frontera(f"L2/L1 = {r:.3f}", [100e-6, 100e-6*r], [0.060, 0.060])
    frontera("RL 60/56", [100e-6, 100e-6], [0.060, 0.056])
    frontera("RL 60/52", [100e-6, 100e-6], [0.060, 0.052])
    frontera("articulo: L 100/106, RL 60/52", [100e-6, 106e-6], [0.060, 0.052])
