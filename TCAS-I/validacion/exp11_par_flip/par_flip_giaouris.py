"""Contraste con datos publicados: boost entrelazado de dos fases de
Giaouris et al., Int. J. Circuit Theory Appl. 42 (2014) 1080, Tabla I
(Vin = 5 V, R = 40 Ohm, L = 1.5 mH, C = 10 uF, T = 100 us, fases iguales,
un controlador de corriente de pico por fase, sin rampa).

Alli el par inestable sale complejo con parte real proxima a -1 y los
autores lo describen como una bifurcacion no descrita antes. Aqui: se
reproducen sus multiplicadores y se les aplica la misma reduccion
(Delta, kappa) que al buck del articulo.
"""
import numpy as np
import par_flip_interleaved as pf
from par_flip_interleaved import monodromia, analiza

pf.T = 100e-6
T = pf.T
Vin, R, C = 5.0, 40.0, 10e-6

class Boost:
    N = 2
    def __init__(self, Ls, Iref, ma=0.0, Vins=(Vin, Vin)):
        self.Ls, self.Iref, self.ma, self.Vins = list(Ls), Iref, ma, Vins
    def A(self, u):                      # estado (i1, i2, v); u_k = 1: interruptor cerrado
        A = np.zeros((3, 3))
        for k in range(2):
            if not u[k]:
                A[k, 2] = -1.0/self.Ls[k]
                A[2, k] = 1.0/C
        A[2, 2] = -1.0/(R*C)
        return A
    def b(self, u):
        return np.array([self.Vins[0]/self.Ls[0], self.Vins[1]/self.Ls[1], 0.0])
    def flw(self, u, x, t):
        from scipy.linalg import expm
        Mx = np.zeros((4, 4)); Mx[:3, :3] = self.A(u); Mx[:3, 3] = self.b(u)
        E = expm(Mx*t)
        return E[:3, :3] @ x + E[:3, 3]

TABLA = [(0.1, 0.1184, "0.3132, -0.1949 +/- 0.0014i"),
         (0.25, 0.4367, "0.5255, -0.8324 +/- 0.0125i"),
         (0.29, 0.4587, "0.5428, -0.9947 +/- 0.0341i"),
         (0.291166, 0.4599, "0.5432, -0.9994 +/- 0.0349i"),
         (0.3, 0.4690, "1.0346... 0.5462, -1.0346 +/- 0.0408i")]

print("1. Tabla I de Giaouris et al. frente a la monodromia por saltaciones")
print("   Iref      d(pub)  d(calc)   publicado                          calculado                         kappa     |Delta|")
g = ([0.1184*T, 0.5*T + 0.1184*T], [0.0605, 0.0829, 5.6977])
ult = None
for Iref, d, pub in TABLA:
    cv = Boost([1.5e-3, 1.5e-3], Iref)
    Ig = ult[0] if ult else 0.1
    for I in np.linspace(Ig, Iref, 12):
        x0, tcs, M, dets = monodromia(Boost([1.5e-3, 1.5e-3], I), *g)
        g = (list(tcs), list(x0))
    ult = (Iref,)
    mu = np.linalg.eigvals(M)
    neg, Delta, k2 = analiza(M)
    pos = [m.real for m in mu if m.real > 0][0]
    print(f"   {Iref:<8}  {d:.4f}  {tcs[0]/T:.4f}   {pub:34s} {pos:.4f}, {neg[1].real:.4f} +/- {abs(neg[1].imag):.4f}i"
          f"        {np.sqrt(max(k2,0)):.4f}   {abs(Delta):.1e}")

print("\n2. En su frontera (Iref = 0.291166 A): umbral de desajuste que convertiria el toro en doblamiento")
x0, tcs, M, dets = monodromia(Boost([1.5e-3, 1.5e-3], 0.291166), *g) if False else (None,)*4
g0 = ([0.4599*T, 0.9599*T], None)
# rehacer la continuacion hasta 0.291166 para tener semilla
g = ([0.1184*T, 0.5*T + 0.1184*T], [0.0605, 0.0829, 5.6977])
for I in np.linspace(0.1, 0.291166, 40):
    x0, tcs, M, dets = monodromia(Boost([1.5e-3, 1.5e-3], I), *g)
    g = (list(tcs), list(x0))
neg, Delta, k2 = analiza(M)
k = np.sqrt(k2)
print(f"   det S = {dets[0]:+.4f}, {dets[1]:+.4f};  kappa = {k:.4f};  ventana del toro: |Delta| < 2 kappa = {2*k:.4f} "
      f"({2*k/abs(dets[0])*100:.1f}% de det S)")
print("   desajuste de L sin rampa (det S = -m2/m1 no depende de L):")
for r in (1.02, 1.05, 1.10, 1.20):
    gg = g
    for rr in np.linspace(1.0, r, 8):
        x0, tcs, M, dets = monodromia(Boost([1.5e-3, 1.5e-3*rr], 0.291166), *gg)
        gg = (list(tcs), list(x0))
    neg, Delta, k2 = analiza(M)
    print(f"     L2/L1 = {r:.2f}:  |Delta|/2k = {abs(Delta)/(2*np.sqrt(k2)):.3f}   par: {neg[0]:.4f}, {neg[1]:.4f}")
print("   desajuste de la tension de entrada vista por cada fase (caida distinta en cada rama):")
for dv in (0.02, 0.05, 0.10, 0.15, 0.20):
    gg = g
    for d in np.linspace(0.0, dv, 8):
        x0, tcs, M, dets = monodromia(Boost([1.5e-3, 1.5e-3], 0.291166, Vins=(Vin, Vin-d)), *gg)
        gg = (list(tcs), list(x0))
    neg, Delta, k2 = analiza(M)
    print(f"     Vin2 = Vin - {dv*1e3:3.0f} mV:  dif det S = {abs(dets[0]-dets[1]):.4f}  |Delta|/2k = {abs(Delta)/(2*np.sqrt(k2)):.3f}   par: {neg[0]:.4f}, {neg[1]:.4f}")
