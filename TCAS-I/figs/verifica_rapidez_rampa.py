"""Verdad numerica de la Sec. V-A: rapidez con rampa de compensacion.

Comprueba tres cosas que el manuscrito afirma o implica:
  1) lambda = ln(m2/m1) con "la rampa incluida en m1" -> ln(m2/(m1+ma))
     frente a la formula correcta lambda = ln|(m2-ma)/(m1+ma)|.
  2) si "rapidez total nula en el canal flip" es equivalente a mu = -1:
     lambda = 0  <=>  ma = (m2-m1)/2, que es LA REGLA CLASICA.
  3) el ciclo de trabajo real en la frontera (el texto dice 0.76, el pie 0.8).
Mismo modelo que fig1_bifurcation.py, sin la simulacion temporal.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

Vin = 12.0
L, RL = 100e-6, 0.060
C, ESR = 100e-6, 0.025
R = 3.0
Ron = 0.040
Vd, Rd = 0.45, 0.020
fs = 100e3
T = 1.0/fs
Iref = 3.35

al = R/(R+ESR)


def matA(Rsw):
    return np.array([[-(Rsw+RL+al*ESR)/L, -al/L],
                     [al/C, -al/(R*C)]])


A_on, b_on = matA(Ron), np.array([Vin/L, 0.0])
A_off, b_off = matA(Rd), np.array([-Vd/L, 0.0])


def aug(A, b):
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = b
    return M


def flw(A, b, x, t):
    E = expm(aug(A, b)*t)
    return E[:2, :2]@x + E[:2, 2]


def punto_de_trabajo(ma):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T-ton)
        return [xc[0]-(Iref-ma*ton), x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, [Iref-0.35, 8.6, 0.8*T], full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    xc = flw(A_on, b_on, x0, ton)
    fm = A_on@xc + b_on
    fp = A_off@xc + b_off
    m1, m2 = fm[0], -fp[0]
    # h = i_L - (Iref - ma t):  hdot- = m1 + ma,  hdot+ = -(m2 - ma)
    hdot_m, hdot_p = m1 + ma, fp[0] + ma
    u = (fp-fm)/hdot_m
    s = u[0]
    S = np.eye(2) + np.outer(u, np.array([1.0, 0.0]))
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    return dict(ton=ton, D=ton/T, m1=m1, m2=m2, s=s, S=S, M=M,
                hdot_m=hdot_m, hdot_p=hdot_p,
                mu=np.sort(np.linalg.eigvals(M).real))


ma_ex = brentq(lambda ma: punto_de_trabajo(ma)['mu'][0] + 1.0, 20e3, 45e3,
               xtol=1.0)
p = punto_de_trabajo(ma_ex)
ma_cl = (p['m2'] - p['m1'])/2

print(f"ma* exacta (mu=-1) = {ma_ex/1e3:.4f} kA/s")
print(f"ma  clasica        = {ma_cl/1e3:.4f} kA/s   (hueco "
      f"{(ma_cl-ma_ex)/1e3:+.4f} kA/s)")
print(f"D en la frontera   = {p['D']:.6f}   "
      f"(el texto dice 0.76, el pie 0.8)")
print(f"m1 = {p['m1']/1e3:.4f}  m2 = {p['m2']/1e3:.4f} kA/s")
print(f"hdot- = m1+ma = {p['hdot_m']/1e3:+.4f}   "
      f"hdot+ = -(m2-ma) = {p['hdot_p']/1e3:+.4f} kA/s")
print(f"1+s   = hdot+/hdot- = {1+p['s']:+.6f}   "
      f"-(m2-ma)/(m1+ma) = {-(p['m2']-ma_ex)/(p['m1']+ma_ex):+.6f}")
print(f"multiplicadores en ma*: {p['mu']}")

lam_bien = np.log(abs(1+p['s']))
lam_paper = np.log(p['m2']/(p['m1']+ma_ex))
print(f"\nlambda CORRECTA  ln|(m2-ma)/(m1+ma)| = {lam_bien:+.6f}")
print(f"lambda del PAPER ln[m2/(m1+ma)]      = {lam_paper:+.6f}"
      f"   error {lam_paper-lam_bien:+.6f}")

# 2) lambda = 0 <=> regla clasica, NO mu = -1
ma_lam0 = brentq(lambda ma: np.log(abs(1+punto_de_trabajo(ma)['s'])),
                 20e3, 45e3, xtol=1.0)
print(f"\nma tal que lambda = 0 : {ma_lam0/1e3:.4f} kA/s")
print(f"  vs regla clasica    : {ma_cl/1e3:.4f} kA/s  -> "
      f"{'COINCIDEN' if abs(ma_lam0-ma_cl) < 5 else 'DIFIEREN'}")
print(f"  vs mu = -1 (exacta) : {ma_ex/1e3:.4f} kA/s  -> "
      f"{'COINCIDEN' if abs(ma_lam0-ma_ex) < 5 else 'DIFIEREN'}")
print("\n=> 'rapidez total nula' NO es equivalente a mu=-1: lambda=0 ES la "
      "regla clasica.")
