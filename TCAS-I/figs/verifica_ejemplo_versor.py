"""Verificacion del ejemplo corriente de la Seccion III-B del manuscrito.

El texto afirma que el buck de control por corriente de pico, en el punto de
trabajo D = 0.40, produce el versor  W_S = 0.2011 + 1.0200 e1f1  con
1+s = -m2/m1 = -0.6706.  Ese numero NO lo genera ningun script del repo, asi
que aqui se recalcula desde el modelo de la Seccion V (plataforma con
perdidas) y se compara con la forma cerrada del caso (iii):

    W = Bhat * exp(lam*Bhat/2) = sinh(lam/2) + cosh(lam/2)*Bhat,
    Bhat = B/s,  B = 2*(U ^ V),  lam = ln|1+s|.

Se comprueba ademas cual es el signo de Bhat en la base {e1,f1}, que es el
punto donde el manuscrito puede estar equivocado.
"""
import numpy as np
from scipy.linalg import expm
from clifford import Cl

# ---------------- plataforma de la Seccion V ----------------
Vin = 12.0
L, RL = 100e-6, 0.060
C, ESR = 100e-6, 0.025
Ron = 0.040
Vd, Rd = 0.45, 0.020
fs = 100e3
T = 1.0/fs


def modelo(R):
    al = R/(R+ESR)

    def matA(Rsw):
        return np.array([[-(Rsw+RL+al*ESR)/L, -al/L],
                         [al/C, -al/(R*C)]])
    A_on, b_on = matA(Ron), np.array([Vin/L, 0.0])
    A_off, b_off = matA(Rd), np.array([-Vd/L, 0.0])
    return A_on, b_on, A_off, b_off


def flw(A, b, x, t):
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = b
    E = expm(M*t)
    return E[:2, :2]@x + E[:2, 2]


def orbita_ccm(R, D):
    """Orbita periodica de ciclo de trabajo D exacto: el problema es LINEAL
    con t_on fijado, asi que no hace falta ni fsolve ni continuacion.
      x1 = E_off(E_on x0 + c_on) + c_off = x0  ->  (I - E)x0 = c
    Iref queda determinado a posteriori como la corriente de pico."""
    A_on, b_on, A_off, b_off = modelo(R)
    ton = D*T
    Fon = expm(np.block([[A_on, b_on[:, None]], [np.zeros((1, 3))]])*ton)
    Foff = expm(np.block([[A_off, b_off[:, None]], [np.zeros((1, 3))]])*(T-ton))
    Ftot = Foff @ Fon
    E, c = Ftot[:2, :2], Ftot[:2, 2]
    x0 = np.linalg.solve(np.eye(2) - E, c)
    return x0, ton


def punto(R, D):
    """Devuelve D, m1, m2, s, u, n, S en el punto de trabajo."""
    A_on, b_on, A_off, b_off = modelo(R)
    x0, ton = orbita_ccm(R, D)
    xc = flw(A_on, b_on, x0, ton)
    Iref = xc[0]
    assert x0[0] > 0, f"valle negativo ({x0[0]:.4f} A): no es CCM"
    fm = A_on@xc + b_on          # campo entrante (subida)
    fp = A_off@xc + b_off        # campo saliente (bajada)
    m1, m2 = fm[0], -fp[0]
    nv = np.array([1.0, 0.0])    # h = i_L - Iref  (sin rampa de compensacion)
    u = (fp - fm)/(nv @ fm)
    s = nv @ u
    S = np.eye(2) + np.outer(u, nv)
    return dict(D=ton/T, m1=m1, m2=m2, s=s, u=u, n=nv, S=S, xc=xc, x0=x0,
                Iref=Iref, ton=ton, A_on=A_on, b_on=b_on,
                A_off=A_off, b_off=b_off)


# ---------------- algebra G(2,2) ----------------
layout, blades = Cl(2, 2)
e1, e2, f1, f2 = (blades['e%d' % k] for k in (1, 2, 3, 4))
w = [(e1+f1)/2, (e2+f2)/2]
v = [(e1-f1)/2, (e2-f2)/2]


def mvexp(B, N=300):
    tot = 1 + 0*B
    term = 1 + 0*B
    for k in range(1, N):
        term = term*B/k
        tot = tot + term
        if max(abs(term.value)) < 1e-19:
            break
    return tot


def inv(W):
    return ~W/float((W*~W)(0))


def accion(W):
    Wi = inv(W)
    return np.column_stack([np.array([2*float((v[i] | (W*w[k]*Wi))(0))
                                      for i in range(2)]) for k in range(2)])


def versor(u, nv):
    U = sum(float(u[i])*w[i] for i in range(2))
    V = sum(float(nv[j])*v[j] for j in range(2))
    B = 2*(U ^ V)
    s = float(nv @ u)
    Bh = B/s
    lam = np.log(abs(1+s))
    W = Bh * mvexp(lam*Bh/2)
    return W, B, Bh, s, lam


# ---------------- D = 0.40 en las dos plataformas ----------------
for R in (15.0, 3.0):
    print(f"\n===== plataforma R = {R:g} ohm, D = 0.40 =====")
    p = punto(R, 0.40)
    uno_mas_s = 1.0 + p['s']
    print(f"Iref  = {p['Iref']:.6f} A   valle = {p['x0'][0]:.6f} A   "
          f"Vout = {p['x0'][1]:.4f} V   D = {p['D']:.6f}")
    print(f"m1 = {p['m1']/1e3:+.4f} kA/s   m2 = {p['m2']/1e3:+.4f} kA/s")
    print(f"-m2/m1 = {-p['m2']/p['m1']:+.6f}    1+s = {uno_mas_s:+.6f}   "
          f"s = {p['s']:+.6f}")
    print(f"u = [{p['u'][0]:+.6f}, {p['u'][1]:+.6e}]   (u2 debe ser 0)")

    W, B, Bh, s, lam = versor(p['u'], p['n'])
    err = np.max(np.abs(accion(W) - p['S']))
    ww = float((W*~W)(0))
    print(f"lambda = ln|1+s| = {lam:+.6f}")
    print(f"sinh(lam/2) = {np.sinh(lam/2):+.6f}   "
          f"cosh(lam/2) = {np.cosh(lam/2):+.6f}")
    print(f"Bhat = {Bh}")
    print(f"W    = {W}")
    print(f"  escalar        = {float(W(0)):+.6f}")
    print(f"  coef. de e1f1  = {float((W(2) | (e1*f1))(0)):+.6f}")
    print(f"|accion(W)-S| = {err:.2e}   W~W = {ww:+.6f}   "
          f"detS = {np.linalg.det(p['S']):+.6f}")
    assert err < 1e-12 and abs(ww+1) < 1e-12

    # el versor opuesto -W actua igual: comprobacion explicita
    err2 = np.max(np.abs(accion(-W) - p['S']))
    print(f"|accion(-W)-S| = {err2:.2e}  -> W esta definido salvo signo global")
