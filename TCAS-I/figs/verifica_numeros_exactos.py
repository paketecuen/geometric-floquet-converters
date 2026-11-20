"""Raices EXACTAS de los tres numeros que el manuscrito publica como nodos
de rejilla (hallazgos M1, M2, M3 de la revision adversarial):

  M1  frontera DCM->CCM: el paper da I_ref* = 0.1205 A, que es el punto medio
      entre dos nodos de linspace. La frontera real es valle(i_L) = 0.
  M2  cruce de mu = -1: el paper da I_ref = 0.544 A, D = 0.507, que es el
      primer nodo con mu < -1 (alli mu = -1.0239 ya).
  M3  ciclo de trabajo en la frontera exacta de rampa m_a* (Fig. 1).

Plataforma de carga ligera R = 15 ohm (Fig. 3) para M1/M2, y R = 3 ohm
(Fig. 1) para M3.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

Vin = 12.0
L, RL = 100e-6, 0.060
C, ESR = 100e-6, 0.025
Ron = 0.040
Vd, Rd = 0.45, 0.020
fs = 100e3
T = 1.0/fs


def mats(R):
    al = R/(R+ESR)

    def matA(Rsw):
        return np.array([[-(Rsw+RL+al*ESR)/L, -al/L],
                         [al/C, -al/(R*C)]])
    return (matA(Ron), np.array([Vin/L, 0.0]),
            matA(Rd), np.array([-Vd/L, 0.0]),
            np.array([[0.0, 0.0], [0.0, -al/(R*C)]]), np.zeros(2))


def flw(A, b, x, t):
    M = np.zeros((3, 3))
    M[:2, :2] = A
    M[:2, 2] = b
    E = expm(M*t)
    return E[:2, :2]@x + E[:2, 2]


A_on, b_on, A_off, b_off, A_idl, b_idl = mats(15.0)


def ccm(Iref, guess):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T-ton)
        return [xc[0]-Iref, x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, guess, full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    xc = flw(A_on, b_on, x0, ton)
    fm, fp = A_on@xc + b_on, A_off@xc + b_off
    nv = np.array([1.0, 0.0])
    u = (fp-fm)/(nv @ fm)
    S = np.eye(2) + np.outer(u, nv)
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    return dict(valle=x0[0], D=ton/T, z=z, s=u[0],
                mu=np.sort(np.linalg.eigvals(M).real))


# ---- M1: frontera exacta = valle nulo ----
g = [0.10, 4.5, 0.30*T]
Iref_b = brentq(lambda I: ccm(I, g)['valle'], 0.10, 0.20, xtol=1e-12)
pb = ccm(Iref_b, g)
print("M1  frontera DCM->CCM (valle = 0)")
print(f"    exacta        I_ref* = {Iref_b:.6f} A   D = {pb['D']:.6f}   "
      f"mu = {np.round(pb['mu'],6)}")
print(f"    publicada     I_ref* = 0.1205 A        -> error "
      f"{100*(0.1205-Iref_b)/Iref_b:+.2f} %  (falla en la 3a cifra)")
print(f"    mu_flip justo en CCM = {pb['mu'][0]:+.6f}  "
      f"(el paper dice 'jumps to -0.124')")

# ---- M2: cruce exacto mu = -1 ----
g2 = [0.35, 4.6, 0.45*T]
Iref_f = brentq(lambda I: ccm(I, g2)['mu'][0] + 1.0, 0.45, 0.65, xtol=1e-12)
pf = ccm(Iref_f, g2)
print("\nM2  cruce de mu = -1")
print(f"    exacto        I_ref = {Iref_f:.6f} A   D = {pf['D']:.6f}   "
      f"mu = {np.round(pf['mu'],6)}")
print(f"    publicado     I_ref = 0.544 A, D = 0.507")
print(f"    error de la regla del medio ciclo D=1/2: exacto "
      f"{100*abs(pf['D']-0.5)/0.5:.2f} %   publicado "
      f"{100*abs(0.507-0.5)/0.5:.2f} %")

# ---- M3: D en la frontera exacta de rampa (Fig. 1, R = 3 ohm) ----
A_on, b_on, A_off, b_off, _, _ = mats(3.0)
Iref1 = 3.35


def trabajo(ma):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T-ton)
        return [xc[0]-(Iref1-ma*ton), x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, [Iref1-0.35, 8.6, 0.8*T],
                               full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    xc = flw(A_on, b_on, x0, ton)
    fm, fp = A_on@xc + b_on, A_off@xc + b_off
    u = (fp-fm)/(fm[0]+ma)
    S = np.eye(2) + np.outer(u, np.array([1.0, 0.0]))
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    return ton/T, np.sort(np.linalg.eigvals(M).real), fm[0], -fp[0]


ma_ex = brentq(lambda ma: trabajo(ma)[1][0] + 1.0, 20e3, 45e3, xtol=1e-3)
D_ex, mu_ex, m1_ex, m2_ex = trabajo(ma_ex)
print("\nM3  ciclo de trabajo en la frontera exacta de rampa (Fig. 1)")
print(f"    m_a* = {ma_ex/1e3:.4f} kA/s   D = {D_ex:.6f}   "
      f"(el texto dice 0.76, el pie 0.8)")
print(f"    D en los extremos del barrido: "
      f"m_a=5 kA/s -> {trabajo(5e3)[0]:.4f}   "
      f"m_a=60 kA/s -> {trabajo(60e3)[0]:.4f}")
print(f"    regla clasica (m2-m1)/2 = {(m2_ex-m1_ex)/2/1e3:.4f} kA/s")
