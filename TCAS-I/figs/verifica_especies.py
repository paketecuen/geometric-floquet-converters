"""Verificacion de los dos hallazgos estructurales de la revision adversarial.

A) La Prop. 2 falla en DCM: hay un evento de pico tipo reflexion (det S<0,
   luego N_r=1 IMPAR) y un evento de corriente nula con det S = 0 EXACTAMENTE,
   de modo que det M = 0 y NO hay ningun multiplicador real negativo. La regla
   sign(det M) = (-1)^{N_r} predice -1: falla.

B) La especie del evento no es una propiedad del detector sino del punto de
   trabajo: hdot^+ = m_a - m_2, asi que el mismo latch de pico pasa de
   reflexion (m_a < m_2) a clamping (m_a = m_2) a pass-through (m_a > m_2) al
   girar el mando de la rampa. Y m_a = m_2 es la condicion clasica de control
   deadbeat, donde un multiplicador se anula.
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


# ============ A) DCM: N_r impar pero det M = 0 y ningun mu < 0 ============
print("=" * 68)
print("A)  DCM  (R = 15 ohm):  N_r = 1 IMPAR,  pero det M = 0")
print("=" * 68)
A_on, b_on, A_off, b_off, A_idl, b_idl = mats(15.0)

for Iref in (0.06, 0.08, 0.10, 0.12):
    def res(z):
        v0, ton, tz = z
        x0 = np.array([0.0, v0])
        xpk = flw(A_on, b_on, x0, ton)
        xz = flw(A_off, b_off, xpk, tz-ton)
        xT = flw(A_idl, b_idl, xz, T-tz)
        return [xpk[0]-Iref, xz[0], xT[1]-v0]
    z, info, ier, msg = fsolve(res, [0.35, 0.5e-6, 8e-6], full_output=True)
    assert ier == 1, msg
    v0, ton, tz = z
    x0 = np.array([0.0, v0])
    xpk = flw(A_on, b_on, x0, ton)
    xz = flw(A_off, b_off, xpk, tz-ton)
    nv = np.array([1.0, 0.0])

    # evento de pico: h = i - Iref
    fm, fp = A_on@xpk + b_on, A_off@xpk + b_off
    u_pk = (fp-fm)/(nv @ fm)
    S_pk = np.eye(2) + np.outer(u_pk, nv)
    # evento de corriente nula: h = i
    gm, gp = A_off@xz + b_off, A_idl@xz + b_idl
    u_0 = (gp-gm)/(nv @ gm)
    S_0 = np.eye(2) + np.outer(u_0, nv)

    M = (expm(A_idl*(T-tz)) @ S_0 @ expm(A_off*(tz-ton)) @ S_pk
         @ expm(A_on*ton))
    mu = np.linalg.eigvals(M)
    n_neg = int(np.sum((abs(mu.imag) < 1e-12) & (mu.real < -1e-12)))
    N_r = int(np.linalg.det(S_pk) < 0) + int(np.linalg.det(S_0) < 0)
    print(f"Iref={Iref:.2f} A | detS_pico={np.linalg.det(S_pk):+.6f} "
          f"detS_cero={np.linalg.det(S_0):+.3e} | N_r={N_r} "
          f"| detM={np.linalg.det(M):+.3e} | mu={np.round(mu.real,6)} "
          f"| mu<0 reales: {n_neg}")
print("\nS_0 en el ultimo caso =\n", np.round(S_0, 12))
print(f"prediccion Prop.2: sign(det M) = (-1)^{N_r} = {(-1)**N_r:+d} "
      f"y >=1 multiplicador real negativo")
print(f"realidad:          sign(det M) =  0   y  {n_neg} multiplicadores "
      f"reales negativos   -> LA PROP. 2 FALLA")

# ============ B) la especie depende de m_a, no del hardware ============
print()
print("=" * 68)
print("B)  La especie del MISMO latch de pico cambia con la rampa m_a")
print("=" * 68)
A_on, b_on, A_off, b_off, _, _ = mats(3.0)
Iref = 3.35


def trabajo(ma):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T-ton)
        return [xc[0]-(Iref-ma*ton), x1[0]-x0[0], x1[1]-x0[1]]
    z, info, ier, msg = fsolve(res, [Iref-0.35, 8.6, 0.6*T],
                               full_output=True)
    assert ier == 1, msg
    x0, ton = z[:2], z[2]
    xc = flw(A_on, b_on, x0, ton)
    fm, fp = A_on@xc + b_on, A_off@xc + b_off
    m1, m2 = fm[0], -fp[0]
    hm, hp = m1 + ma, ma - m2
    u = (fp-fm)/hm
    S = np.eye(2) + np.outer(u, np.array([1.0, 0.0]))
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    return dict(m1=m1, m2=m2, hm=hm, hp=hp, uno_s=hp/hm, D=ton/T,
                detS=np.linalg.det(S), detM=np.linalg.det(M),
                mu=np.sort(np.linalg.eigvals(M).real))


# deadbeat autoconsistente: m_a = m_2(m_a)
ma_db = brentq(lambda ma: trabajo(ma)['m2'] - ma, 60e3, 140e3, xtol=1.0)
print(f"deadbeat autoconsistente  m_a = m_2 :  m_a = {ma_db/1e3:.4f} kA/s")
print(f"{'m_a [kA/s]':>11} {'m_2':>9} {'1+s':>10} {'det S':>10} "
      f"{'det M':>11} {'especie':>13}  multiplicadores")
for ma in (20e3, 34.038e3, 60e3, 80e3, ma_db, 110e3, 130e3):
    p = trabajo(ma)
    esp = ('reflexion' if p['uno_s'] < -1e-9 else
           'clamping' if abs(p['uno_s']) < 1e-9 else 'pass-through')
    marca = '  <-- deadbeat' if abs(ma-ma_db) < 1 else ''
    print(f"{ma/1e3:11.3f} {p['m2']/1e3:9.3f} {p['uno_s']:+10.6f} "
          f"{p['detS']:+10.6f} {p['detM']:+11.3e} {esp:>13}  "
          f"{np.round(p['mu'],6)}{marca}")
print("\n=> N_r NO es un entero del circuito: es funcion del punto de trabajo.")
print("=> det S = 0 <=> m_a = m_2 <=> control deadbeat clasico "
      "(un multiplicador exactamente nulo).")
