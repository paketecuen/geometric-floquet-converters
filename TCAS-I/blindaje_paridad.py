import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve
from clifford import Cl

np.set_printoptions(precision=6, suppress=True)

# ================= G(n,n) generico =================
def alg(n):
    layout, blades = Cl(n, n)
    E = [blades['e%d' % k] for k in range(1, 2*n+1)]
    w = [(E[i] + E[i+n])/2 for i in range(n)]
    v = [(E[i] - E[i+n])/2 for i in range(n)]
    return dict(n=n, E=E, w=w, v=v)

def biv(a, G):
    B = 0*G['E'][0]
    for i in range(G['n']):
        for j in range(G['n']):
            if a[i, j] != 0:
                B = B + 2*a[i, j]*(G['w'][i] ^ G['v'][j])
    return B

def mvexp(B, N=120):
    tot, term = 1 + 0*B, 1 + 0*B
    for k in range(1, N):
        term = term*B/k
        tot = tot + term
        if max(abs(term.value)) < 1e-18:
            break
    return tot

def inv(W):
    c = float((W*~W)(0))
    return ~W/c

def accion(W, G):
    Wi = inv(W)
    n = G['n']
    return np.column_stack([np.array([2*float((G['v'][i] | (W*G['w'][k]*Wi))(0))
                                      for i in range(n)]) for k in range(n)])

def versor_salt(S, u, nv, eje, G):
    s = float(nv @ u)
    if 1 + s > 0:
        m = np.log(1+s)/s * np.outer(u, nv)
        return mvexp(biv(m, G)/2)
    g = np.zeros((G['n'], G['n'])); g[eje, eje] = np.log(-(1+s))
    return (G['E'][eje]*G['E'][eje+G['n']]) * mvexp(biv(g, G)/2)

Vin, C, R, fs = 12.0, 100e-6, 3.0, 100e3
T = 1.0/fs

# ================= TEST 1: 2 fases ASIMETRICAS (rompe Z2) =================
L1, L2 = 100e-6, 130e-6
Ir1, Ir2 = 0.95, 0.80
A2 = np.array([[0, 0, -1/L1], [0, 0, -1/L2], [1/C, 1/C, -1/(R*C)]])
def b2(u1, u2): return np.array([Vin*u1/L1, Vin*u2/L2, 0.0])

def flw(A, x, t, bb):
    M = np.zeros((A.shape[0]+1,)*2); M[:-1, :-1] = A; M[:-1, -1] = bb
    E = expm(M*t)
    return E[:-1, :-1]@x + E[:-1, -1]

def res2(z):
    x0, tc1, tc2 = z[:3], z[3], z[4]
    xa = flw(A2, x0, tc1, b2(1, 0))
    xb = flw(A2, xa, T/2-tc1, b2(0, 0))
    xc = flw(A2, xb, tc2-T/2, b2(0, 1))
    xT = flw(A2, xc, T-tc2, b2(0, 0))
    return [xa[0]-Ir1, xc[1]-Ir2, *(xT-x0)]

z, info, ier, msg = fsolve(res2, [0.6, 0.6, 4.5, 0.4*T, 0.9*T], full_output=True)
assert ier == 1, msg
x0, tc1, tc2 = z[:3], z[3], z[4]
xc1 = flw(A2, x0, tc1, b2(1, 0))
fm1 = A2@xc1 + b2(1, 0); fp1 = A2@xc1 + b2(0, 0)
n1 = np.array([1., 0., 0.]); u1 = (fp1-fm1)/(n1@fm1); s1 = n1@u1
S1 = np.eye(3) + np.outer(u1, n1)
xc2 = flw(A2, flw(A2, xc1, T/2-tc1, b2(0, 0)), tc2-T/2, b2(0, 1))
fm2 = A2@xc2 + b2(0, 1); fp2 = A2@xc2 + b2(0, 0)
n2 = np.array([0., 1., 0.]); u2 = (fp2-fm2)/(n2@fm2); s2 = n2@u2
S2 = np.eye(3) + np.outer(u2, n2)
M2 = expm(A2*(T-tc2)) @ S2 @ expm(A2*(tc2-tc1)) @ S1 @ expm(A2*tc1)
mu2 = np.linalg.eigvals(M2)

G3 = alg(3)
W2 = (mvexp(biv(A2*(T-tc2), G3)/2) * versor_salt(S2, u2, n2, 1, G3) *
      mvexp(biv(A2*(tc2-tc1), G3)/2) * versor_salt(S1, u1, n1, 0, G3) *
      mvexp(biv(A2*tc1, G3)/2))
print("TEST 1 - interleaved 2 fases ASIMETRICO (L2 = 1.3 L1, Iref distintos):")
print(f"  D1 = {tc1/T:.4f}  D2 = {(tc2-T/2)/T:.4f}  (sin simetria Z2)")
print(f"  1+s1 = {1+s1:.4f}   1+s2 = {1+s2:.4f}   (ya no degenerados)")
print(f"  mu = {np.sort_complex(mu2)}")
print(f"  det M = {np.linalg.det(M2):.4f}  (>0)   W~W = {float((W2*~W2)(0)):+.4f}")
print(f"  |accion - M| = {np.max(np.abs(accion(W2, G3)-M2)):.2e}")

# ================= TEST 2: 3 fases simetricas, D<1/3 (paridad IMPAR) =================
L = 100e-6
Ir = 0.53
A3 = np.array([[0, 0, 0, -1/L], [0, 0, 0, -1/L], [0, 0, 0, -1/L],
               [1/C, 1/C, 1/C, -1/(R*C)]])
def b3(u): return np.array([Vin*u[0]/L, Vin*u[1]/L, Vin*u[2]/L, 0.0])

def tray3(x0, tcs, tfin):
    """Segmentos: [0,tc1] f1ON, [tc1,T/3] off, [T/3,tc2] f2ON, [tc2,2T/3] off,
       [2T/3,tc3] f3ON, [tc3,T] off."""
    hitos = [(tcs[0], (1, 0, 0)), (T/3, (0, 0, 0)), (tcs[1], (0, 1, 0)),
             (2*T/3, (0, 0, 0)), (tcs[2], (0, 0, 1)), (T, (0, 0, 0))]
    t, x = 0.0, x0.copy()
    for tf, u in hitos:
        if tfin <= tf:
            return flw(A3, x, tfin-t, b3(u))
        x = flw(A3, x, tf-t, b3(u)); t = tf
    return x

def res3(z):
    x0, tcs = z[:4], z[4:]
    r = [tray3(x0, tcs, tcs[k])[k] - Ir for k in range(3)]
    return r + list(tray3(x0, tcs, T) - x0)

z0 = [0.28, 0.40, 0.52, 3.6, 3.0e-6, 6.4e-6, 9.7e-6]
z, info, ier, msg = fsolve(res3, z0, full_output=True)
assert ier == 1, msg
x0, tcs = z[:4], z[4:]

Ss, us, ns = [], [], []
for k in range(3):
    xck = tray3(x0, tcs, tcs[k])
    u_on = [0, 0, 0]; u_on[k] = 1
    fm = A3@xck + b3(u_on); fp = A3@xck + b3((0, 0, 0))
    nk = np.zeros(4); nk[k] = 1.0
    uk = (fp-fm)/(nk@fm)
    Ss.append(np.eye(4) + np.outer(uk, nk)); us.append(uk); ns.append(nk)

M3 = (expm(A3*(T-tcs[2])) @ Ss[2] @ expm(A3*(tcs[2]-tcs[1])) @ Ss[1] @
      expm(A3*(tcs[1]-tcs[0])) @ Ss[0] @ expm(A3*tcs[0]))
mu3 = np.linalg.eigvals(M3)

G4 = alg(4)
W3 = (mvexp(biv(A3*(T-tcs[2]), G4)/2) * versor_salt(Ss[2], us[2], ns[2], 2, G4) *
      mvexp(biv(A3*(tcs[2]-tcs[1]), G4)/2) * versor_salt(Ss[1], us[1], ns[1], 1, G4) *
      mvexp(biv(A3*(tcs[1]-tcs[0]), G4)/2) * versor_salt(Ss[0], us[0], ns[0], 0, G4) *
      mvexp(biv(A3*tcs[0], G4)/2))
print()
print("TEST 2 - interleaved 3 fases (paridad IMPAR de eventos reflexion):")
print(f"  D = {tcs[0]/T:.4f}   cruces en {np.array(tcs)*1e6} us")
print(f"  1+s por fase = {[f'{1+n@u:.4f}' for n, u in zip(ns, us)]}")
print(f"  mu = {np.sort_complex(mu3)}")
print(f"  det M = {np.linalg.det(M3):.4f}  (<0: multiplicador real negativo garantizado)")
print(f"  W~W = {float((W3*~W3)(0)):+.4f}   <- (-1)^3: canal flip reabierto")
print(f"  |accion - M| = {np.max(np.abs(accion(W3, G4)-M3)):.2e}")
