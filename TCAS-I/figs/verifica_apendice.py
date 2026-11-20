"""Verificación numérica de la forma cerrada del Apéndice:
  caso (i)  s=0     : W = 1 + B/2
  caso (ii) 1+s>0   : W = exp(theta*Bhat/2),  Bhat=B/s, theta=ln(1+s)
  caso (iii) 1+s<0  : W = Bhat * exp(lam*Bhat/2),  lam=ln|1+s|
con u, n GENERICOS (sin alinear con ejes), en G(3,3) y G(4,4)."""
import numpy as np
from clifford import Cl

rng = np.random.default_rng(7)

def alg(n):
    layout, blades = Cl(n, n)
    E = [blades['e%d' % k] for k in range(1, 2*n+1)]
    w = [(E[i] + E[i+n])/2 for i in range(n)]
    v = [(E[i] - E[i+n])/2 for i in range(n)]
    return dict(n=n, E=E, w=w, v=v)

def mvexp(B, N=200):
    tot, term = 1 + 0*B, 1 + 0*B
    for k in range(1, N):
        term = term*B/k
        tot = tot + term
        if max(abs(term.value)) < 1e-19:
            break
    return tot

def inv(W):
    c = float((W*~W)(0)); return ~W/c

def accion(W, G):
    Wi = inv(W); n = G['n']
    return np.column_stack([np.array([2*float((G['v'][i] | (W*G['w'][k]*Wi))(0))
                                      for i in range(n)]) for k in range(n)])

def lift_u(u, G): return sum(float(u[i])*G['w'][i] for i in range(G['n']))
def lift_n(nv, G): return sum(float(nv[j])*G['v'][j] for j in range(G['n']))

def testea(n_dim, s_obj, etiqueta):
    G = alg(n_dim)
    # u, n genericos con n^T u = s_obj
    u = rng.normal(size=n_dim); nv = rng.normal(size=n_dim)
    u = u * (s_obj / (nv @ u))              # ajusta la escala para el s deseado
    s = nv @ u
    S = np.eye(n_dim) + np.outer(u, nv)
    U, V = lift_u(u, G), lift_n(nv, G)
    B = 2*(U ^ V)
    # comprobaciones de la Prop. 1
    B2 = float((B*B)(0))
    assert abs((B*B) - B2) < 1e-12, "B^2 no es escalar"   # norma de la parte no escalar
    assert abs(B2 - s**2) < 1e-12, f"B^2={B2} != s^2={s**2}"
    if abs(s) < 1e-14:                       # caso (i)
        W = 1 + B/2
    elif 1 + s > 0:                          # caso (ii)
        Bh = B/s
        W = mvexp(np.log(1+s)*Bh/2)
    else:                                    # caso (iii)
        Bh = B/s
        W = Bh * mvexp(np.log(abs(1+s))*Bh/2)
    err = np.max(np.abs(accion(W, G) - S))
    ww = float((W*~W)(0))
    print(f"{etiqueta}: n={n_dim}  s={s:+.4f}  1+s={1+s:+.4f}  "
          f"|accion(W)-S|={err:.2e}  W~W={ww:+.4f}  detS={np.linalg.det(S):+.4f}")
    assert err < 1e-12
    return ww

# caso (ii): hiperbolico, u y n genericos
testea(3, +0.7, "caso (ii) 1+s>0, s>0 ")
testea(3, -0.4, "caso (ii) 1+s>0, s<0 ")
# caso (iii): reflexion, u y n genericos
ww = testea(3, -1.6706, "caso (iii) 1+s<0     ")
assert abs(ww + 1) < 1e-12
ww = testea(4, -2.977, "caso (iii) 1+s<0, n=4")
assert abs(ww + 1) < 1e-12
# caso (i): parabolico (n^T u = 0), u y n genericos ortogonales
G = alg(3)
u = rng.normal(size=3); nv = rng.normal(size=3)
u = u - nv*(nv@u)/(nv@nv)                    # proyecta: n^T u = 0
S = np.eye(3) + np.outer(u, nv)
U, V = lift_u(u, G), lift_n(nv, G)
B = 2*(U ^ V)
W = 1 + B/2
err = np.max(np.abs(accion(W, G) - S))
print(f"caso (i)  s=0       : n=3  |accion(W)-S|={err:.2e}  W~W={float((W*~W)(0)):+.4f}")
assert err < 1e-12
# composicion: dos reflexiones + un flujo, paridad del producto
A = rng.normal(size=(3, 3))*0.3
Ba = 0*G['E'][0]
for i in range(3):
    for j in range(3):
        Ba = Ba + 2*A[i, j]*(G['w'][i] ^ G['v'][j])
Rf = mvexp(Ba/2)
from scipy.linalg import expm
u1 = rng.normal(size=3); n1 = rng.normal(size=3); u1 = u1*(-1.8/(n1@u1))
u2 = rng.normal(size=3); n2 = rng.normal(size=3); u2 = u2*(-2.5/(n2@u2))
S1, S2 = np.eye(3)+np.outer(u1, n1), np.eye(3)+np.outer(u2, n2)
def Wsalt(u, nv):
    U, V = lift_u(u, G), lift_n(nv, G)
    B = 2*(U ^ V); s = nv@u; Bh = B/s
    return Bh * mvexp(np.log(abs(1+s))*Bh/2)
Wtot = Wsalt(u2, n2) * Rf * Wsalt(u1, n1)
Mtot = S2 @ expm(A) @ S1
err = np.max(np.abs(accion(Wtot, G) - Mtot))
ww = float((Wtot*~Wtot)(0))
print(f"composicion S2*e^A*S1: |accion-M|={err:.2e}  W~W={ww:+.4f}  "
      f"detM={np.linalg.det(Mtot):+.4f}  (-1)^2=+1 OK")
assert err < 1e-11 and abs(ww-1) < 1e-11
print("\nTODAS LAS FORMAS CERRADAS DEL APENDICE VERIFICADAS")
