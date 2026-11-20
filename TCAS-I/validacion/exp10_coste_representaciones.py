"""Exp. 10 — COSTE del lift segun como se represente, con un unico nucleo compilado.

Responde a la pregunta de si la dimension 4^n del algebra es el coste del metodo. Rutas para el
producto de dos lifts de GL(n) y para el ciclo completo (K = 2 eventos, 3 flujos):

  T0  multivector generico, tabla de Cayley densa            16^n multiplicaciones por producto
  T1  representacion espinorial (Fock), matriz 2^n x 2^n      8^n
  T2  bloques por grado, Q = (+)_k Lambda^k M                 sum_k C(n,k)^3
  T3  versor factorizado: lista de normales de reflexion en R^{2n} (representacion de NuGA);
      una saltacion son dos reflexiones; el ciclo completo cuesta O(n^3)
  T4  matrices n x n                                          n^3

Antes se verifica que T1, T2 y T3 representan el mismo objeto que el lift 1 + u n del articulo.
Las medidas con la biblioteca de A. H. Eid (C# y MATLAB) estan en resultados/exp10_nuga_*.txt y se
generan con los programas de la carpeta nuga/, que necesitan esa biblioteca.

Salida: resultados/exp10.json
"""
import itertools, json, math, pathlib, sys, time
import numpy as np
from numba import njit
from scipy.linalg import expm

RES = pathlib.Path(__file__).resolve().parent / 'resultados'; RES.mkdir(exist_ok=True)


@njit(cache=True)
def mm(A, B):
    n, m, p = A.shape[0], A.shape[1], B.shape[1]; Cm = np.zeros((n, p))
    for i in range(n):
        for k in range(m):
            a = A[i, k]
            for j in range(p): Cm[i, j] += a*B[k, j]
    return Cm


@njit(cache=True)
def rep_mm(A, B, reps):
    s = 0.0
    for _ in range(reps): s += mm(A, B)[0, 0]
    return s


@njit(cache=True)
def cadena(N, X):
    """Aplica las reflexiones (columnas de N, la ultima primero) a las columnas de X; metrica de Witt intercalada."""
    dim, k = N.shape; c = X.shape[1]; Y = X.copy()
    for i in range(k - 1, -1, -1):
        nn = 0.0
        for j in range(0, dim, 2): nn += 2.0*N[j, i]*N[j + 1, i]
        for col in range(c):
            d = 0.0
            for j in range(0, dim, 2): d += Y[j, col]*N[j + 1, i] + Y[j + 1, col]*N[j, i]
            tau = -2.0*d/nn
            for j in range(dim): Y[j, col] += tau*N[j, i]
    return Y


@njit(cache=True)
def ciclo_matriz(P0, P1, P2, u1, n1, u2, n2):
    n = P0.shape[0]; M = P0.copy()
    for (u, nv, P) in ((u1, n1, P1), (u2, n2, P2)):
        r = np.zeros(n)
        for j in range(n):
            for i in range(n): r[j] += nv[i]*M[i, j]
        for i in range(n):
            for j in range(n): M[i, j] += u[i]*r[j]
        M = mm(P, M)
    return M


@njit(cache=True)
def rep_cadena(N, X, reps):
    s = 0.0
    for _ in range(reps): s += cadena(N, X)[0, 0]
    return s


@njit(cache=True)
def rep_ciclo(P0, P1, P2, u1, n1, u2, n2, reps):
    s = 0.0
    for _ in range(reps): s += ciclo_matriz(P0, P1, P2, u1, n1, u2, n2)[0, 0]
    return s


def medir(fun, *args, base=2e7, coste=1.0):
    fun(*args, 1); reps = max(3, int(base/coste)); t0 = time.perf_counter(); fun(*args, reps)
    return (time.perf_counter() - t0)/reps


def Dk(X, n, k):
    """Compuesto aditivo: Lambda^k(e^X) = exp(D_k X) y Lambda^k(I + u n^T) = I + D_k(u n^T)."""
    subs = list(itertools.combinations(range(n), k)); idx = {s: i for i, s in enumerate(subs)}
    D = np.zeros((len(subs), len(subs)))
    for I in subs:
        for pos, i in enumerate(I):
            rest = I[:pos] + I[pos + 1:]
            for j in range(n):
                if j in rest: continue
                J = tuple(sorted(rest + (j,))); D[idx[J], idx[I]] += (-1.0)**(pos - J.index(j))*X[j, i]
    return D


def fock(n):
    N = 2**n; cre = []
    for i in range(n):
        Ad = np.zeros((N, N))
        for m in range(N):
            if not (m >> i) & 1: Ad[m | (1 << i), m] = (-1.0)**bin(m & ((1 << i) - 1)).count('1')
        cre.append(Ad)
    return cre, [A.T.copy() for A in cre]


rng = np.random.default_rng(11); peor = 0.0
print("== verificacion: las tres representaciones son el mismo objeto ==")
for n in (2, 3, 4):
    A0, A1 = rng.normal(size=(n, n))*0.5, rng.normal(size=(n, n))*0.5
    u, nv = rng.normal(size=n), rng.normal(size=n); sigma = -1.7; u *= sigma/(nv @ u)
    S = np.eye(n) + np.outer(u, nv); M = expm(A1) @ S @ expm(A0)
    tr = [np.trace(expm(Dk(A1, n, k)) @ (np.eye(math.comb(n, k)) + Dk(np.outer(u, nv), n, k)) @ expm(Dk(A0, n, k))) for k in range(n + 1)]
    cre, ani = fock(n); gen = lambda A: sum(A[i, j]*cre[i] @ ani[j] for i in range(n) for j in range(n))
    QF = expm(gen(A1)) @ (np.eye(2**n) + sum(u[i]*cre[i] for i in range(n)) @ sum(nv[j]*ani[j] for j in range(n))) @ expm(gen(A0))
    p = np.zeros(2*n); p[0::2] = u; p[1::2] = nv; q = np.zeros(2*n); q[0::2] = u/sigma; q[1::2] = (1 + 1/sigma)*nv
    E = np.zeros((2*n, n)); E[0::2, :] = np.eye(n)
    e = [abs(sum(tr) - np.linalg.det(np.eye(n) + M)), abs(sum((-1)**k*t for k, t in enumerate(tr)) - np.linalg.det(np.eye(n) - M)),
         abs(tr[-1] - np.linalg.det(M)), abs(np.trace(QF) - sum(tr)), np.max(np.abs(cadena(np.column_stack([p, q]), E)[0::2, :] - S)),
         abs(np.sign((p[0::2] @ p[1::2])*(q[0::2] @ q[1::2])) - np.sign(1 + sigma))]
    peor = max(peor, max(e)); print(f"n={n}: errores (det(I+M), det(I-M), det M, traza de Fock, saltacion = 2 reflexiones, signo) = {np.array(e)}")
assert peor < 1e-12

print("\n== coste: multiplicaciones exactas y tiempo con el mismo nucleo ==")
print(" n | por producto: T0=16^n  T1=8^n   T2 bloques  T4=n^3 | T2/T4 | ciclo: T3 versor factorizado  T4 matrices  T3/T4 | tiempos: T2/T4 (producto)  T3/T4 (ciclo)")
filas = []; K = 2
for n in range(2, 9):
    Mn = rng.normal(size=(n, n)); t4 = medir(rep_mm, Mn, Mn, coste=n**3 + 50)
    t2 = sum(medir(rep_mm, B, B, coste=B.shape[0]**3 + 50) for B in (rng.normal(size=(math.comb(n, k),)*2) for k in range(n + 1)))
    k_refl = 2*n*(K + 1) + 2*K                       # cota de Cartan-Dieudonne por rotor (2n) mas dos por saltacion
    Nn = rng.normal(size=(2*n, k_refl)); E = np.zeros((2*n, n)); E[0::2, :] = np.eye(n)
    P = [rng.normal(size=(n, n)) for _ in range(3)]; uu = [rng.normal(size=n) for _ in range(4)]
    t3c = medir(rep_cadena, Nn, E, base=2e6, coste=k_refl*n*4*n/50 + 1); t4c = medir(rep_ciclo, P[0], P[1], P[2], uu[0], uu[1], uu[2], uu[3], base=2e6, coste=n**3/10 + 1)
    m2 = sum(math.comb(n, k)**3 for k in range(n + 1)); m3 = k_refl*n*(4*n + 2); m4c = K*n**3 + K*2*n*n
    filas.append(dict(n=n, T0=16**n, T1=8**n, T2=m2, T4=n**3, T2_T4=m2/n**3, reflexiones=k_refl, T3_ciclo=m3, T4_ciclo=m4c, T3_T4_ciclo=m3/m4c,
                      t_T2_T4=t2/t4, t_T3_T4_ciclo=t3c/t4c))
    print(f"{n:2d} | {16**n:14.3g} {8**n:9d} {m2:9d} {n**3:6d} | {m2/n**3:6.1f} | {m3:20d} {m4c:12d} {m3/m4c:7.1f} | {t2/t4:14.1f} {t3c/t4c:18.1f}")
assert all(9 < f['T3_T4_ciclo'] < 12 for f in filas)          # factor constante: coste polinomico
json.dump(dict(peor_error_verificacion=float(peor), filas=filas), open(RES / 'exp10.json', 'w'), indent=1)
print("\nresultados/exp10.json escrito"); sys.exit(0)
