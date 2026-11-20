"""Utilidades compartidas por la campaña de validación.

Un solo sitio donde se construyen las saltaciones, los versores y la
monodromía, para que los tres experimentos midan exactamente lo mismo.
"""
import numpy as np
from scipy.linalg import expm

# ---------------------------------------------------------------- matrices --

def salt_desde_sigma(rng, n, sigma):
    """Saltacion S = I + u n^T con n^T u = sigma exacto, u y n genericos
    (sin alinear con ejes). Devuelve (S, u, nv)."""
    while True:
        u = rng.normal(size=n)
        nv = rng.normal(size=n)
        d = nv @ u
        if abs(d) > 1e-6:
            break
    if abs(sigma) < 1e-300:              # caso parabolico: u en ker n^T
        u = u - nv*(nv @ u)/(nv @ nv)
    else:
        u = u*(sigma/d)
    return np.eye(n) + np.outer(u, nv), u, nv


def monodromia(rng, n, sigmas, esc_A=0.7, tau_max=1.0):
    """M = Phi_m S_K Phi_{m-1} ... S_1 Phi_0 con matrices de intervalo
    aleatorias. Devuelve (M, lista_de_(S,u,nv), lista_de_(A,tau))."""
    K = len(sigmas)
    salts = [salt_desde_sigma(rng, n, s) for s in sigmas]
    flujos = []
    for _ in range(K+1):
        A = rng.normal(size=(n, n))*esc_A/np.sqrt(n)
        tau = rng.uniform(0.05, tau_max)
        flujos.append((A, tau))
    M = expm(flujos[0][0]*flujos[0][1])
    for k in range(K):
        M = expm(flujos[k+1][0]*flujos[k+1][1]) @ salts[k][0] @ M
    return M, salts, flujos


def n_reales_negativos(M, tol=1e-11):
    mu = np.linalg.eigvals(M)
    return int(np.sum((np.abs(mu.imag) < tol) & (mu.real < -tol)))


def N_r(sigmas, tol=1e-300):
    """Numero de eventos tipo reflexion: det S = 1+sigma < 0."""
    return int(sum(1 for s in sigmas if 1.0 + s < 0.0))


# ------------------------------------------------------------------ algebra --

_CACHE_ALG = {}


def algebra(n):
    """Cacheado: construir Cl(n,n) cuesta lo suyo (2^{2n} blades) y llamarlo
    dentro de un bucle domina el tiempo de toda la campana."""
    if n in _CACHE_ALG:
        return _CACHE_ALG[n]
    from clifford import Cl
    layout, blades = Cl(n, n)
    E = [blades['e%d' % k] for k in range(1, 2*n+1)]
    w = [(E[i] + E[i+n])/2 for i in range(n)]
    v = [(E[i] - E[i+n])/2 for i in range(n)]
    G = dict(n=n, layout=layout, w=w, v=v, uno=1 + 0*E[0])
    _CACHE_ALG[n] = G
    return G


def mvexp(B, uno, N=60):
    """Exponencial de multivector con escalado y cuadrado: la serie directa
    pierde precision en cuanto ||B|| crece, y aqui B viene de A*tau."""
    nrm = float(np.max(np.abs(B.value)))
    s = max(0, int(np.ceil(np.log2(nrm))) + 1) if nrm > 1 else 0
    Bs = B/(2**s)
    tot, term = uno + 0*B, uno + 0*B
    for k in range(1, N):
        term = term*Bs/k
        tot = tot + term
        if float(np.max(np.abs(term.value))) < 1e-18:
            break
    for _ in range(s):
        tot = tot*tot
    return tot


def biv_matriz(A, G):
    """B_A = 2 sum_ij A_ij w_i ^ v_j."""
    n = G['n']
    B = 0*G['w'][0]
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0.0:
                B = B + 2*float(A[i, j])*(G['w'][i] ^ G['v'][j])
    return B


def versor_salt(u, nv, G):
    """Versor de una saltacion segun la Prop. 1. Devuelve (W, especie) o
    (None, 'clamping') si 1+sigma = 0, donde no existe versor."""
    n = G['n']
    U = sum(float(u[i])*G['w'][i] for i in range(n))
    V = sum(float(nv[j])*G['v'][j] for j in range(n))
    B = 2*(U ^ V)
    s = float(nv @ u)
    if abs(1.0 + s) < 1e-14:
        return None, 'clamping'
    if s == 0.0:
        # UNICA rama que hace falta, y no es un margen de seguridad: con
        # sigma = 0 exacto la forma cerrada da 0/0. Ocurre de verdad, en todo
        # evento disparado por tiempo (ver exp5_retardo_digital.py).
        return G['uno'] + B/2, 'parabolico'
    Bh = B/s
    # log1p(s) en vez de log(1+s): al formar 1+s se pierden los digitos de s
    # cuando s es diminuto, que es justo donde vive la singularidad removible.
    # A sigma = 1e-15, log(1+s)/s da 1.1102 y log1p(s)/s da 0.999999999999999.
    lam = np.log1p(s) if 1.0 + s > 0.0 else np.log(-(1.0 + s))
    if 1.0 + s > 0:
        return mvexp(lam*Bh/2, G['uno']), 'hiperbolico'
    return Bh*mvexp(lam*Bh/2, G['uno']), 'reflexion'


def accion(W, G):
    """Matriz de la accion sandwich X -> W X W^-1 sobre el cono de las w."""
    n = G['n']
    c = float((W*~W)(0))
    Wi = ~W/c
    return np.column_stack([
        np.array([2*float((G['v'][i] | (W*G['w'][k]*Wi))(0)) for i in range(n)])
        for k in range(n)])


def versor_monodromia(salts, flujos, G):
    """W = R_m W_{S_K} R_{m-1} ... W_{S_1} R_0. None si algun evento es
    clamping (el versor no existe)."""
    W = mvexp(biv_matriz(flujos[0][0]*flujos[0][1], G)/2, G['uno'])
    for k, (S, u, nv) in enumerate(salts):
        Wk, esp = versor_salt(u, nv, G)
        if Wk is None:
            return None
        W = mvexp(biv_matriz(flujos[k+1][0]*flujos[k+1][1], G)/2,
                  G['uno'])*Wk*W
    return W
