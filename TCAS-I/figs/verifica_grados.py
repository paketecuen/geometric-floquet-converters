"""Verificación numérica de la Prop. 3 (grados del lift) y de su uso:
  Q_Phi = e^{tr(A tau)/2} e^{B_{A tau}/2},  Q_S = 1 + u n,  Q_M = producto ordenado
  (a) Q_M Q_M~ = det M
  (b) 2^n <Q_M>_0 = det(I+M)
  (c) 2^n <Q_M Gamma>_0 = det(I-M),  Gamma = prod_i e_i^+ e_i^-
  (d) 2^n <Q_M H(z)>_0 = det(I+zM),  H(z) = prod_i [(1+z)/2 + (1-z)/2 e_i^+ e_i^-]
en n = 2, 3, 4 con flujos y saltaciones aleatorias de todas las especies,
incluidas singulares; y (e) la frontera de flip del buck de la Sec. V-A como
raíz de <Q_M>_0, sin autovalores."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'validacion'))
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, fsolve
from comun import algebra, biv_matriz, mvexp, salt_desde_sigma

rng = np.random.default_rng(3)


def Q_flow(A, tau, G):
    return float(np.exp(np.trace(A)*tau/2))*mvexp(biv_matriz(A*tau, G)/2, G['uno'])


def Q_salt(u, nv, G):
    n = G['n']
    U = sum(float(u[i])*G['w'][i] for i in range(n))
    V = sum(float(nv[j])*G['v'][j] for j in range(n))
    return G['uno'] + U*V


peor = 0.0
for n in (2, 3, 4):
    G = algebra(n); E = [G['layout'].blades['e%d' % k] for k in range(1, 2*n+1)]
    Gam = G['uno']
    for i in range(n):
        Gam = Gam*(E[i]*E[i+n])

    def H(z):
        h = G['uno']
        for i in range(n):
            h = h*((1 + z)/2*G['uno'] + (1 - z)/2*(E[i]*E[i+n]))
        return h
    for _ in range(4):
        K = 2
        sig = rng.choice([0.0, 0.6, -0.5, -1.7, 2.0, -1.0], size=K)
        flujos = [(rng.normal(size=(n, n))*0.6/np.sqrt(n), rng.uniform(0.1, 1.0)) for _ in range(K+1)]
        salts = [salt_desde_sigma(rng, n, s) for s in sig]
        M = expm(flujos[0][0]*flujos[0][1]); Q = Q_flow(*flujos[0], G)
        for k in range(K):
            M = expm(flujos[k+1][0]*flujos[k+1][1]) @ salts[k][0] @ M
            Q = Q_flow(*flujos[k+1], G) * Q_salt(salts[k][1], salts[k][2], G) * Q
        ea = abs(float((Q*~Q)(0)) - np.linalg.det(M))
        eb = abs(2**n*float(Q(0)) - np.linalg.det(np.eye(n) + M))
        ec = abs(2**n*float((Q*Gam)(0)) - np.linalg.det(np.eye(n) - M))
        ed = max(abs(2**n*float((Q*H(z))(0)) - np.linalg.det(np.eye(n) + z*M)) for z in (-0.7, 0.3, 2.0))
        peor = max(peor, ea, eb, ec, ed)
        print(f"n={n} sigmas={np.round(sig, 2)}  |QQ~-detM|={ea:.1e}  |2^n<Q>-det(I+M)|={eb:.1e}"
              f"  |2^n<QGam>-det(I-M)|={ec:.1e}  |2^n<QH(z)>-det(I+zM)|={ed:.1e}")
assert peor < 1e-11, peor
print("IDENTIDADES (a)-(d) VERIFICADAS")

# (e) buck realista de la Sec. V-A: frontera de flip como raiz de <Q_M>_0
Vin, L, RL, C, ESR, R, Ron, Vd, Rd, T, Iref = 12.0, 100e-6, 0.060, 100e-6, 0.025, 3.0, 0.040, 0.45, 0.020, 1e-5, 3.35
al = R/(R + ESR)
matA = lambda Rsw: np.array([[-(Rsw + RL + al*ESR)/L, -al/L], [al/C, -al/(R*C)]])
A_on, b_on, A_off, b_off = matA(Ron), np.array([Vin/L, 0.0]), matA(Rd), np.array([-Vd/L, 0.0])


def flw(A, b, x, t):
    Z = np.zeros((3, 3)); Z[:2, :2] = A; Z[:2, 2] = b
    Ez = expm(Z*t); return Ez[:2, :2]@x + Ez[:2, 2]


def orbita(ma):
    def res(z):
        x0, ton = z[:2], z[2]
        xc = flw(A_on, b_on, x0, ton); x1 = flw(A_off, b_off, xc, T - ton)
        return [xc[0] - (Iref - ma*ton), x1[0] - x0[0], x1[1] - x0[1]]
    z, _, ier, msg = fsolve(res, [Iref - 0.35, 8.6, 0.8*T], full_output=True); assert ier == 1, msg
    x0, ton = z[:2], z[2]; xc = flw(A_on, b_on, x0, ton)
    fm, fp = A_on@xc + b_on, A_off@xc + b_off
    u = (fp - fm)/(fm[0] + ma); nv = np.array([1.0, 0.0])
    return expm(A_off*(T - ton)) @ (np.eye(2) + np.outer(u, nv)) @ expm(A_on*ton), u, nv, ton


G2 = algebra(2)


def escalar(ma):
    M, u, nv, ton = orbita(ma)
    return float((Q_flow(A_off, T - ton, G2)*Q_salt(u, nv, G2)*Q_flow(A_on, ton, G2))(0)), M


ma_star = brentq(lambda ma: escalar(ma)[0], 30e3, 40e3, xtol=1e-4)
q0, M = escalar(ma_star)
mu = np.sort(np.linalg.eigvals(M).real)
print(f"raiz de <Q_M>_0: m_a* = {ma_star:.4f} A/s   <Q_M>_0 = {q0:.1e}   multiplicadores alli = {mu}")
assert abs(mu[0] + 1.0) < 1e-6 and abs(ma_star - 34038.06) < 0.5
print("FRONTERA DE FLIP LOCALIZADA SIN AUTOVALORES")
