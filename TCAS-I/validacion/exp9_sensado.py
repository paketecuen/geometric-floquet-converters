"""Exp. 9 — Dinamica de SENSADO: el recuento N_r depende del nivel de detalle del modelo.

Buck de la Sec. V-A (R = 3 ohm, Iref = 3.35 A, m_a = 50 kA/s) con el comparador viendo la corriente
a traves de un filtro de sensado de primer orden:

    d i_f/dt = w_f (i_L - i_f),     h = i_f - (Iref - m_a t),     n = (0, 0, 1)

La derivada de i_f no salta en la conmutacion (i_L es continua), luego h'+ = h'- y det S = 1 exacto:
el evento con latch deja de ser de reflexion y pasa a ser de paso con sigma = 0, N_r = 0 y det M > 0,
para CUALQUIER ancho de banda del filtro. El multiplicador de flip del modelo de dos estados persiste
y el signo negativo lo lleva un multiplicador rapido, del orden de exp(-w_f T).

Se comprueba ademas hasta donde la matriz ensamblada conserva esos invariantes: signo y logaritmo de
det M por LU frente a los valores por factores (recuento y suma de rapideces).

Salida: resultados/exp9.json
"""
import json, pathlib, sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve

RES = pathlib.Path(__file__).resolve().parent / 'resultados'; RES.mkdir(exist_ok=True)
Vin, L, RL, C, ESR, R, Ron, Vd, Rd, T, Iref, ma = 12.0, 100e-6, 0.060, 100e-6, 0.025, 3.0, 0.040, 0.45, 0.020, 1e-5, 3.35, 50e3
al = R/(R + ESR)
A2 = lambda Rsw: np.array([[-(Rsw + RL + al*ESR)/L, -al/L], [al/C, -al/(R*C)]])


def flujo(A, b, x, t):
    n = len(x); Z = np.zeros((n + 1, n + 1)); Z[:n, :n] = A; Z[:n, n] = b
    return (expm(Z*t) @ np.append(x, 1.0))[:n]


# ---- modelo de dos estados: sensado ideal ----
Aon2, Aoff2 = A2(Ron), A2(Rd); bon2, boff2 = np.array([Vin/L, 0.0]), np.array([-Vd/L, 0.0])
def res2(p):
    xs = flujo(Aon2, bon2, p[:2], p[2]*T); xT = flujo(Aoff2, boff2, xs, (1 - p[2])*T)
    return [xT[0] - p[0], xT[1] - p[1], xs[0] - Iref + ma*p[2]*T]
p2 = fsolve(res2, [2.5, 5.0, 0.5], xtol=1e-13); x02, d2 = p2[:2], p2[2]
xs = flujo(Aon2, bon2, x02, d2*T); n2 = np.array([1.0, 0.0]); fm, fp = Aon2 @ xs + bon2, Aoff2 @ xs + boff2
S2 = np.eye(2) + np.outer(fp - fm, n2)/(n2 @ fm + ma); M2 = expm(Aoff2*(1 - d2)*T) @ S2 @ expm(Aon2*d2*T)
mu2 = np.sort(np.linalg.eigvals(M2).real)
print(f"sensado ideal (2 estados): D = {d2:.4f}, det S = {np.linalg.det(S2):+.4f} (reflexion, N_r = 1), det M = {np.linalg.det(M2):+.4f}, mu = {np.round(mu2, 4)}\n")
assert np.linalg.det(S2) < 0 and np.linalg.det(M2) < 0

print(" f_f/f_sw   det S  N_r | ln|det M| por factores  signo | det M por LU   signo | mu de flip   mu de tension | mu rapido: por factores / eig")
filas = []
for ratio in (0.5, 1, 2, 5, 8, 12, 20, 50):
    wf = 2*np.pi*ratio/T
    def A3(Rsw):
        A = np.zeros((3, 3)); A[:2, :2] = A2(Rsw); A[2, 0] = wf; A[2, 2] = -wf; return A
    Aon, Aoff = A3(Ron), A3(Rd); bon, boff = np.array([Vin/L, 0, 0.0]), np.array([-Vd/L, 0, 0.0])
    def res(p):
        xs = flujo(Aon, bon, p[:3], p[3]*T); xT = flujo(Aoff, boff, xs, (1 - p[3])*T)
        return [xT[0] - p[0], xT[1] - p[1], xT[2] - p[2], xs[2] - Iref + ma*p[3]*T]
    p, info, ier, msg = fsolve(res, [x02[0], x02[1], x02[0], d2], xtol=1e-13, full_output=True)
    assert ier == 1 and 0.02 < p[3] < 0.98, (ratio, msg)
    d = p[3]; xs = flujo(Aon, bon, p[:3], d*T); n = np.array([0, 0, 1.0]); fm, fp = Aon @ xs + bon, Aoff @ xs + boff
    hm = n @ fm + ma; sigma = n @ (fp - fm)/hm
    M = expm(Aoff*(1 - d)*T) @ (np.eye(3) + np.outer(fp - fm, n)/hm) @ expm(Aon*d*T)
    Nr = int(1 + sigma < 0); lnfac = np.trace(Aon)*d*T + np.trace(Aoff)*(1 - d)*T + np.log(abs(1 + sigma))
    detLU = np.linalg.det(M); mu = np.linalg.eigvals(M); mu = mu[np.argsort(-np.abs(mu))]
    assert abs(mu[0].imag) < 1e-9 and abs(mu[1].imag) < 1e-9
    lentos = np.sort(mu[:2].real); rapido_fac = np.exp(lnfac)/(lentos[0]*lentos[1])
    print(f"{ratio:8.1f}  {1 + sigma:+.4f}  {Nr}  | {lnfac:14.3f}           +   | {detLU:+10.2e}    {'+' if detLU > 0 else '-'}{' MAL' if detLU <= 0 else '    '} "
          f"| {lentos[0]:+.4f}     {lentos[1]:+.4f}      | {rapido_fac:+.2e} / {mu[2].real:+.2e}")
    assert abs(sigma) < 1e-12 and Nr == 0                      # el evento es de paso a cualquier ancho de banda
    assert lentos[0] < 0 < lentos[1] and rapido_fac < 0        # dos multiplicadores reales negativos: paridad par
    if ratio >= 5: assert abs(lentos[0] - mu2[0]) < 0.01 and abs(lentos[1] - mu2[1]) < 0.001
    filas.append(dict(ratio=ratio, uno_sigma=float(1 + sigma), N_r=Nr, ln_det_factores=float(lnfac), det_LU=float(detLU),
                      signo_LU_correcto=bool(detLU > 0), mu_flip=float(lentos[0]), mu_tension=float(lentos[1]),
                      mu_rapido_factores=float(rapido_fac), mu_rapido_eig=float(mu[2].real)))
print("\nEl evento es de paso (sigma = 0, N_r = 0) con cualquier filtro; el multiplicador de flip persiste y el signo")
print("lo lleva el multiplicador rapido. Los invariantes por factores son exactos a cualquier rigidez.")
json.dump(dict(dos_estados=dict(D=float(d2), detS=float(np.linalg.det(S2)), detM=float(np.linalg.det(M2)), mu=list(map(float, mu2))), filas=filas),
          open(RES / 'exp9.json', 'w'), indent=1)
print("resultados/exp9.json escrito"); sys.exit(0)
