"""Variante de signatura: en la metrica de la energia E = (L i^2 + C v^2)/2 los flujos de intervalo
son casi rotaciones euclideas. Descomposicion polar Phi~ = R P (R ortogonal, P simetrica positiva):
angulo de R = intercambio conservativo L-C;  log de los valores singulares de P = disipacion.
Plataforma de la Sec. V-A en la frontera de flip (m_a = 34.038 kA/s)."""
import numpy as np
from scipy.linalg import expm, polar, svdvals
from scipy.optimize import fsolve
Vin, L, RL, C, ESR, R, Ron, Vd, Rd, T, Iref, ma = 12.0, 100e-6, 0.060, 100e-6, 0.025, 3.0, 0.040, 0.45, 0.020, 1e-5, 3.35, 34038.0566
al = R/(R + ESR)
A2 = lambda Rsw: np.array([[-(Rsw + RL + al*ESR)/L, -al/L], [al/C, -al/(R*C)]])
Aon, Aoff = A2(Ron), A2(Rd); bon, boff = np.array([Vin/L, 0.0]), np.array([-Vd/L, 0.0])
def fl(A, b, x, t):
    Z = np.zeros((3, 3)); Z[:2, :2] = A; Z[:2, 2] = b; return (expm(Z*t) @ np.append(x, 1.0))[:2]
def res(p):
    xs = fl(Aon, bon, p[:2], p[2]*T); xT = fl(Aoff, boff, xs, (1-p[2])*T)
    return [xT[0]-p[0], xT[1]-p[1], xs[0] - Iref + ma*p[2]*T]
p = fsolve(res, [2.5, 5.0, 0.7], xtol=1e-13); d = p[2]; xs = fl(Aon, bon, p[:2], d*T)
n = np.array([1.0, 0.0]); fm, fp = Aon@xs+bon, Aoff@xs+boff; u = (fp-fm)/(n@fm + ma); sigma = n@u
W = np.diag([np.sqrt(L), np.sqrt(C)]); Wi = np.linalg.inv(W)          # coordenadas de energia z = W x
print(f"D = {d:.4f}, 1+sigma = {1+sigma:+.4f}")
tot_rot = 0.0; tot_dis = 0.0; desv = []
for nombre, A, tau in (("on ", Aon, d*T), ("off", Aoff, (1-d)*T)):
    Ph = W @ expm(A*tau) @ Wi; Rm, P = polar(Ph); ang = np.arctan2(Rm[1, 0], Rm[0, 0]); sv = svdvals(Ph)
    tot_rot += ang; tot_dis += np.log(sv).sum(); desv.append(np.linalg.norm(Ph - Rm, 2))
    print(f"flujo {nombre}: angulo de rotacion {ang:+.5f} rad   rapideces de disipacion ln(s) = {np.round(np.log(sv), 5)}   "
          f"||Phi~ - R||_2 = {np.linalg.norm(Ph - Rm, 2):.4f}   tr(A) tau = {np.trace(A)*tau:+.5f}")
St = W @ (np.eye(2) + np.outer(u, n)) @ Wi; sv = svdvals(St)
print(f"saltacion: valores singulares {np.round(sv, 4)}  ln = {np.round(np.log(sv), 4)}   suma = ln|1+sigma| = {np.log(abs(1+sigma)):+.4f}   ||S~ - I||_2 = {np.linalg.norm(St - np.eye(2), 2):.3f}")
assert max(desv) < 0.026 and abs(tot_dis - (np.trace(Aon)*d*T + np.trace(Aoff)*(1-d)*T)) < 1e-12
print(f"presupuesto: disipacion de los flujos {tot_dis:+.5f}  +  saltacion {np.log(abs(1+sigma)):+.5f}  =  ln|det M| = {tot_dis + np.log(abs(1+sigma)):+.5f};  rotacion total {tot_rot:+.4f} rad, que no entra en el presupuesto")

print("METRICA DE LA ENERGIA VERIFICADA: los flujos distan menos del 2.6 % de una rotacion")
