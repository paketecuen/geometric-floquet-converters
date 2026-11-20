"""Ejemplo completo de dos estados (Sec. III del manuscrito revisado) y parametros de la forma
normal de colision de borde en la frontera CCM-DCM (Sec. IV), ambos en la plataforma de carga
ligera R = 15 ohm, m_a = 0.

(a) Cadena completa en D = 0.40: pendientes -> sigma -> tipo -> presupuesto -> M -> los tres
    escalares del lift sin normalizar -> veredicto de estabilidad, comprobado con autovalores
    y con el producto geometrico en G(2,2).
(b) En la frontera CCM-DCM, la traza y el determinante de la monodromia a cada lado
    (tau_L, delta_L en CCM; tau_R, delta_R en DCM), que son los parametros de la forma normal.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq
from clifford import Cl

Vin, L, RL, C, ESR, R, Ron, Vd, Rd, T = 12.0, 100e-6, 0.060, 100e-6, 0.025, 15.0, 0.040, 0.45, 0.020, 1e-5
al = R/(R + ESR)
matA = lambda Rsw: np.array([[-(Rsw + RL + al*ESR)/L, -al/L], [al/C, -al/(R*C)]])
A_on, b_on, A_off, b_off = matA(Ron), np.array([Vin/L, 0.0]), matA(Rd), np.array([-Vd/L, 0.0])
A_idl = np.array([[0.0, 0.0], [0.0, -al/(R*C)]])


def flw(A, b, x, t):
    Z = np.zeros((3, 3)); Z[:2, :2] = A; Z[:2, 2] = b
    return (expm(Z*t) @ np.append(x, 1.0))[:2]


def ccm(Iref, guess):
    def res(z):
        xc = flw(A_on, b_on, z[:2], z[2]); x1 = flw(A_off, b_off, xc, T - z[2])
        return [xc[0] - Iref, x1[0] - z[0], x1[1] - z[1]]
    z, info, ier, msg = fsolve(res, guess, xtol=1e-13, full_output=True); assert ier == 1
    return z


# ---------------- (a) ejemplo completo en D = 0.40 ----------------
Iref = brentq(lambda I: ccm(I, [0.3, 4.5, 0.4*T])[2]/T - 0.40, 0.3, 0.6, xtol=1e-12)
z = ccm(Iref, [0.3, 4.5, 0.4*T]); x0, ton = z[:2], z[2]; D = ton/T
xc = flw(A_on, b_on, x0, ton); fm, fp = A_on @ xc + b_on, A_off @ xc + b_off
m1, m2 = fm[0], -fp[0]; n = np.array([1.0, 0.0]); u = (fp - fm)/m1; sigma = n @ u; lam = np.log(abs(1 + sigma))
Pon, Poff = expm(A_on*ton), expm(A_off*(T - ton)); S = np.eye(2) + np.outer(u, n); M = Poff @ S @ Pon
tr_tau = np.trace(A_on)*ton + np.trace(A_off)*(T - ton)
print(f"(a) I_ref = {Iref:.4f} A, D = {D:.4f}, v_o = {al*(x0[1]):.3f} V")
print(f"    pendientes: m1 = {m1/1e3:.2f} kA/s, m2 = {m2/1e3:.2f} kA/s;  1+sigma = -m2/m1 = {1 + sigma:+.4f}  -> reflexion, N_r = 1;  lambda = {lam:+.4f}")
print(f"    u = [{u[0]:+.4f}, {u[1]:+.4f}]")
print(f"    presupuesto: sum tr(A)tau = {tr_tau:+.4f};  ln|det M| = {tr_tau + lam:+.4f};  det M = -exp(.) = {-np.exp(tr_tau + lam):+.4f}")
print(f"    Phi_on =\n{np.round(Pon, 4)}\n    Phi_off =\n{np.round(Poff, 4)}\n    S =\n{np.round(S, 4)}\n    M =\n{np.round(M, 4)}")
dM, dp, dm = np.linalg.det(M), np.linalg.det(np.eye(2) + M), np.linalg.det(np.eye(2) - M); mu = np.sort(np.linalg.eigvals(M).real)
print(f"    escalares: Q Q~ = det M = {dM:+.4f};  4<Q>_0 = det(I+M) = {dp:+.4f};  4<Q Gamma>_0 = det(I-M) = {dm:+.4f}")
print(f"    veredicto: det M < 1 ({dM < 1}), det(I+M) > 0 ({dp > 0}), det(I-M) > 0 ({dm > 0})  -> estable;  autovalores {np.round(mu, 4)}")
assert abs(dM + np.exp(tr_tau + lam)) < 1e-12 and dM < 1 and dp > 0 and dm > 0 and np.all(np.abs(mu) < 1)
# el mismo calculo con el producto geometrico en G(2,2)
lay, bl = Cl(2, 2); e = [bl['e%d' % k] for k in range(1, 5)]; w = [(e[0] + e[2])/2, (e[1] + e[3])/2]; v = [(e[0] - e[2])/2, (e[1] - e[3])/2]
uno = 1 + 0*e[0]
def mvexp(B, N=80):
    tot, term = uno + 0*B, uno + 0*B
    for k in range(1, N): term = term*B/k; tot = tot + term
    return tot
biv = lambda A: sum(2*float(A[i, j])*(w[i] ^ v[j]) for i in range(2) for j in range(2))
Qf = lambda A, tau: float(np.exp(np.trace(A)*tau/2))*mvexp(biv(A*tau)/2)
Q = Qf(A_off, T - ton)*(uno + (u[0]*w[0] + u[1]*w[1])*(n[0]*v[0] + n[1]*v[1]))*Qf(A_on, ton)
Gam = (e[0]*e[2])*(e[1]*e[3])
ga = [float((Q*~Q)(0)), 4*float(Q(0)), 4*float((Q*Gam)(0))]
print(f"    con el producto geometrico: {np.round(ga, 6)}  (diferencia maxima con las matrices {max(abs(ga[0]-dM), abs(ga[1]-dp), abs(ga[2]-dm)):.1e})")
assert max(abs(ga[0] - dM), abs(ga[1] - dp), abs(ga[2] - dm)) < 1e-11

# ---------------- (b) forma normal en la frontera CCM-DCM ----------------
def res_borde(z):                                            # incognitas: v0, ton, I_ref; la corriente de valle es nula
    xc = flw(A_on, b_on, np.array([0.0, z[0]]), z[1]); x1 = flw(A_off, b_off, xc, T - z[1])
    return [xc[0] - z[2], x1[0], x1[1] - z[0]]
zb, info, ier, msg = fsolve(res_borde, [0.92, 0.11*T, 0.1216], xtol=1e-14, full_output=True); assert ier == 1, msg
Ib, ton, x0 = zb[2], zb[1], np.array([0.0, zb[0]])
xc = flw(A_on, b_on, x0, ton); fm, fp = A_on @ xc + b_on, A_off @ xc + b_off
Spk = np.eye(2) + np.outer((fp - fm)/fm[0], n)
M_ccm = expm(A_off*(T - ton)) @ Spk @ expm(A_on*ton)
S0 = np.diag([0.0, 1.0])                                   # saltacion singular de corriente nula (el reposo dura cero en el borde)
M_dcm = S0 @ M_ccm
print(f"\n(b) frontera CCM-DCM: I_ref* = {Ib:.5f} A, D = {ton/T:.4f}")
print(f"    lado CCM: tau_L = {np.trace(M_ccm):+.4f}, delta_L = {np.linalg.det(M_ccm):+.4f}, multiplicadores {np.round(np.sort(np.linalg.eigvals(M_ccm).real), 4)}  (N_r = 1: delta_L < 0)")
print(f"    lado DCM: tau_R = {np.trace(M_dcm):+.4f}, delta_R = {np.linalg.det(M_dcm):+.4f}, multiplicadores {np.round(np.sort(np.linalg.eigvals(M_dcm).real), 4)}  (evento singular: delta_R = 0)")
assert np.linalg.det(M_ccm) < 0 and abs(np.linalg.det(M_dcm)) < 1e-14 and abs(np.trace(M_dcm)) < 1 and np.all(np.abs(np.linalg.eigvals(M_ccm)) < 1)
print("    ambos lados son atractores (|tau_R| < 1 y los dos multiplicadores de CCM dentro del circulo): el punto fijo persiste al cruzar el borde")
print("\nEJEMPLO COMPLETO Y FORMA NORMAL VERIFICADOS")
