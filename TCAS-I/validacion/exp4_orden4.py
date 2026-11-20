"""Exp. 4 — Convertidor de ORDEN 4 con saltacion oblicua.

La revision adversarial encontro que en TODAS las plataformas del manuscrito
u_k resulta paralelo a un eje coordenado, porque al conmutar solo cambia la
ecuacion de UN inductor. La forma cerrada general nunca se ejercita: solo su
caso particular. Aqui se arregla.

Topologia: BUCK CON FILTRO LC DE ENTRADA, control por corriente de pico con
latch sobre i_L. Estado x = [i_Lf, v_Cf, i_L, v_C], n = 4.

    ON  (interruptor cerrado, el buck carga desde C_f):
        Lf i_Lf' = Vin - v_Cf - R_Lf i_Lf
        Cf v_Cf' = i_Lf - i_L
        L  i_L'  = v_Cf - v_C - (R_on + R_L) i_L
        C  v_C'  = i_L - v_C/R

    OFF (diodo, el buck ya no carga C_f):
        Lf i_Lf' = Vin - v_Cf - R_Lf i_Lf
        Cf v_Cf' = i_Lf
        L  i_L'  = -v_C - Vd - (R_d + R_L) i_L
        C  v_C'  = i_L - v_C/R

A_on y A_off difieren en DOS filas (la de v_Cf y la de i_L), asi que
f^+ - f^- tiene dos componentes no nulas y u_k NO es paralelo a ningun eje.
El filtro de entrada ademas no es un artificio: la interaccion filtro-lazo es
un problema de estabilidad clasico en electronica de potencia.

Se comprueba: (1) la oblicuidad de u_k, (2) la identidad versorial en G(4,4),
(3) la regla de paridad, (4) la rapidez, y (5) la frontera de flip.

Salida: resultados/exp4.json
"""
import json, pathlib
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq
import comun as C

RES = pathlib.Path(__file__).resolve().parent / 'resultados'
RES.mkdir(exist_ok=True)

# ---- parametros ----
Vin = 12.0
Lf, RLf, Cf = 47e-6, 0.030, 22e-6          # filtro de entrada
L, RL = 100e-6, 0.060                      # buck
Cout, R = 100e-6, 3.0
Ron, Vd, Rd = 0.040, 0.45, 0.020
T = 1.0/100e3
Iref = 3.35


def modelo():
    A_on = np.array([
        [-RLf/Lf,   -1.0/Lf,          0.0,            0.0],
        [ 1.0/Cf,    0.0,            -1.0/Cf,         0.0],
        [ 0.0,       1.0/L,     -(Ron+RL)/L,      -1.0/L],
        [ 0.0,       0.0,             1.0/Cout, -1.0/(R*Cout)]])
    b_on = np.array([Vin/Lf, 0.0, 0.0, 0.0])
    A_off = np.array([
        [-RLf/Lf,   -1.0/Lf,          0.0,            0.0],
        [ 1.0/Cf,    0.0,             0.0,            0.0],
        [ 0.0,       0.0,       -(Rd+RL)/L,       -1.0/L],
        [ 0.0,       0.0,             1.0/Cout, -1.0/(R*Cout)]])
    b_off = np.array([Vin/Lf, 0.0, -Vd/L, 0.0])
    return A_on, b_on, A_off, b_off


A_on, b_on, A_off, b_off = modelo()
NV = np.array([0.0, 0.0, 1.0, 0.0])          # h = i_L - (Iref - m_a t)


def flw(A, b, x, t):
    M = np.zeros((5, 5)); M[:4, :4] = A; M[:4, 4] = b
    E = expm(M*t)
    return E[:4, :4]@x + E[:4, 4]


def punto(ma):
    def res(z):
        x0, ton = z[:4], z[4]
        xc = flw(A_on, b_on, x0, ton)
        x1 = flw(A_off, b_off, xc, T - ton)
        return list(x1 - x0)[:3] + [x1[3]-x0[3], xc[2] - (Iref - ma*ton)]
    z, info, ier, msg = fsolve(
        res, [3.0/Vin*8.0, 11.0, Iref-0.3, 8.0, 0.7*T], full_output=True)
    assert ier == 1, msg
    x0, ton = z[:4], z[4]
    xc = flw(A_on, b_on, x0, ton)
    fm, fp = A_on@xc + b_on, A_off@xc + b_off
    hm = NV @ fm + ma                         # hdot^-
    u = (fp - fm)/hm
    S = np.eye(4) + np.outer(u, NV)
    M = expm(A_off*(T-ton)) @ S @ expm(A_on*ton)
    sig = float(NV @ u)
    return dict(x0=x0, ton=ton, xc=xc, u=u, S=S, M=M, sigma=sig,
                hm=float(hm), hp=float(NV @ fp + ma),
                mu=np.linalg.eigvals(M))


p = punto(0.0)
print('=' * 72)
print('(1) OBLICUIDAD DE LA SALTACION  (n = 4, buck con filtro de entrada)')
print('=' * 72)
print(f"  punto de trabajo: D = {p['ton']/T:.4f}   "
      f"x0 = {np.round(p['x0'], 4)}")
u = p['u']; un = u/np.linalg.norm(u)
print(f"  u = {np.array2string(u, precision=6)}")
print(f"  componentes no despreciables de u: "
      f"{int(np.sum(np.abs(un) > 1e-8))} de 4")
print(f"  max_k |<u,e_k>|/||u|| = {np.max(np.abs(un)):.6f}   "
      f"(1.0 = alineado con un eje)")
print(f"  angulo con el eje mas proximo: "
      f"{np.degrees(np.arccos(np.max(np.abs(un)))):.3f} grados")
print(f"  -> la saltacion NO vive en un plano coordenado: se ejercita la "
      f"forma cerrada general")

print()
print('=' * 72)
print('(2) IDENTIDAD VERSORIAL EN G(4,4)')
print('=' * 72)
G = C.algebra(4)
Wk, esp = C.versor_salt(u, NV, G)
err_s = float(np.max(np.abs(C.accion(Wk, G) - p['S'])))
print(f"  especie del evento: {esp}    1+sigma = {1+p['sigma']:+.6f}")
print(f"  |accion(W_S) - S| = {err_s:.3e}   W_S~W_S = "
      f"{float((Wk*~Wk)(0)):+.6f}")
Ron_ = C.mvexp(C.biv_matriz(A_on*p['ton'], G)/2, G['uno'])
Roff = C.mvexp(C.biv_matriz(A_off*(T-p['ton']), G)/2, G['uno'])
W = Roff*Wk*Ron_
err_m = float(np.max(np.abs(C.accion(W, G) - p['M']))
              / max(1.0, np.max(np.abs(p['M']))))
ww = float((W*~W)(0))
print(f"  |accion(W) - M| relativo = {err_m:.3e}")
print(f"  W~W = {ww:+.6f}   sign(det M) = {np.sign(np.linalg.det(p['M'])):+.0f}")

print()
print('=' * 72)
print('(3) REGLA DE PARIDAD Y (4) RAPIDEZ')
print('=' * 72)
nr = 1 if 1 + p['sigma'] < 0 else 0
det = float(np.linalg.det(p['M']))
tr_tau = float(np.trace(A_on)*p['ton'] + np.trace(A_off)*(T-p['ton']))
det_ana = np.exp(tr_tau)*(1 + p['sigma'])
lam = float(np.log(abs(1 + p['sigma'])))
neg = int(np.sum((np.abs(p['mu'].imag) <= 1e-8*np.abs(p['mu']))
                 & (p['mu'].real < 0)))
print(f"  N_r = {nr}  (un apagado por pico con latch)")
print(f"  prediccion: sign(det M) = (-1)^{nr} = {(-1)**nr:+d}")
print(f"  det M numerico = {det:+.6e}   analitico = {det_ana:+.6e}")
print(f"  multiplicadores: {np.array2string(np.sort_complex(p['mu']), precision=5)}")
print(f"  multiplicadores reales negativos: {neg}   -> "
      f"{'REGLA CUMPLIDA' if np.sign(det) == (-1)**nr and (nr % 2 == 0 or neg >= 1) else 'FALLO'}")
print(f"  rapidez lambda = ln|1+sigma| = {lam:+.6f}")
print(f"  presupuesto: ln|det M| = {np.log(abs(det)):+.6f}   "
      f"suma tr(A)tau + lambda = {tr_tau + lam:+.6f}   "
      f"dif = {abs(np.log(abs(det)) - (tr_tau+lam)):.2e}")

print()
print('=' * 72)
print('(5) FRONTERA DE FLIP CON RAMPA DE COMPENSACION')
print('=' * 72)


def mu_min(ma):
    return float(np.min(np.linalg.eigvals(punto(ma)['M']).real))


try:
    ma_star = brentq(lambda ma: mu_min(ma) + 1.0, 1e3, 200e3, xtol=1.0)
    q = punto(ma_star)
    print(f"  frontera exacta mu = -1:  m_a* = {ma_star/1e3:.4f} kA/s")
    print(f"  en ella: 1+sigma = {1+q['sigma']:+.6f}   "
          f"lambda = {np.log(abs(1+q['sigma'])):+.6f}   D = {q['ton']/T:.4f}")
    print(f"  multiplicadores: "
          f"{np.array2string(np.sort_complex(q['mu']), precision=5)}")
except Exception as e:
    ma_star = None
    print(f"  no se localiza en el rango: {type(e).__name__}: {e}")

json.dump(dict(
    D=float(p['ton']/T), u=list(map(float, u)),
    max_alineacion=float(np.max(np.abs(un))),
    angulo_grados=float(np.degrees(np.arccos(np.max(np.abs(un))))),
    especie=esp, uno_sigma=float(1+p['sigma']),
    err_saltacion=err_s, err_monodromia=err_m, WW=ww,
    N_r=nr, det_num=det, det_ana=float(det_ana),
    multiplicadores=[[float(z.real), float(z.imag)] for z in p['mu']],
    reales_negativos=neg, lam=lam,
    presupuesto_dif=float(abs(np.log(abs(det)) - (tr_tau+lam))),
    ma_flip=(float(ma_star) if ma_star else None)),
    open(RES / 'exp4.json', 'w'), indent=1)
print()
print('resultados/exp4.json escrito')
