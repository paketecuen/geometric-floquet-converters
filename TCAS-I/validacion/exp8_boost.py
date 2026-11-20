"""Exp. 8 — BOOST con control por corriente de pico (topologia distinta de todas las anteriores).

En el boost la conmutacion cambia las DOS ecuaciones de estado (el inductor deja de ver la salida y
el condensador deja de recibir la corriente del inductor), asi que el salto de campo u_k no es
paralelo a ningun eje ya en n = 2, y A_on y A_off son estructuralmente distintas.

    ON :  L i' = Vin - (RL + Ron) i                      C v' = -al v / R
    OFF:  L i' = Vin - Vd - (RL + Rd + al ESR) i - al v   C v' = al i - al v / R        al = R/(R + ESR)

Evento: h = i - (Iref - m_a t), n = (1, 0), con latch.

Se comprueba: (1) u_k oblicuo y det S = h'+/h'-; (2) paridad y presupuesto; (3) la frontera de flip
como raiz del escalar det(I + M) = 4 <Q_M>_0; (4) el hueco con la regla clasica en forma cerrada,
m_a* = (m2 - kappa m1)/(1 + kappa); (5) esa frontera frente a una simulacion en el dominio del tiempo
independiente (amplitud^2 del ciclo de periodo 2 extrapolada a cero).

Salida: resultados/exp8.json
"""
import json, pathlib, sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

RES = pathlib.Path(__file__).resolve().parent / 'resultados'; RES.mkdir(exist_ok=True)
Vin, L, RL, C, ESR, R, Ron, Vd, Rd, T, Iref = 5.0, 100e-6, 0.060, 100e-6, 0.025, 10.0, 0.040, 0.45, 0.020, 1e-5, 3.0
al = R/(R + ESR)
A_on = np.array([[-(RL + Ron)/L, 0.0], [0.0, -al/(R*C)]]); b_on = np.array([Vin/L, 0.0])
A_off = np.array([[-(RL + Rd + al*ESR)/L, -al/L], [al/C, -al/(R*C)]]); b_off = np.array([(Vin - Vd)/L, 0.0])
n = np.array([1.0, 0.0])


def flujo(A, b, x, t):
    Z = np.zeros((3, 3)); Z[:2, :2] = A; Z[:2, 2] = b
    return (expm(Z*t) @ np.append(x, 1.0))[:2]


def punto(ma, guess=(2.3, 11.0, 0.55)):
    def res(p):
        xs = flujo(A_on, b_on, p[:2], p[2]*T); xT = flujo(A_off, b_off, xs, (1 - p[2])*T)
        return [xT[0] - p[0], xT[1] - p[1], xs[0] - Iref + ma*p[2]*T]
    p, info, ier, msg = fsolve(res, guess, xtol=1e-13, full_output=True)
    assert ier == 1 and 0.05 < p[2] < 0.95, (ma, p, msg)
    d = p[2]; xs = flujo(A_on, b_on, p[:2], d*T)
    fm, fp = A_on @ xs + b_on, A_off @ xs + b_off
    hm, hp = n @ fm + ma, n @ fp + ma
    u = (fp - fm)/hm; sigma = n @ u
    S = np.eye(2) + np.outer(u, n)
    M = expm(A_off*(1 - d)*T) @ S @ expm(A_on*d*T)
    return dict(x0=p[:2], d=d, xs=xs, u=u, sigma=sigma, hm=hm, hp=hp, m1=n @ fm, m2=-(n @ fp), S=S, M=M,
                trtau=np.trace(A_on)*d*T + np.trace(A_off)*(1 - d)*T)


# ---- (1) y (2) en un punto estable ----
q = punto(60e3)
un = q['u']/np.linalg.norm(q['u'])
detM = np.linalg.det(q['M']); mu = np.sort(np.linalg.eigvals(q['M']).real)
print(f"m_a = 60 kA/s: D = {q['d']:.4f}, v_o = {al*q['x0'][1]:.2f} V")
print(f"  u = [{q['u'][0]:+.4f}, {q['u'][1]:+.4f}]  (direccion unitaria {np.round(un, 4)}: no alineada con ningun eje)")
print(f"  1+sigma = {1 + q['sigma']:+.5f}  =  h'+/h'- = {q['hp']/q['hm']:+.5f}   -> reflexion, N_r = 1")
print(f"  det M = {detM:+.5f}  (signo (-1)^1);  multiplicadores {np.round(mu, 5)}")
cierre = abs(np.log(abs(detM)) - (q['trtau'] + np.log(abs(1 + q['sigma']))))
print(f"  presupuesto: ln|det M| - [sum tr(A)tau + lambda] = {cierre:.1e}")
assert min(abs(un)) > 0.05 and abs(1 + q['sigma'] - q['hp']/q['hm']) < 1e-12 and detM < 0 and mu[0] < 0 < mu[1] and cierre < 1e-12

# ---- (3) frontera de flip por el escalar ----
f = lambda ma: np.linalg.det(np.eye(2) + punto(ma)['M'])
ma_flip = brentq(f, 5e3, 60e3, xtol=1e-6)
b = punto(ma_flip); mu_b = np.sort(np.linalg.eigvals(b['M']).real)
print(f"\nflip: det(I+M) = 0 en m_a* = {ma_flip/1e3:.4f} kA/s;  D = {b['d']:.4f}, v_o = {al*b['x0'][1]:.2f} V, multiplicadores {np.round(mu_b, 5)}")
assert abs(mu_b[0] + 1) < 1e-7

# ---- (4) regla clasica y hueco en forma cerrada ----
ma_regla_orbita = (b['m2'] - b['m1'])/2
ma_auto = brentq(lambda ma: ma - (punto(ma)['m2'] - punto(ma)['m1'])/2, 5e3, 60e3, xtol=1e-6)
lam_star = np.log(abs(1 + b['sigma'])); lam_pred = np.log(mu_b[1]) - b['trtau']
kappa = np.exp(lam_star); ma_cerrada = (b['m2'] - kappa*b['m1'])/(1 + kappa)
print(f"regla clasica: autoconsistente {ma_auto/1e3:.4f} kA/s; evaluada en la orbita frontera {ma_regla_orbita/1e3:.4f} kA/s")
print(f"rapidez en la frontera lambda* = {lam_star:+.5f} = ln mu+ - sum tr(A)tau = {np.log(mu_b[1]):+.5f} {-b['trtau']:+.5f} = {lam_pred:+.5f}")
print(f"forma cerrada m_a* = (m2 - kappa m1)/(1 + kappa) = {ma_cerrada/1e3:.4f} kA/s;  hueco con la regla en la misma orbita: "
      f"{(ma_regla_orbita - ma_flip)/1e3:+.4f} kA/s = (m1+m2)/2 tanh(lambda*/2) = {0.5*(b['m1'] + b['m2'])*np.tanh(lam_star/2)/1e3:+.4f}")
assert abs(lam_star - lam_pred) < 1e-9 and abs(ma_cerrada - ma_flip) < 1e-3

# ---- (5) simulacion independiente en el dominio del tiempo ----
def autos(A, b):
    lam, Vec = np.linalg.eig(A); return lam, Vec, np.linalg.inv(Vec), -np.linalg.solve(A, b)
EON, EOFF = autos(A_on, b_on), autos(A_off, b_off)


def paso(x, E, t):
    lam, Vec, Vinv, xeq = E
    return (xeq + Vec @ (np.exp(lam*t)*(Vinv @ (x - xeq)))).real


def ciclo(x, ma):
    """Un periodo con la logica real del latch. Devuelve (estado, saturado): saturado indica que el
    comparador no ha actuado dentro del periodo (ON todo el ciclo) o que ya estaba disparado en el
    flanco de reloj (OFF todo el ciclo)."""
    h = lambda t: paso(x, EON, t)[0] - Iref + ma*t
    if h(0.0) >= 0: return paso(x, EOFF, T), True
    if h(T) <= 0: return paso(x, EON, T), True
    lo, hi = 0.0, T
    for _ in range(60):
        mid = 0.5*(lo + hi)
        if h(mid) < 0: lo = mid
        else: hi = mid
    ts = 0.5*(lo + hi)
    return paso(paso(x, EON, ts), EOFF, T - ts), False


def amplitud2(ma, x0, nmax=80000):
    """Amplitud^2 del ciclo de periodo 2 (o 0 si es de periodo 1). Devuelve None si el atractor no es
    un ciclo limpio de periodo 1 o 2 sin ciclos saturados."""
    x = x0.copy(); hist = []; sat = []
    for k in range(nmax):
        x, st = ciclo(x, ma); hist.append(x[0]); sat.append(st)
        if k > 400 and k % 200 == 0:
            ult = np.array(hist[-8:])
            if abs(ult[-1] - ult[-3]) < 1e-13 and abs(ult[-2] - ult[-4]) < 1e-13: break
    ult = np.array(hist[-64:]); pares, impares = ult[0::2], ult[1::2]
    if any(sat[-64:]) or np.ptp(pares) > 1e-8 or np.ptp(impares) > 1e-8: return None, x
    return float((pares.mean() - impares.mean())**2), x


x = b['x0'] + np.array([1e-3, 0.0])
ventana = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12e3
mas_todas = np.linspace(ma_flip - 0.01e3, ma_flip - ventana, 12); mas = []; amp = []
for ma in mas_todas:
    a2, x = amplitud2(ma, x)
    if a2 is not None and a2 > 1e-14: mas.append(ma); amp.append(a2)
mas = np.array(mas); assert len(mas) >= 8, f"solo {len(mas)} ciclos limpios de periodo 2 en la ventana"
coef = np.polyfit(mas, amp, 1); ma_medido = -coef[1]/coef[0]
r2 = 1 - np.sum((np.polyval(coef, mas) - amp)**2)/np.sum((np.array(amp) - np.mean(amp))**2)
a_estable, _ = amplitud2(ma_flip + 0.05e3, b['x0'] + np.array([1e-3, 0.0]))
print(f"\ndominio del tiempo: periodo 1 en m_a* + 0.05 kA/s (amplitud^2 = {a_estable:.1e}); "
      f"{len(mas)} de {len(mas_todas)} puntos son ciclos limpios de periodo 2 en [{mas.min()/1e3:.3f}, {mas.max()/1e3:.3f}] kA/s")
print(f"  ajuste lineal de la amplitud^2 (R^2 = {r2:.5f}) -> frontera medida {ma_medido/1e3:.4f} kA/s frente a {ma_flip/1e3:.4f} del escalar "
      f"({abs(ma_medido - ma_flip)/ma_flip*100:.3f} %);  la regla clasica autoconsistente queda a {abs(ma_auto - ma_flip)/ma_flip*100:.3f} %")
assert a_estable is not None and a_estable < 1e-12 and r2 > 0.999 and abs(ma_medido - ma_flip)/ma_flip < 2e-3

json.dump(dict(ma60=dict(D=float(q['d']), u=list(map(float, q['u'])), uno_sigma=float(1 + q['sigma']), detM=float(detM), mu=list(map(float, mu))),
               ma_flip=float(ma_flip), D_flip=float(b['d']), vo_flip=float(al*b['x0'][1]), mu_flip=list(map(float, mu_b)),
               m1=float(b['m1']), m2=float(b['m2']), ma_regla_autoconsistente=float(ma_auto), ma_regla_en_orbita=float(ma_regla_orbita),
               lam_star=float(lam_star), ln_mu_mas=float(np.log(mu_b[1])), tr_tau=float(b['trtau']), ma_forma_cerrada=float(ma_cerrada),
               ma_medido=float(ma_medido), R2=float(r2), mas=list(map(float, mas)), amplitud2=amp), open(RES / 'exp8.json', 'w'), indent=1)
print("\nresultados/exp8.json escrito"); sys.exit(0)
