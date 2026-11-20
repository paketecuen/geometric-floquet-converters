"""Exp. 7 — Buck en MODO TENSION (el circuito de Deane-Hamill, Fossas-Olivar y Giaouris 2008).

Convertidor estructuralmente distinto de los de la campana: no hay latch ni comparador de
corriente. El interruptor esta cerrado mientras a (v - Vref) < v_rampa(t). El evento es el cruce
de la tension de control con la rampa del modulador:

    h(x, t) = a (v - Vref) - VL - (VU - VL) t / T,      n = (0, a),   dh/dt = -(VU - VL)/T

En el buck ideal la conmutacion solo cambia di/dt, asi que n^T (f+ - f-) = 0:
sigma = 0 exacto, el evento es de paso (parabolico), N_r = 0 y

    det M = exp(tr(A) T) = exp(-T / (R C))      en TODO punto de operacion.

Se comprueba: (1) sigma = 0 y det M constante; (2) los multiplicadores viven en el circulo de
radio exp(-T/(2RC)) mientras son complejos; (3) la frontera de flip como raiz del escalar
det(I + M) = 4 <Q_M>_0, sin autovalores; (4) esa frontera frente a una simulacion en el dominio
del tiempo independiente de la construccion de Floquet (amplitud^2 del ciclo de periodo 2,
ajuste lineal extrapolado a cero).

Salida: resultados/exp7.json
"""
import json, pathlib, sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve, brentq

RES = pathlib.Path(__file__).resolve().parent / 'resultados'; RES.mkdir(exist_ok=True)
L, C, R, a, Vref, VL, VU, T = 20e-3, 47e-6, 22.0, 8.4, 11.3, 3.8, 8.2, 400e-6
A = np.array([[0.0, -1/L], [1/C, -1/(R*C)]])


def flujo(b, x, t):
    Z = np.zeros((3, 3)); Z[:2, :2] = A; Z[:2, 2] = b
    return (expm(Z*t) @ np.append(x, 1.0))[:2]


def orbita(Vin, guess=(0.5, 12.0, 0.5)):
    """Orbita de periodo 1: OFF durante d T, ON el resto. Devuelve M, sigma, d."""
    bon, boff = np.array([Vin/L, 0.0]), np.zeros(2)

    def res(p):
        xs = flujo(boff, p[:2], p[2]*T); xT = flujo(bon, xs, (1 - p[2])*T)
        return [xT[0] - p[0], xT[1] - p[1], a*(xs[1] - Vref) - VL - (VU - VL)*p[2]]
    p, info, ier, msg = fsolve(res, guess, xtol=1e-13, full_output=True)
    assert ier == 1 and 0 < p[2] < 1, (Vin, msg)
    xs = flujo(boff, p[:2], p[2]*T)
    n = np.array([0.0, a]); fm, fp = A @ xs + boff, A @ xs + bon
    hm = n @ fm - (VU - VL)/T
    sigma = n @ (fp - fm)/hm
    S = np.eye(2) + np.outer(fp - fm, n)/hm
    return expm(A*(1 - p[2])*T) @ S @ expm(A*p[2]*T), sigma, p[2], p


# ---- (1) y (2): sigma, det M y el circulo ----
detM_teo = float(np.exp(-T/(R*C))); radio = float(np.exp(-T/(2*R*C)))
print(f"det M teorico = exp(-T/RC) = {detM_teo:.6f};  radio del circulo = {radio:.4f}")
tabla = []
for Vin in (20.0, 22.0, 24.0, 24.5, 25.0):
    M, sigma, d, _ = orbita(Vin)
    mu = np.linalg.eigvals(M)
    tabla.append(dict(Vin=Vin, sigma=float(sigma), detM=float(np.linalg.det(M)),
                      mu=[[float(z.real), float(z.imag)] for z in mu], mod=[float(abs(z)) for z in mu]))
    print(f"Vin={Vin:5.2f}  sigma={sigma:+.1e}  det M={np.linalg.det(M):.6f}  mu={np.round(mu, 4)}  |mu|={np.round(np.abs(mu), 4)}")
    assert abs(sigma) < 1e-12 and abs(np.linalg.det(M) - detM_teo) < 1e-10
    if abs(mu[0].imag) > 1e-9:
        assert np.allclose(np.abs(mu), radio, atol=1e-9)

# ---- (2 bis): Tabla I de Giaouris et al., TCAS-I 55(4) 2008: multiplicadores publicados ----
publicado = {14.0: (0.1559, [-0.6265 + 0.5354j, -0.6265 - 0.5354j]), 20.0: (0.4024, [-0.6919 + 0.4477j, -0.6919 - 0.4477j]),
             24.0: (0.4993, [-0.8211 + 0.0708j, -0.8211 - 0.0708j]), 25.0: (0.5187, [-1.0929, -0.6214])}
peor_pub = 0.0
for Vin, (d_pub, mu_pub) in publicado.items():
    M, _, d, _ = orbita(Vin, guess=(0.5, 12.0, d_pub))
    mu = np.sort_complex(np.linalg.eigvals(M)); mu_pub = np.sort_complex(np.array(mu_pub, dtype=complex))
    err = float(np.max(np.abs(mu - mu_pub))); peor_pub = max(peor_pub, err, abs(d - d_pub))
    prod_pub = float(np.real(np.prod(mu_pub)))
    print(f"publicado Vin={Vin:4.0f}: d'={d_pub}  aqui {d:.4f};  max|mu - mu_pub| = {err:.1e};  producto de los publicados = {prod_pub:.4f}")
    assert err < 6e-5 and abs(d - d_pub) < 6e-5 and abs(prod_pub - detM_teo) < 2e-4
print(f"la Tabla I publicada se reproduce a los cuatro digitos impresos; sus productos valen exp(-T/RC) = {detM_teo:.4f}")

# ---- (3): flip como raiz del escalar ----
f = lambda V: np.linalg.det(np.eye(2) + orbita(V)[0])
V_flip = brentq(f, 24.0, 25.0, xtol=1e-11)
M, _, d_flip, _ = orbita(V_flip); mu_flip = np.sort(np.linalg.eigvals(M).real)
print(f"flip: det(I+M) = 0 en Vin* = {V_flip:.4f} V,  d_off = {d_flip:.4f},  multiplicadores {np.round(mu_flip, 5)}")
assert abs(mu_flip[0] + 1) < 1e-8 and abs(mu_flip[1] + detM_teo) < 1e-8
# choque del par complejo sobre el eje real: discriminante tr^2 - 4 det = 0
disc = lambda V: np.trace(orbita(V)[0])**2 - 4*detM_teo
V_choque = brentq(disc, 22.0, V_flip, xtol=1e-9)
print(f"el par complejo llega al eje real en Vin = {V_choque:.4f} V (en mu = -{radio:.4f})")

# ---- (4): simulacion independiente en el dominio del tiempo ----
lam, Vec = np.linalg.eig(A); Vinv = np.linalg.inv(Vec)


def paso(x, b, t):
    """Flujo exacto x(t) = xeq + e^{At}(x - xeq) por autovalores (independiente de expm aumentada)."""
    xeq = -np.linalg.solve(A, b)
    return (xeq + (Vec @ (np.exp(lam*t)*(Vinv @ (x - xeq))))).real


def ciclo(x, Vin):
    """Un periodo con la logica del modulador: OFF hasta que a (v - Vref) cae bajo la rampa, luego ON."""
    bon, boff = np.array([Vin/L, 0.0]), np.zeros(2)
    h = lambda t: a*(paso(x, boff, t)[1] - Vref) - VL - (VU - VL)*t/T
    assert h(0.0) > 0 and h(T) < 0, "el ciclo no tiene un unico cruce: fuera del regimen supuesto"
    lo, hi = 0.0, T
    for _ in range(60):
        mid = 0.5*(lo + hi)
        if h(mid) > 0: lo = mid
        else: hi = mid
    ts = 0.5*(lo + hi)
    return paso(paso(x, boff, ts), bon, T - ts)


def amplitud2(Vin, x0, nmax=40000):
    x = x0.copy(); hist = []
    for k in range(nmax):
        x = ciclo(x, Vin); hist.append(x[1])
        if k > 400 and k % 200 == 0:
            ult = np.array(hist[-8:])
            if abs(ult[-1] - ult[-3]) < 1e-12 and abs(ult[-2] - ult[-4]) < 1e-12: break
    ult = np.array(hist[-64:]); pares, impares = ult[0::2], ult[1::2]
    assert np.ptp(pares) < 1e-8 and np.ptp(impares) < 1e-8, "no es un ciclo de periodo 1 o 2"
    return float((pares.mean() - impares.mean())**2), x


_, _, _, p0 = orbita(V_flip)
x = np.array([p0[0], p0[1] + 1e-3])
Vs = np.linspace(V_flip + 0.02, V_flip + 0.12, 9); amp = []
for Vin in Vs:
    a2, x = amplitud2(Vin, x); amp.append(a2)
coef = np.polyfit(Vs, amp, 1); V_medido = -coef[1]/coef[0]
r2 = 1 - np.sum((np.polyval(coef, Vs) - amp)**2)/np.sum((np.array(amp) - np.mean(amp))**2)
a_antes, _ = amplitud2(V_flip - 0.05, np.array([p0[0], p0[1] + 1e-3]))
print(f"dominio del tiempo: periodo 1 en Vin* - 0.05 V (amplitud^2 = {a_antes:.1e}); ciclo 2 por encima;")
print(f"  ajuste lineal de la amplitud^2 (R^2 = {r2:.5f}) -> frontera medida {V_medido:.4f} V frente a {V_flip:.4f} V del escalar "
      f"({abs(V_medido - V_flip)/V_flip*100:.3f} %)")
assert a_antes < 1e-12 and r2 > 0.999 and abs(V_medido - V_flip) < 0.01

json.dump(dict(peor_diferencia_con_tabla_publicada=peor_pub, detM_teorico=detM_teo, radio=radio, tabla=tabla, V_flip=float(V_flip), d_off_flip=float(d_flip),
               mu_flip=list(map(float, mu_flip)), V_choque=float(V_choque), V_medido=float(V_medido), R2=float(r2),
               Vs=list(map(float, Vs)), amplitud2=amp), open(RES / 'exp7.json', 'w'), indent=1)
print("\nresultados/exp7.json escrito"); sys.exit(0)
