"""El signo bajo la raiz es el del tipo de evento.

Mismo convertidor de dos fases iguales; se sube la rampa hasta pasar la
pendiente deadbeat, de modo que los dos eventos pasan de reflexion
(det S < 0) a paso (det S > 0). Con reflexiones el acoplamiento separa el par
fuera del eje real (complejo); con eventos de paso lo separa a lo largo del
eje (dos reales), con la MISMA interferencia X:  mu = d +/- X sqrt(d).
"""
import numpy as np
import par_flip_interleaved as pf
from par_flip_interleaved import Conv, T, monodromia, analiza

def X_formula(alpha_con_signo, L):
    """X con d = det S = alpha_con_signo (negativo en reflexion)."""
    d = alpha_con_signo
    h = T/2
    e, ac = pf.al*pf.ESR/L, pf.al**2/(L*pf.C)
    eta, bg = e*h + 0.5*ac*h*h, ac*h*h
    return -eta*(1+d) - bg*(3*d+1)/(d-1)

g = ([4.0e-6, 9.0e-6], [1.30, 1.45, 4.2])
print("   ma[kA/s]   det S      tipo        par (monodromia)                      d +/- X sqrt(d)  (forma cerrada)")
for ma in (0, 20e3, 40e3, 60e3, 75e3, 85e3, 100e3, 120e3, 150e3):
    Iref = 1.60 + ma*0.396*T
    x0, tcs, M, dets = monodromia(Conv([100e-6]*2, [0.060]*2, Iref, ma), *g)
    g = (list(tcs), list(x0))
    mu = np.linalg.eigvals(M)
    par = sorted(mu, key=lambda z: abs(z - np.mean(dets)))[:2]
    d = dets[0]
    X = X_formula(d, 100e-6)
    s = X*np.sqrt(complex(d))
    tipo = "reflexion" if d < 0 else "paso"
    fc = (f"{d:+.5f} +/- {abs(s.imag):.5f}j" if d < 0 else f"{d+abs(s.real):+.5f}, {d-abs(s.real):+.5f}")
    pm = (f"{par[0].real:+.5f} +/- {abs(par[0].imag):.5f}j" if abs(par[0].imag) > 1e-9
          else f"{max(p.real for p in par):+.5f}, {min(p.real for p in par):+.5f}")
    print(f"   {ma/1e3:6.0f}    {d:+.5f}   {tipo:10s}  {pm:36s}  {fc}")
