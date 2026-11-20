"""Forma cerrada de kappa y de Delta frente a la monodromia.

Reduccion: en el bloque de corrientes, con d = det S = -alpha (reflexion),
    Me ~ [[mu1, X1], [d X2, mu2]],   autovalores  mu_m +/- sqrt(Delta^2/4 + d X1 X2)
    X_k = -(1-alpha) e_k h + a_k c h^2 (1 - 6 alpha + alpha^2) / (2 (1+alpha)),   h = T/2
    e_k = al ESR / L_k,  a_k c = al^2 / (L_k C)
de modo que  kappa^2 = alpha X1 X2  y, en la frontera (alpha = 1),
    kappa* = T^2 / (4 C sqrt(L1 L2)) = (pi f0 / fs)^2.
Como d < 0 (reflexion), un acoplamiento simetrico separa el par FUERA del eje
real; con d > 0 (paso) lo separaria a lo largo del eje.
"""
import numpy as np
import par_flip_interleaved as pf
from par_flip_interleaved import Conv, T, monodromia, analiza

def kappa_formula(alpha, Ls, C, ESR):
    al = pf.R/(pf.R+ESR); h = T/2
    X = []
    for L in Ls:
        e, ac = al*ESR/L, al*al/(L*C)
        X.append(-(1-alpha)*e*h + ac*h*h*(1 - 6*alpha + alpha**2)/(2*(1+alpha)))
    return np.sqrt(alpha*X[0]*X[1]) if X[0]*X[1] > 0 else float('nan')

def pon(C, ESR):
    pf.C, pf.ESR, pf.al = C, ESR, pf.R/(pf.R+ESR)

def punto(Ls, RLs, Iref, ma, g):
    x0, tcs, M, dets = monodromia(Conv(Ls, RLs, Iref, ma), *g)
    neg, Delta, k2 = analiza(M)
    return x0, tcs, M, dets, Delta, np.sqrt(max(k2, 0.0)), neg

print("1. kappa: monodromia frente a forma cerrada (fases iguales, ma = 0, D < 0.5)")
print("   C[uF] ESR[mOhm]  Iref    D     alpha    kappa(num)  kappa(formula)  error")
for C in (47e-6, 100e-6, 220e-6):
    for ESR in (0.0, 0.025, 0.050):
        pon(100e-6, 0.025)
        g = ([4.0e-6, 9.0e-6], [1.30, 1.45, 4.2])
        # continuacion suave hasta (C, ESR)
        for s in np.linspace(0, 1, 6):
            pon(100e-6 + s*(C-100e-6), 0.025 + s*(ESR-0.025))
            x0, tcs, M, dets, Delta, k, neg = punto([100e-6]*2, [0.060]*2, 1.60, 0.0, g)
            g = (list(tcs), list(x0))
        for Iref in (1.60, 1.30, 1.00, 1.80, 1.95):
            gi = g
            for I in np.linspace(1.60, Iref, 5):
                x0, tcs, M, dets, Delta, k, neg = punto([100e-6]*2, [0.060]*2, I, 0.0, gi)
                gi = (list(tcs), list(x0))
            a = abs(dets[0])
            kf = kappa_formula(a, [100e-6]*2, C, ESR)
            print(f"   {C*1e6:5.0f}  {ESR*1e3:5.0f}     {Iref:.2f}  {tcs[0]/T:.3f}  {a:.4f}   {k:.4e}  {kf:.4e}   {100*(kf/k-1):+5.1f}%")
pon(100e-6, 0.025)

print("\n2. En la frontera de estabilidad con rampa (alpha = 1): kappa* = T^2/(4 C sqrt(L1 L2))")
from td1_validacion import frontera, MA
for nombre, Ls, RLs in [("emparejado", [100e-6]*2, [0.060]*2), ("L2/L1 = 1.02", [100e-6, 102e-6], [0.060]*2),
                        ("RL 60/52", [100e-6]*2, [0.060, 0.052])]:
    Is, g = frontera(Ls, RLs)
    x0, tcs, M, dets, Delta, k, neg = punto(Ls, RLs, Is, MA, g)
    ks = pf.al**2*T*T/(4*pf.C*np.sqrt(Ls[0]*Ls[1]))
    # Delta: diferencia de los multiplicadores flip que tendria cada fase sola
    rho = [((RLs[i]+pf.Ron)*d + (RLs[i]+pf.Rd)*(1-d))*T/Ls[i] + pf.al*pf.ESR*T/Ls[i]
           for i, d in enumerate([tcs[0]/T, (tcs[1]+T/2)/T])]
    Df = dets[0]*np.exp(-rho[0]) - dets[1]*np.exp(-rho[1])
    print(f"   {nombre:14s} kappa(num) = {k:.4e}  T^2/(4LC) = {ks:.4e} ({100*(ks/k-1):+.1f}%)"
          f"   Delta(num) = {Delta:+.3e}  det S1 e^-rho1 - det S2 e^-rho2 = {Df:+.3e}")
print(f"\n   (pi f0/fs)^2 = {(np.pi*(1/(2*np.pi*np.sqrt(100e-6*100e-6)))/1e5)**2:.4e};"
      f"  batido emparejado = 2 pi/kappa* = {2*np.pi/(pf.al**2*T*T/(4*pf.C*100e-6)):.0f} ciclos")
