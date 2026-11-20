"""Los numeros que cita la Seccion V-D del manuscrito, con la forma cerrada.

    X_k   = -(a T e^{lam/2} / (2 L_k)) [ (a T / 4C)(3 - cosh lam)/cosh(lam/2) - 2 r_C sinh(lam/2) ]
    kappa = sqrt(-(1+sigma) X_1 X_2),   lam = ln|1+sigma|,   a = R/(R + r_C)
    mu_k  = (1+sigma_k) exp(-rho_k),    Delta = mu_1 - mu_2
"""
import numpy as np
import par_flip_interleaved as pf
from par_flip_interleaved import Conv, T, monodromia, analiza

def cerrada(Ls, RLs, tcs, dets):
    a = pf.al
    lam = np.log(np.sqrt(abs(dets[0]*dets[1])))
    br = (a*T/(4*pf.C))*(3-np.cosh(lam))/np.cosh(lam/2) - 2*pf.ESR*np.sinh(lam/2)
    X = [-(a*T*np.exp(lam/2)/(2*L))*br for L in Ls]
    kap = np.sqrt(np.exp(lam)*X[0]*X[1])
    D = [tcs[0]/T, (tcs[1] - T/2)/T if tcs[1] > T/2 else (tcs[1] + T/2)/T]
    rho = [((RLs[i]+pf.Ron)*D[i] + (RLs[i]+pf.Rd)*(1-D[i]))*T/Ls[i] + a*pf.ESR*T/Ls[i] for i in range(2)]
    mu = [dets[i]*np.exp(-rho[i]) for i in range(2)]
    return kap, mu[0]-mu[1], lam, mu

print("Tabla III (L 100/106 uH, RL 60/52 mOhm, Iref = 1.60 A, ma = 0)")
Ls, RLs = [100e-6, 106e-6], [0.060, 0.052]
x0, tcs, M, dets = monodromia(Conv(Ls, RLs, 1.60), [4.0e-6, 9.0e-6], [1.30, 1.45, 4.2])
neg, Dn, k2n = analiza(M)
kap, Dc, lam, mu = cerrada(Ls, RLs, tcs, dets)
print(f"  monodromia : par {neg[1].real:.6f} +/- {abs(neg[1].imag):.4e} j")
print(f"  reduccion  : Delta = {Dn:+.4e}, kappa = {np.sqrt(k2n):.4e}, |Delta|/2k = {abs(Dn)/(2*np.sqrt(k2n)):.3f}, "
      f"Im = {np.sqrt(k2n - Dn**2/4):.4e}")
print(f"  cerrada    : Delta = {Dc:+.4e}, kappa = {kap:.4e}, |Delta|/2k = {abs(Dc)/(2*kap):.3f}, "
      f"Im = {np.sqrt(kap**2 - Dc**2/4):.4e},  media = {0.5*(mu[0]+mu[1]):.5f}, lambda = {lam:.4f}")

print("\nFrontera con rampa de 10 kA/s, fases iguales")
from td1_validacion import frontera, MA
Ls, RLs = [100e-6, 100e-6], [0.060, 0.060]
Is, g = frontera(Ls, RLs)
x0, tcs, M, dets = monodromia(Conv(Ls, RLs, Is, MA), *g)
neg, Dn, k2n = analiza(M)
kap, Dc, lam, mu = cerrada(Ls, RLs, tcs, dets)
ks = pf.al**2*T*T/(4*pf.C*100e-6)
print(f"  Iref* = {Is:.4f} A, D = {tcs[0]/T:.4f}, det S = {dets[0]:.5f}, lambda = {lam:+.4f}")
print(f"  kappa: monodromia {np.sqrt(k2n):.4e}; cerrada(lambda) {kap:.4e}; a^2 T^2/(4LC) = {ks:.4e}; (pi f0/fs)^2 = {ks/pf.al**2:.4e}; a = {pf.al:.4f}")
print(f"  batido: 2 pi/kappa = {2*np.pi/ks:.0f} ciclos = {1e5*ks/(2*np.pi):.1f} Hz;  monodromia en Iref*: {2*np.pi/(np.pi-abs(np.angle(neg[1]))):.0f} ciclos")
print(f"  ventana: 2 kappa = {2*ks:.4e} = {200*ks/abs(dets[0]):.2f} % de |1+sigma|")
