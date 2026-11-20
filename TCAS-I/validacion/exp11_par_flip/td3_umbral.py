"""Paso 3: barrido fino del desajuste de L a 1 mA por encima de la frontera
(ma = 10 kA/s). La frecuencia del batido simulada frente a la ley
    theta = sqrt(kappa^2 - Delta^2/4) / |mu|        (rad por ciclo)
y el punto en que el batido desaparece (|Delta| = 2 kappa)."""
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import par_flip_interleaved as pf
from par_flip_interleaved import Conv, T, monodromia, analiza
from sim_dosfases import corre
from td1_validacion import frontera, MA
from td2_atractor import clasifica

NCIC, COLA, DI = 3_000_000, 1_000_000, 1e-3
RATIOS = [1.000, 1.002, 1.004, 1.006, 1.008, 1.010, 1.012, 1.013, 1.014, 1.0145,
          1.015, 1.0155, 1.016, 1.017, 1.018, 1.020, 1.025]

def uno(r):
    Ls, RLs = [100e-6, 100e-6*r], [0.060, 0.060]
    Is, g = frontera(Ls, RLs)
    x0, tcs, M, dets = monodromia(Conv(Ls, RLs, Is + DI, MA), *g)
    neg, Delta, k2 = analiza(M)
    k = np.sqrt(max(k2, 0.0))
    # forma cerrada
    ks = pf.al**2*T*T/(4*pf.C*np.sqrt(Ls[0]*Ls[1]))
    D = [tcs[0]/T, (tcs[1]+T/2)/T]
    rho = [((RLs[i]+pf.Ron)*D[i] + (RLs[i]+pf.Rd)*(1-D[i]))*T/Ls[i] + pf.al*pf.ESR*T/Ls[i] for i in range(2)]
    Dc = dets[0]*np.exp(-rho[0]) - dets[1]*np.exp(-rho[1])
    th_lin = (np.pi - abs(np.angle(neg[1]))) if abs(neg[0].imag) > 1e-12 else 0.0
    th_cf = np.sqrt(max(ks**2 - Dc**2/4, 0.0))
    out = corre(Conv(Ls, RLs, Is + DI, MA), x0 + 1e-6*np.array([1.0, -1.0, 0.0]), tcs, NCIC)[0]
    tipo, d = clasifica(out[-COLA:])
    th_sim = 2*np.pi/d['batido'] if tipo == "batido" else 0.0
    serie = out[-8000:].copy() if r in (1.000, 1.010, 1.020) else None
    return dict(r=r, q=abs(Delta)/(2*k), q_cf=abs(Dc)/(2*ks), tipo=tipo, th_lin=th_lin, th_cf=th_cf,
                th_sim=th_sim, a1=d.get('a1', 0.0), a2=d.get('a2', 0.0),
                neg=[complex(n) for n in neg]), serie

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(uno, RATIOS))
    print("  L2/L1   |D|/2k  (cerrada)   lineal            theta_lin   theta_cerrada  theta_sim    atractor")
    series = {}
    for d, s in res:
        lin = "complejo" if d['th_lin'] > 0 else f"reales {d['neg'][0].real:.4f},{d['neg'][1].real:.4f}"
        print(f"  {d['r']:.4f}  {d['q']:6.3f}  {d['q_cf']:6.3f}     {lin:26s} {d['th_lin']:.4e}  {d['th_cf']:.4e}   {d['th_sim']:.4e}   "
              f"{d['tipo']}  (a1 {d['a1']*1e3:.2f}, a2 {d['a2']*1e3:.2f} mA)")
        if s is not None:
            series[f"r{d['r']:.3f}"] = s
    json.dump([{k: (v if k != 'neg' else [[z.real, z.imag] for z in v]) for k, v in d.items()} for d, _ in res],
              open("resultados/td3_umbral.json", "w"), indent=1)
    np.savez("resultados/td3_series.npz", **series)
    kk = pf.al**2*T*T/(4*pf.C*100e-6)
    print(f"\n  kappa* = {kk:.4e} rad/ciclo -> batido maximo {1e5*kk/(2*np.pi):.1f} Hz a 100 kHz")
