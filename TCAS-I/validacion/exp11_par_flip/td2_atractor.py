"""Paso 2: el atractor mas alla de la frontera de estabilidad.

Para cada desajuste se arranca en la orbita de periodo 1 (inestable) con una
perturbacion antisimetrica de 1e-6 A y se simula hasta el regimen
permanente. Se clasifica la cola: periodo 1, periodo 2 (doblamiento limpio),
periodo p, o cuasiperiodico con batido lento. Se compara con lo que predice
el par flip de la monodromia en ese mismo punto.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from par_flip_interleaved import Conv, T, monodromia, analiza
from sim_dosfases import prep, corre
from td1_validacion import frontera, CASOS, MA

NCIC, COLA = 600_000, 120_000
DELTAS_mA = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def clasifica(out):
    """out: estados estroboscopicos de la cola. Devuelve (tipo, datos)."""
    y = out[:, 0]
    tol = 1e-9
    dif = lambda p: np.max(np.abs(y[p:] - y[:-p]))
    if dif(1) < tol:
        return "periodo 1", {}
    a1 = 0.5*np.mean(np.abs(out[1:, 0] - out[:-1, 0]))
    a2 = 0.5*np.mean(np.abs(out[1:, 1] - out[:-1, 1]))
    if dif(2) < tol:
        return "periodo 2", dict(a1=a1, a2=a2)
    for p in (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        if dif(p) < tol:
            return f"periodo {p}", dict(a1=a1, a2=a2)
    # envolvente lenta: muestras pares
    e = y[::2] - np.mean(y[::2])
    n = len(e)
    F = np.abs(np.fft.rfft(e*np.hanning(n)))
    F[0] = 0.0
    kp = int(np.argmax(F))
    # interpolacion parabolica del pico
    if 0 < kp < len(F)-1:
        a, b, c = F[kp-1], F[kp], F[kp+1]
        kp_f = kp + 0.5*(a-c)/(a-2*b+c)
    else:
        kp_f = float(kp)
    batido = 2.0*n/kp_f if kp_f > 0 else np.inf        # en ciclos de conmutacion
    return "batido", dict(a1=a1, a2=a2, batido=batido, rango=np.ptp(e),
                          pico=F[kp]/np.sum(F))


def un_punto(arg):
    nombre, Ls, RLs, Is, g, dmA = arg
    Iref = Is + dmA*1e-3
    cv = Conv(Ls, RLs, Iref, MA)
    # continuacion de la orbita inestable desde la frontera
    gg = g
    for I in np.linspace(Is, Iref, max(2, int(abs(dmA)/2)+2)):
        x0, tcs, M, dets = monodromia(Conv(Ls, RLs, I, MA), *gg)
        gg = (list(tcs), list(x0))
    neg, Delta, k2 = analiza(M)
    if abs(neg[0].imag) > 1e-9:
        th = np.pi - abs(np.angle(neg[1]))
        lin = f"complejo |mu|={abs(neg[0]):.5f} batido {2*np.pi/th:6.0f}"
    else:
        lin = f"reales {neg[0].real:.5f}, {neg[1].real:.5f}"
    out = corre(cv, x0 + 1e-6*np.array([1.0, -1.0, 0.0]), tcs, NCIC)[0]
    tipo, d = clasifica(out[-COLA:])
    return nombre, dmA, lin, tipo, d, abs(Delta)/(2*np.sqrt(max(k2, 1e-300)))


if __name__ == "__main__":
    sel = sys.argv[1:]
    tareas = []
    for nombre, Ls, RLs in CASOS:
        if sel and not any(s in nombre for s in sel):
            continue
        Is, g = frontera(Ls, RLs)
        for dmA in DELTAS_mA:
            tareas.append((nombre, Ls, RLs, Is, g, dmA))
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(un_punto, tareas))
    actual = None
    for nombre, dmA, lin, tipo, d, q in res:
        if nombre != actual:
            print(f"\n{nombre}")
            print("  dI[mA]  |D|/2k   lineal (orbita de periodo 1)              atractor simulado")
            actual = nombre
        extra = ""
        if tipo == "batido":
            extra = (f"cada {d['batido']:7.0f} ciclos, rango {d['rango']*1e3:7.3f} mA, "
                     f"a1 {d['a1']*1e3:7.3f} a2 {d['a2']*1e3:7.3f} mA")
        elif tipo.startswith("periodo") and d:
            extra = f"a1 {d['a1']*1e3:7.3f} mA, a2 {d['a2']*1e3:7.3f} mA"
        print(f"  {dmA:6.1f}  {q:6.3f}   {lin:42s} {tipo:10s} {extra}")
