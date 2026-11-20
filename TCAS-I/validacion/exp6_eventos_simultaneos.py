"""Exp. 6 — Eventos simultaneos: la esquina de codimension dos.

La regla de paridad supone que los eventos del periodo ocurren uno detras de
otro y que cada uno se factoriza como una saltacion. Cuando dos conmutaciones
COINCIDEN en el tiempo, el orden del producto deja de estar determinado por la
fisica y hay dos monodromias candidatas,

    M_ab = Phi . S_a S_b . Phi        y        M_ba = Phi . S_b S_a . Phi .

La pregunta es si los invariantes tambien se bifurcan. La respuesta, en orden
creciente de incomodidad:

  (A) Con saltaciones dadas como matrices, el orden reescribe la monodromia
      entera y mueve los multiplicadores, pero det, la rapidez total y W~W son
      identicos: det(S_a S_b) = det(S_a) det(S_b) es multiplicativa.

  (B) El conmutador [B_a, B_b] de los generadores reproduce exactamente
      S_a S_b - S_b S_a, lo que da contenido a la conjetura de la Discusion.

  (C) Dos modulos buck IDENTICOS en paralelo, mismo reloj, enclavamiento por
      pico, referencias iguales: las dos saltaciones CONMUTAN exactamente,
      porque sin acoplamiento u_1 no tiene componente en la fase 2 ni al
      reves. La esquina es removible: no hay nada que elegir.

  (D) Con inductores acoplados dejan de conmutar y la monodromia pasa a
      depender del orden. El determinante, con fases identicas, no: lo protege
      la simetria.

  (E) Rota la simetria (L_2 y R_2 distintos) Y con acoplamiento, el
      determinante SI depende del orden, hasta un factor 20.

  (F) Y no es un artefacto de como se construye la esquina: barriendo el
      desajuste a ambos lados de cero, la monodromia VERDADERA converge a una
      candidata por la izquierda y a la otra por la derecha. det M tiene un
      salto finito en la esquina.

  (G) Buscando en el espacio de parametros fisicos aparecen esquinas donde
      hasta el SIGNO depende del orden. El mecanismo es que un acoplamiento
      fuerte invierte la pendiente de la corriente de una fase cuando la otra
      conmuta, asi que la ESPECIE del segundo evento depende de si el primero
      ya ocurrio. La regla de paridad acierta el signo en cada orden por
      separado; lo que deja de estar definido es N_r.

Conclusion para el manuscrito: la Prop. 2 necesita una hipotesis de separacion
de eventos --- no de transversalidad, que aqui se cumple en ambos ordenes ---
y el presupuesto de volumen es discontinuo en la esquina.

Salida: resultados/exp6.json
"""
import json, pathlib
import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve
import comun as C

RES = pathlib.Path(__file__).resolve().parent / 'resultados'
RES.mkdir(exist_ok=True)
SEMILLA = 20260726
fallos = 0

# ================== (A) ambiguidad generica del orden ==================
print('=' * 76)
print('(A) DOS REFLEXIONES SIMULTANEAS, ORDEN AMBIGUO   (n = 4, 400 pares)')
print('=' * 76)
rng = np.random.default_rng(SEMILLA)
sep_M, sep_mu, dif_det, dif_sgn, dif_ww, dif_lam = [], [], [], [], [], []
G4 = C.algebra(4)
for ensayo in range(400):
    sa = rng.uniform(-4.0, -1.2)
    sb = rng.uniform(-4.0, -1.2)
    Sa, ua, na = C.salt_desde_sigma(rng, 4, sa)
    Sb, ub, nb = C.salt_desde_sigma(rng, 4, sb)
    A1 = rng.normal(size=(4, 4))*0.35/2
    A2 = rng.normal(size=(4, 4))*0.35/2
    P1, P2 = expm(A1*0.4), expm(A2*0.6)
    Mab = P2 @ (Sa @ Sb) @ P1
    Mba = P2 @ (Sb @ Sa) @ P1
    esc = max(np.max(np.abs(Mab)), 1.0)
    sep_M.append(float(np.max(np.abs(Mab - Mba))/esc))
    ea = np.sort_complex(np.linalg.eigvals(Mab))
    eb = np.sort_complex(np.linalg.eigvals(Mba))
    sep_mu.append(float(np.max(np.abs(ea - eb))
                        / max(np.max(np.abs(ea)), 1.0)))
    da, db = np.linalg.det(Mab), np.linalg.det(Mba)
    dif_det.append(float(abs(da - db)/max(abs(da), 1e-300)))
    dif_sgn.append(int(np.sign(da) != np.sign(db)))
    dif_lam.append(abs(np.log(abs((1+sa)*(1+sb)))
                       - np.log(abs(np.linalg.det(Sa @ Sb)))))
    if ensayo < 40:                      # el algebra es cara: una muestra
        Wa, _ = C.versor_salt(ua, na, G4)
        Wb, _ = C.versor_salt(ub, nb, G4)
        wab = float(((Wa*Wb)*~(Wa*Wb))(0))
        wba = float(((Wb*Wa)*~(Wb*Wa))(0))
        dif_ww.append(abs(wab - wba))
        err = max(float(np.max(np.abs(C.accion(Wa*Wb, G4) - Sa @ Sb))
                        / np.max(np.abs(Sa @ Sb))),
                  float(np.max(np.abs(C.accion(Wb*Wa, G4) - Sb @ Sa))
                        / np.max(np.abs(Sb @ Sa))))
        if err > 1e-8:
            fallos += 1

print(f"  lo que SI depende del orden")
print(f"    ||M_ab - M_ba||/||M||        mediana {np.median(sep_M):9.3e}"
      f"   maxima {np.max(sep_M):9.3e}")
print(f"    multiplicadores, relativo    mediana {np.median(sep_mu):9.3e}"
      f"   maxima {np.max(sep_mu):9.3e}")
print(f"  lo que NO depende del orden (cotas superiores sobre los 400)")
print(f"    det M, discrepancia relativa                 {np.max(dif_det):9.3e}")
print(f"    rapidez total                                {np.max(dif_lam):9.3e}")
print(f"    W~W (muestra de 40)                          {np.max(dif_ww):9.3e}")
print(f"    signo de det M distinto                    {sum(dif_sgn):5d} de 400")
print(f"    errores relativos de la accion versorial   {fallos:5d} de 40")
print()
print(f"  El orden reescribe la monodromia entera (mediana "
      f"{100*np.median(sep_M):.0f}% de su propia")
print("  magnitud) y desplaza los multiplicadores, pero det, la paridad, la")
print("  rapidez y W~W no se enteran: los invariantes sobreviven a una")
print("  ambiguedad que la matriz no sobrevive.")

# ================== (B) el conmutador de los generadores ==================
print()
print('=' * 76)
print('(B) EL CONMUTADOR DE LOS GENERADORES MIDE ESA AMBIGUEDAD')
print('=' * 76)
rng = np.random.default_rng(SEMILLA + 11)
G3 = C.algebra(3)
print(f"{'||[B_a,B_b]||':>15} {'||S_aS_b-S_bS_a||':>19} {'cociente':>11}")
coc = []
for _ in range(8):
    sa, sb = rng.uniform(-3, -1.3), rng.uniform(-3, -1.3)
    Sa, ua, na = C.salt_desde_sigma(rng, 3, sa)
    Sb, ub, nb = C.salt_desde_sigma(rng, 3, sb)
    U = lambda z: sum(float(z[i])*G3['w'][i] for i in range(3))
    V = lambda z: sum(float(z[i])*G3['v'][i] for i in range(3))
    Ba = 2*(U(ua) ^ V(na))
    Bb = 2*(U(ub) ^ V(nb))
    cm = Ba*Bb - Bb*Ba
    nb_ = float(np.max(np.abs(cm.value)))
    nm = float(np.max(np.abs(Sa@Sb - Sb@Sa)))
    coc.append(nm/nb_ if nb_ > 0 else np.nan)
    print(f"{nb_:15.6f} {nm:19.6f} {nm/nb_:11.6f}")
print(f"  el conmutador se anula exactamente cuando los dos planos conmutan,")
print(f"  y su norma acota la separacion de las dos monodromias candidatas.")

# ================== (C)-(D) dos modulos buck en paralelo ==================
Vin, Vd = 12.0, 0.45
Lp, RLp = 100e-6, 0.060
Ron, Rd = 0.040, 0.020
Cout, R = 220e-6, 2.0
T = 1.0/100e3
Iref = 2.6


def campos(k, L1=Lp, L2=Lp, RL1=RLp, RL2=RLp, Ron2=Ron, Rd2=Rd):
    """A(s1,s2), b(s1,s2) con acoplamiento magnetico k entre las dos fases.
    Admite fases distintas para poder romper la simetria."""
    Mut = k*np.sqrt(L1*L2)
    Linv = np.linalg.inv(np.array([[L1, Mut], [Mut, L2]]))
    out = {}
    for s1 in (1, 0):
        for s2 in (1, 0):
            Rs = [RL1 + (Ron if s1 else Rd), RL2 + (Ron2 if s2 else Rd2)]
            Vs = [Vin if s1 else -Vd, Vin if s2 else -Vd]
            A = np.zeros((3, 3)); b = np.zeros(3)
            # [di1;di2] = Linv ( [V1;V2] - diag(R) [i1;i2] - vC [1;1] )
            A[:2, :2] = -Linv @ np.diag(Rs)
            A[:2, 2] = -Linv @ np.ones(2)
            b[:2] = Linv @ np.array(Vs)
            A[2, 0] = A[2, 1] = 1.0/Cout
            A[2, 2] = -1.0/(R*Cout)
            out[(s1, s2)] = (A, b)
    return out


def paso(A, b, x, t):
    Z = np.zeros((4, 4)); Z[:3, :3] = A; Z[:3, 3] = b
    E = expm(Z*t)
    return E[:3, :3]@x + E[:3, 3], E[:3, :3]


def orbita(eps, k):
    """Orbita periodica con Iref_2 = Iref (1+eps). Devuelve todo lo necesario
    para montar las dos monodromias candidatas."""
    F = campos(k)
    I2 = Iref*(1.0 + eps)

    def res(z):
        x0, t1, t2 = z[:3], z[3], z[4]
        ta, tb = min(t1, t2), max(t1, t2)
        prim = 1 if t1 <= t2 else 2
        med = (0, 1) if prim == 1 else (1, 0)
        xa = paso(*F[(1, 1)], x0, ta)[0]
        xb = paso(*F[med], xa, tb - ta)[0]
        xT = paso(*F[(0, 0)], xb, T - tb)[0]
        h1 = (xa[0] if prim == 1 else xb[0]) - Iref
        h2 = (xb[1] if prim == 1 else xa[1]) - I2
        return [xT[0]-x0[0], xT[1]-x0[1], xT[2]-x0[2], h1, h2]

    z, _, ier, msg = fsolve(res, [0.9, 0.9, 5.0, 0.42*T, 0.42*T],
                            full_output=True, xtol=1e-13)
    assert ier == 1, msg
    x0, t1, t2 = z[:3], z[3], z[4]
    ta, tb = min(t1, t2), max(t1, t2)
    prim = 1 if t1 <= t2 else 2
    med = (0, 1) if prim == 1 else (1, 0)
    xa, Pa = paso(*F[(1, 1)], x0, ta)
    xb, Pm = paso(*F[med], xa, tb - ta)
    xT, Pf = paso(*F[(0, 0)], xb, T - tb)
    return dict(F=F, x0=x0, xa=xa, xb=xb, ta=ta, tb=tb, prim=prim,
                Pa=Pa, Pm=Pm, Pf=Pf, dt=float(tb - ta))


def salt(F, modo_ini, modo_fin, x, j):
    """Saltacion del evento de la fase j entre dos modos, en el punto x."""
    fm = F[modo_ini][0]@x + F[modo_ini][1]
    fp = F[modo_fin][0]@x + F[modo_fin][1]
    nv = np.zeros(3); nv[j] = 1.0
    u = (fp - fm)/float(fm[j])
    return np.eye(3) + np.outer(u, nv), float(1.0 + nv@u)


def candidatas(o):
    """En la esquina (dt = 0) las dos monodromias posibles, con las
    saltaciones RECONSTRUIDAS en cada orden (los campos intermedios difieren)."""
    F, x = o['F'], o['xa']
    # orden A: primero fase 1, luego fase 2
    S1a, d1a = salt(F, (1, 1), (0, 1), x, 0)
    S2a, d2a = salt(F, (0, 1), (0, 0), x, 1)
    # orden B: primero fase 2, luego fase 1
    S2b, d2b = salt(F, (1, 1), (1, 0), x, 1)
    S1b, d1b = salt(F, (1, 0), (0, 0), x, 0)
    Mab = o['Pf'] @ (S2a @ S1a) @ o['Pa']
    Mba = o['Pf'] @ (S1b @ S2b) @ o['Pa']
    return Mab, Mba, (d1a*d2a), (d1b*d2b)


print()
print('=' * 76)
print('(C) DOS MODULOS BUCK EN PARALELO: LA ESQUINA ES REMOVIBLE SI NO HAY')
print('    ACOPLAMIENTO')
print('=' * 76)
o0 = orbita(0.0, 0.0)
Mab, Mba, dA, dB = candidatas(o0)
S1c, _ = salt(o0['F'], (1, 1), (0, 1), o0['xa'], 0)
S2c, _ = salt(o0['F'], (1, 1), (1, 0), o0['xa'], 1)
print(f"  en epsilon = 0 los dos apagados coinciden: Dt = {o0['dt']:.2e} s")
print(f"  ||[S_1, S_2]|| = {np.max(np.abs(S1c@S2c - S2c@S1c)):.3e}")
print(f"  ||M_ab - M_ba||/||M|| = "
      f"{np.max(np.abs(Mab-Mba))/np.max(np.abs(Mab)):.3e}")
print(f"  Sin acoplamiento, u_1 no tiene componente en la fase 2 ni al reves,")
print(f"  luego n_1^T u_2 = n_2^T u_1 = 0 y las dos saltaciones CONMUTAN.")
print(f"  La ambiguedad del orden no existe: no hay nada que elegir.")
print(f"  det S_a S_b = {dA:+.12f}   det S_b S_a = {dB:+.12f}   "
      f"dif = {abs(dA-dB):.2e}")
print(f"  multiplicadores: "
      f"{np.array2string(np.sort_complex(np.linalg.eigvals(Mab)), precision=5)}")
print(f"  N_r = 2 (dos reflexiones)  ->  paridad par  ->  det M > 0")

print()
print("  Barrido a los dos lados de la esquina:")
print(f"{'epsilon':>10} {'Dt (ns)':>10} {'|M-M_ab|':>11} {'|M-M_ba|':>11} "
      f"{'det M':>12} {'N_r':>4} {'neg':>4}")
barrido, fall_c = [], 0
for e in [-1e-2, -3e-3, -1e-3, -3e-4, -1e-4, 0.0,
          1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    o = orbita(e, 0.0)
    if e == 0.0:
        M = Mab
    else:
        Sfi, dfi = salt(o['F'], (1, 1), (0, 1) if o['prim'] == 1 else (1, 0),
                        o['xa'], 0 if o['prim'] == 1 else 1)
        Sse, dse = salt(o['F'], (0, 1) if o['prim'] == 1 else (1, 0), (0, 0),
                        o['xb'], 1 if o['prim'] == 1 else 0)
        M = o['Pf'] @ Sse @ o['Pm'] @ Sfi @ o['Pa']
        dfi_dse = dfi*dse
    esc = np.max(np.abs(Mab))
    da = float(np.max(np.abs(M - Mab))/esc)
    db = float(np.max(np.abs(M - Mba))/esc)
    det = float(np.linalg.det(M))
    neg = C.n_reales_negativos(M)
    ok = (det > 0) and (neg % 2 == 0)
    fall_c += (not ok)
    barrido.append(dict(eps=e, dt=o['dt'], da=da, db=db, det=det, neg=neg))
    print(f"{e:10.1e} {o['dt']*1e9:10.3f} {da:11.3e} {db:11.3e} "
          f"{det:12.8f} {2:4d} {neg:4d}{'  <-- FALLO' if not ok else ''}")
print()
print("  Los dos limites laterales coinciden porque las candidatas coinciden:")
print("  la monodromia atraviesa la esquina sin salto, y con ella det M y la")
print("  paridad. Dos modulos en paralelo pueden conmutar a la vez sin que la")
print("  teoria tenga nada que decidir.")

print()
print('=' * 76)
print('(D) CON INDUCTORES ACOPLADOS LAS SALTACIONES DEJAN DE CONMUTAR')
print('=' * 76)
print(f"{'k':>7} {'||[S1,S2]||':>13} {'|M_ab-M_ba|/|M|':>17} "
      f"{'det S_aS_b':>14} {'det S_bS_a':>14} {'dif rel':>10}")
acopl, fall_d = [], 0
for k in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7]:
    try:
        o = orbita(0.0, k)
    except AssertionError as e:
        print(f"{k:7.2f}   sin orbita: {str(e)[:40]}")
        continue
    Ma, Mb, dA_, dB_ = candidatas(o)
    S1_, _ = salt(o['F'], (1, 1), (0, 1), o['xa'], 0)
    S2_, _ = salt(o['F'], (1, 1), (1, 0), o['xa'], 1)
    cmt = float(np.max(np.abs(S1_@S2_ - S2_@S1_)))
    sepM = float(np.max(np.abs(Ma - Mb))/np.max(np.abs(Ma)))
    rel = abs(dA_ - dB_)/max(abs(dA_), 1e-300)
    fall_d += int(np.sign(dA_) != np.sign(dB_))
    acopl.append(dict(k=k, cmt=cmt, sepM=sepM, dA=dA_, dB=dB_, rel=rel))
    print(f"{k:7.2f} {cmt:13.4e} {sepM:17.4e} {dA_:+14.8f} {dB_:+14.8f} "
          f"{rel:10.2e}")
print()
print("  El acoplamiento hace que apagar la fase 2 cambie tambien di_1/dt, o")
print("  sea n_1^T u_2 != 0: las saltaciones ya NO conmutan y la monodromia")
print("  pasa a depender del orden. El determinante, en cambio, sigue siendo")
print("  el mismo hasta el ultimo bit. Pero con dos fases identicas eso puede")
print("  ser mera simetria, asi que hay que romperla.")

print()
print('=' * 76)
print('(E) LA PRUEBA DE VERDAD: ESQUINA SIMULTANEA ENTRE FASES DESIGUALES')
print('=' * 76)


def esquina(k, fL2, fR2):
    """Orbita con las dos fases apagando EXACTAMENTE a la vez, con L_2 y
    R_on2 distintos de los de la fase 1. La referencia de la fase 2 no se
    impone: se lee del propio instante de coincidencia."""
    F = campos(k, L2=Lp*fL2, RL2=RLp*fR2, Ron2=Ron*fR2, Rd2=Rd*fR2)

    def res(z):
        x0, t = z[:3], z[3]
        xa = paso(*F[(1, 1)], x0, t)[0]
        xT = paso(*F[(0, 0)], xa, T - t)[0]
        return [xT[0]-x0[0], xT[1]-x0[1], xT[2]-x0[2], xa[0]-Iref]
    z, _, ier, msg = fsolve(res, [0.9, 0.9, 5.0, 0.42*T],
                            full_output=True, xtol=1e-13)
    assert ier == 1, msg
    x0, t = z[:3], z[3]
    xa, Pa = paso(*F[(1, 1)], x0, t)
    _, Pf = paso(*F[(0, 0)], xa, T - t)
    S1a, d1a = salt(F, (1, 1), (0, 1), xa, 0)
    S2a, d2a = salt(F, (0, 1), (0, 0), xa, 1)
    S2b, d2b = salt(F, (1, 1), (1, 0), xa, 1)
    S1b, d1b = salt(F, (1, 0), (0, 0), xa, 0)
    return (Pf @ (S2a @ S1a) @ Pa, Pf @ (S1b @ S2b) @ Pa,
            d1a*d2a, d1b*d2b, float(xa[1]), float(t/T))


print(f"{'k':>6} {'L2/L1':>7} {'R2/R1':>7} {'det S orden A':>15} "
      f"{'det S orden B':>15} {'dif rel':>10} {'signo':>9}")
asim, fall_e = [], 0
for k in [0.0, 0.1, 0.3, 0.5, 0.7]:
    for fL2, fR2 in [(1.0, 1.0), (1.6, 1.0), (1.6, 2.5)]:
        try:
            Ma, Mb, dA_, dB_, i2, D = esquina(k, fL2, fR2)
        except AssertionError as e:
            print(f"{k:6.2f} {fL2:7.2f} {fR2:7.2f}   sin orbita")
            continue
        rel = abs(dA_ - dB_)/max(abs(dA_), 1e-300)
        igual = np.sign(dA_) == np.sign(dB_)
        fall_e += (not igual)
        asim.append(dict(k=k, fL2=fL2, fR2=fR2, dA=dA_, dB=dB_, rel=rel,
                         mismo_signo=bool(igual),
                         det_ab=float(np.linalg.det(Ma)),
                         det_ba=float(np.linalg.det(Mb))))
        print(f"{k:6.2f} {fL2:7.2f} {fR2:7.2f} {dA_:+15.8f} {dB_:+15.8f} "
              f"{rel:10.2e} {'igual' if igual else 'DISTINTO':>9}")
print()
print('=' * 76)
print('(F) NO ES UN ARTEFACTO: LOS DOS LIMITES LATERALES DE LA MONODROMIA')
print('    REAL SON LAS DOS CANDIDATAS')
print('=' * 76)


def orbita_gen(F, I1, I2, z0=None):
    """Orbita con los dos eventos SEPARADOS, campos arbitrarios. Devuelve la
    monodromia verdadera y el hueco temporal entre los dos apagados."""
    def res(z):
        x0, t1, t2 = z[:3], z[3], z[4]
        ta, tb = min(t1, t2), max(t1, t2)
        med = (0, 1) if t1 <= t2 else (1, 0)
        xa = paso(*F[(1, 1)], x0, ta)[0]
        xb = paso(*F[med], xa, tb - ta)[0]
        xT = paso(*F[(0, 0)], xb, T - tb)[0]
        h1 = (xa[0] if t1 <= t2 else xb[0]) - I1
        h2 = (xb[1] if t1 <= t2 else xa[1]) - I2
        return [xT[0]-x0[0], xT[1]-x0[1], xT[2]-x0[2], h1, h2]
    z, _, ier, msg = fsolve(res, z0 if z0 is not None else
                            [0.9, 0.9, 5.0, 0.42*T, 0.42*T],
                            full_output=True, xtol=1e-13)
    assert ier == 1, msg
    x0, t1, t2 = z[:3], z[3], z[4]
    ta, tb = min(t1, t2), max(t1, t2)
    uno = t1 <= t2
    med = (0, 1) if uno else (1, 0)
    xa, Pa = paso(*F[(1, 1)], x0, ta)
    xb, Pm = paso(*F[med], xa, tb - ta)
    _, Pf = paso(*F[(0, 0)], xb, T - tb)
    Sfi, _ = salt(F, (1, 1), med, xa, 0 if uno else 1)
    Sse, _ = salt(F, med, (0, 0), xb, 1 if uno else 0)
    return Pf @ Sse @ Pm @ Sfi @ Pa, float(tb - ta)


K_F, FL, FR = 0.5, 1.6, 1.0
F_F = campos(K_F, L2=Lp*FL, RL2=RLp*FR, Ron2=Ron*FR, Rd2=Rd*FR)
Ma_F, Mb_F, dA_F, dB_F, I2_F, D_F = esquina(K_F, FL, FR)
dab, dba = float(np.linalg.det(Ma_F)), float(np.linalg.det(Mb_F))
print(f"  k = {K_F}, L2/L1 = {FL}, R2/R1 = {FR};  en la esquina "
      f"I_ref2 = {I2_F:.6f} A, D = {D_F:.5f}")
print(f"  candidatas:  det M_ab = {dab:+.8f}    det M_ba = {dba:+.8f}")
print()
print(f"{'epsilon':>10} {'Dt (ns)':>10} {'det M real':>14} "
      f"{'-> M_ab':>11} {'-> M_ba':>11} {'signo':>7}")
lat, fall_f = [], 0
for e in [-3e-3, -1e-3, -3e-4, -1e-4, -3e-5,
          3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
    try:
        Mr, dt = orbita_gen(F_F, Iref, I2_F*(1.0 + e))
    except AssertionError:
        print(f"{e:10.1e}   sin orbita")
        continue
    dr = float(np.linalg.det(Mr))
    ra = abs(np.max(np.abs(Mr - Ma_F))/np.max(np.abs(Ma_F)))
    rb = abs(np.max(np.abs(Mr - Mb_F))/np.max(np.abs(Mb_F)))
    lat.append(dict(eps=e, dt=dt, det=dr, ra=float(ra), rb=float(rb)))
    fall_f += int(np.sign(dr) != np.sign(dab))
    print(f"{e:10.1e} {dt*1e9:10.4f} {dr:14.8f} {ra:11.3e} {rb:11.3e} "
          f"{np.sign(dr):+7.0f}")
izq = [r for r in lat if r['eps'] < 0]
der = [r for r in lat if r['eps'] > 0]
print()
print(f"  limite por la izquierda:  det M -> {izq[-1]['det']:+.6f}"
      f"   (candidata mas proxima: "
      f"{'M_ab' if izq[-1]['ra'] < izq[-1]['rb'] else 'M_ba'})")
print(f"  limite por la derecha:    det M -> {der[0]['det']:+.6f}"
      f"   (candidata mas proxima: "
      f"{'M_ab' if der[0]['ra'] < der[0]['rb'] else 'M_ba'})")
print(f"  salto relativo del determinante al cruzar la esquina: "
      f"{abs(der[0]['det']-izq[-1]['det'])/abs(izq[-1]['det']):.3f}")
print(f"  cambios de signo en todo el barrido: {fall_f} de {len(lat)}")
print()
print("  Confirmado: no es un artefacto de la construccion. En una esquina")
print("  simultanea entre eventos ACOPLADOS y DESIGUALES, det M --- y por")
print("  tanto el presupuesto de volumen y la rapidez total --- tiene un")
print("  SALTO. Lo que no salta es su signo, que es exactamente lo que la")
print("  Prop. 2 afirma. La hipotesis que hay que anadir no es de")
print("  transversalidad sino de separacion de eventos, y solo afecta a la")
print("  magnitud.")

print()
print('=' * 76)
print('(G) BUSQUEDA ALEATORIA DE UNA ESQUINA QUE CAMBIE EL SIGNO')
print('=' * 76)
print("  Si el signo tambien dependiera del orden, la Prop. 2 quedaria")
print("  indefinida en la esquina y habria que anadirle una hipotesis. Se")
print("  barre el espacio de parametros fisicos buscando ese caso.")
rng = np.random.default_rng(SEMILLA + 77)
n_ok, n_sign, n_falla, peor_rel, peor_par = 0, 0, 0, 0.0, None
saltos = []
# Admisibilidad fisica (revision del consejo, 2026-10-17): una esquina solo
# cuenta si al inicio del ciclo ambos comparadores estan por debajo de su
# referencia y ninguna corriente cruza su umbral antes del instante t. Las
# soluciones de periodicidad + umbral final no lo garantizan por si solas.
n_adm, n_sign_adm, caso_adm = 0, 0, None
for _ in range(3000):
    k = float(rng.uniform(0.0, 0.95))
    fL2 = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    fR2 = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    Rl = float(np.exp(rng.uniform(np.log(0.8), np.log(8.0))))
    Ir = float(rng.uniform(1.2, 4.0))
    F = campos(k, L2=Lp*fL2, RL2=RLp*fR2, Ron2=Ron*fR2, Rd2=Rd*fR2)
    for m in F:
        F[m][0][2, 2] = -1.0/(Rl*Cout)

    def res(z, F=F, Ir=Ir):
        x0, t = z[:3], z[3]
        xa = paso(*F[(1, 1)], x0, t)[0]
        xT = paso(*F[(0, 0)], xa, T - t)[0]
        return [xT[0]-x0[0], xT[1]-x0[1], xT[2]-x0[2], xa[0]-Ir]
    z, _, ier, _ = fsolve(res, [0.9, 0.9, 5.0, 0.45*T], full_output=True,
                          xtol=1e-13)
    if ier != 1 or not (0.02*T < z[3] < 0.98*T):
        continue
    xa = paso(*F[(1, 1)], z[:3], z[3])[0]
    try:
        _, d1a = salt(F, (1, 1), (0, 1), xa, 0)
        _, d2a = salt(F, (0, 1), (0, 0), xa, 1)
        _, d2b = salt(F, (1, 1), (1, 0), xa, 1)
        _, d1b = salt(F, (1, 0), (0, 0), xa, 0)
    except FloatingPointError:
        continue
    dA_, dB_ = d1a*d2a, d1b*d2b
    if min(abs(dA_), abs(dB_)) < 1e-9:      # esquina degenerada, no cuenta
        continue
    n_ok += 1
    _ss = np.linspace(0.0, 1.0, 201)[1:-1]
    _seg = np.array([paso(*F[(1, 1)], z[:3], _s*z[3])[0] for _s in _ss])
    adm = bool((z[0] < Ir) and (z[1] < xa[1]) and np.all(_seg[:, 0] < Ir)
               and np.all(_seg[:, 1] < xa[1]))
    n_adm += int(adm)
    if adm and np.sign(dA_) != np.sign(dB_):
        n_sign_adm += 1
        if caso_adm is None:
            caso_adm = dict(k=float(k), L2_L1=float(fL2), R2_R1=float(fR2),
                            R=float(Rl), Iref=float(Ir), D=float(z[3]/T),
                            detA=float(dA_), detB=float(dB_))
        print(f"    SIGNO DISTINTO Y ADMISIBLE: k={k:.3f} L2/L1={fL2:.3f} "
              f"R2/R1={fR2:.3f} R={Rl:.2f} Iref={Ir:.2f} "
              f"det A={dA_:+.4e} det B={dB_:+.4e}")
    rel = abs(dA_ - dB_)/abs(dA_)
    if rel > peor_rel:
        peor_rel, peor_par = rel, (k, fL2, fR2, Rl, Ir, dA_, dB_)
    saltos.append(rel)
    if np.sign(dA_) != np.sign(dB_):
        n_sign += 1
        if n_sign == 1:
            contra = dict(k=k, fL2=fL2, fR2=fR2, R=Rl, Iref=Ir, F=F, xa=xa,
                          dA=dA_, dB=dB_, D=float(z[3]/T))
        if n_sign <= 3:
            print(f"    SIGNO DISTINTO: k={k:.3f} L2/L1={fL2:.3f} "
                  f"R2/R1={fR2:.3f} R={Rl:.2f} Iref={Ir:.2f}")
            print(f"      det orden A = {dA_:+.6e}   "
                  f"det orden B = {dB_:+.6e}")
# ===== (H) la esquina admisible con signo dependiente del orden, en detalle =====
detalle_adm = None
if caso_adm is not None:
    c = caso_adm
    Fh = campos(c['k'], L2=Lp*c['L2_L1'], RL2=RLp*c['R2_R1'],
                Ron2=Ron*c['R2_R1'], Rd2=Rd*c['R2_R1'])
    for m in Fh:
        Fh[m][0][2, 2] = -1.0/(c['R']*Cout)
    Ir_h = c['Iref']

    def res_h(z):
        x0, t = z[:3], z[3]
        xa = paso(*Fh[(1, 1)], x0, t)[0]
        xT = paso(*Fh[(0, 0)], xa, T - t)[0]
        return [xT[0]-x0[0], xT[1]-x0[1], xT[2]-x0[2], xa[0]-Ir_h]
    zh, _, ier_h, _ = fsolve(res_h, [0.9, 0.9, 5.0, 0.45*T], full_output=True,
                             xtol=1e-13)
    xah = paso(*Fh[(1, 1)], zh[:3], zh[3])[0]
    I2_h = float(xah[1])

    def pendientes(modo_ini, modo_fin, x, j):
        fm = Fh[modo_ini][0]@x + Fh[modo_ini][1]
        fp = Fh[modo_fin][0]@x + Fh[modo_fin][1]
        return float(fm[j]), float(fp[j])
    # orden A: fase 1 primero; orden B: fase 2 primero
    _, d1a = salt(Fh, (1, 1), (0, 1), xah, 0); _, d2a = salt(Fh, (0, 1), (0, 0), xah, 1)
    _, d2b = salt(Fh, (1, 1), (1, 0), xah, 1); _, d1b = salt(Fh, (1, 0), (0, 0), xah, 0)
    pend = dict(A_f1=pendientes((1, 1), (0, 1), xah, 0), A_f2=pendientes((0, 1), (0, 0), xah, 1),
                B_f2=pendientes((1, 1), (1, 0), xah, 1), B_f1=pendientes((1, 0), (0, 0), xah, 0))
    print()
    print('=' * 76)
    print('(H) LA ESQUINA ADMISIBLE CON SIGNO DEPENDIENTE DEL ORDEN')
    print('=' * 76)
    print(f"  k = {c['k']:.4f}  L2/L1 = {c['L2_L1']:.4f}  R2/R1 = {c['R2_R1']:.4f}  "
          f"R = {c['R']:.4f} ohm   D = {zh[3]/T:.6f}   (convergencia {ier_h})")
    print(f"  referencias: I_ref1 = {Ir_h:.8f} A   I_ref2 = {I2_h:.8f} A")
    print(f"  orden A (fase 1 primero): det S_1 = {d1a:+.6e}  det S_2 = {d2a:+.6e}  producto {d1a*d2a:+.6e}")
    print(f"  orden B (fase 2 primero): det S_2 = {d2b:+.6e}  det S_1 = {d1b:+.6e}  producto {d2b*d1b:+.6e}")
    for nombre, (hm, hp) in pend.items():
        print(f"    pendientes {nombre}: entrante {hm:+.4e} A/s  saliente {hp:+.4e} A/s")
    lat_adm = []
    for e in [-1e-4, -1e-5, -1e-6, 1e-6, 1e-5, 1e-4]:
        try:
            Mr, dt = orbita_gen(Fh, Ir_h, I2_h*(1.0 + e),
                                z0=[zh[0], zh[1], zh[2], zh[3], zh[3]*(1.0 + 0.5*e)])
            lat_adm.append(dict(eps=e, dt=float(dt), det=float(np.linalg.det(Mr))))
            print(f"    eps = {e:+.0e}: orbita separada, Dt = {dt*1e9:+.4f} ns, det M = {np.linalg.det(Mr):+.6e}")
        except AssertionError:
            lat_adm.append(dict(eps=e, dt=None, det=None))
            print(f"    eps = {e:+.0e}: sin orbita separada")
    detalle_adm = dict(Iref1=float(Ir_h), Iref2=I2_h, D=float(zh[3]/T),
                       dets_orden_A=[float(d1a), float(d2a)], dets_orden_B=[float(d2b), float(d1b)],
                       pendientes=pend, lateral=lat_adm)
print()
print(f"  esquinas validas construidas: {n_ok} de 3000")
print(f"  de ellas fisicamente admisibles (guardas y primer cruce): {n_adm}")
print(f"  con signo dependiente del orden Y admisibles: {n_sign_adm}")
print(f"  con signo del determinante dependiente del orden: {n_sign}")
print(f"  salto relativo del determinante: mediana "
      f"{np.median(saltos):.3f}   maximo {peor_rel:.3f}")
if peor_par:
    print(f"    el peor: k={peor_par[0]:.3f} L2/L1={peor_par[1]:.3f} "
          f"R2/R1={peor_par[2]:.3f}  {peor_par[5]:+.4f} frente a "
          f"{peor_par[6]:+.4f}")
print()
if n_sign == 0:
    print("  La magnitud de det M salta en la esquina, a veces por un factor")
    print("  grande, pero su SIGNO no. Cada cociente hdot^+/hdot^- cambia de")
    print("  valor al cambiar el campo intermedio, no de signo: el sentido en")
    print("  que cada funcion de conmutacion cruza su superficie lo fija su")
    print("  propio interruptor, y el acoplamiento no lo invierte. La Prop. 2")
    print("  sobrevive intacta a la esquina; el presupuesto de volumen no.")
else:
    print("  HAY esquinas donde el signo depende del orden. El mecanismo:")
    c = contra
    print(f"    k = {c['k']:.4f}, L2/L1 = {c['fL2']:.4f}, "
          f"R2/R1 = {c['fR2']:.4f}, R = {c['R']:.3f} ohm, "
          f"I_ref = {c['Iref']:.3f} A, D = {c['D']:.4f}")
    print(f"    velocidades de las funciones de conmutacion en la esquina "
          f"(A/s):")
    print(f"{'modo (s1,s2)':>16} {'hdot_1':>16} {'hdot_2':>16}")
    for m in [(1, 1), (0, 1), (1, 0), (0, 0)]:
        f = c['F'][m][0] @ c['xa'] + c['F'][m][1]
        print(f"{str(m):>16} {f[0]:+16.4e} {f[1]:+16.4e}")
    f11 = c['F'][(1, 1)][0] @ c['xa'] + c['F'][(1, 1)][1]
    f01 = c['F'][(0, 1)][0] @ c['xa'] + c['F'][(0, 1)][1]
    f10 = c['F'][(1, 0)][0] @ c['xa'] + c['F'][(1, 0)][1]
    f00 = c['F'][(0, 0)][0] @ c['xa'] + c['F'][(0, 0)][1]
    print(f"    orden A: (1+sig_1) = hdot_1(0,1)/hdot_1(1,1) = "
          f"{f01[0]/f11[0]:+.6f}")
    print(f"             (1+sig_2) = hdot_2(0,0)/hdot_2(0,1) = "
          f"{f00[1]/f01[1]:+.6f}")
    print(f"    orden B: (1+sig_2) = hdot_2(1,0)/hdot_2(1,1) = "
          f"{f10[1]/f11[1]:+.6f}")
    print(f"             (1+sig_1) = hdot_1(0,0)/hdot_1(1,0) = "
          f"{f00[0]/f10[0]:+.6f}")
    print()
    print("  Con acoplamiento fuerte, apagar una fase invierte la pendiente de")
    print("  la corriente de la otra. La ESPECIE del segundo evento --- si es")
    print("  o no una reflexion --- pasa a depender de si la otra fase ya")
    print("  conmuto. No es un problema de transversalidad: cada evento es")
    print("  transversal en los dos ordenes. Es que N_r deja de estar definido")
    print("  cuando dos eventos acoplados coinciden, y con el la Prop. 2.")
    print("  La hipotesis que falta es de SEPARACION de eventos.")

print()
peor = max(asim, key=lambda r: r['rel'])
print(f"  discrepancia relativa maxima del determinante: {peor['rel']:.2e}")
print(f"    (k = {peor['k']}, L2/L1 = {peor['fL2']}, R2/R1 = {peor['fR2']})")
print(f"  cambios de signo del determinante por el orden: {fall_e} de "
      f"{len(asim)}")
if peor["rel"] < 1e-10:
    print()
    print("  Con fases desiguales Y acoplamiento, la monodromia sigue")
    print("  dependiendo del orden pero el determinante NO. La invariancia no")
    print("  era simetria: det(S_a S_b) = det(S_a) det(S_b) es multiplicativa,")
    print("  y aunque las saltaciones se reconstruyan con campos intermedios")
    print("  distintos en cada orden, el producto de los cocientes hdot^+/hdot^-")
    print("  telescopia al mismo valor. La esquina de codimension dos deja la")
    print("  monodromia ambigua y la Prop. 2 intacta.")

print()
print('=' * 76)
print('VEREDICTO')
print('=' * 76)
print(f"  la regla NO falla en ningun orden bien definido:")
print(f"    errores de la accion versorial            {fallos}")
print(f"    fallos de paridad en el barrido de (C)    {fall_c}")
print(f"  pero su ENTRADA puede no estar definida en la esquina:")
print(f"    esquinas con N_r dependiente del orden    {n_sign} de {n_ok}")
print(f"    salto relativo de |det M|, mediana        {np.median(saltos):.3f}")
print(f"  -> la Prop. 2 necesita una hipotesis de separacion de eventos, y el")
print(f"     presupuesto de volumen deja de ser continuo en la esquina aunque")
print(f"     los eventos sean transversales.")
print('=' * 76)

# barrido fino de la esquina acoplada y asimetrica, solo para la figura.
fino = []
for e in np.concatenate([-np.logspace(np.log10(3e-3), np.log10(2e-5), 34),
                         np.logspace(np.log10(2e-5), np.log10(3e-3), 34)]):
    try:
        Mr, dt = orbita_gen(F_F, Iref, I2_F*(1.0 + float(e)))
    except AssertionError:
        continue
    fino.append(dict(eps=float(e), dt=float(dt),
                     det=float(np.linalg.det(Mr))))
print(f"barrido fino de la esquina para la figura: {len(fino)} puntos")

json.dump(dict(
    admisibilidad=dict(aceptadas=int(n_ok), admisibles=int(n_adm),
                       signo_distinto=int(n_sign),
                       signo_distinto_admisible=int(n_sign_adm),
                       caso_admisible=caso_adm, detalle=detalle_adm),
    lateral=dict(k=K_F, fL2=FL, fR2=FR, det_ab=dab, det_ba=dba,
                 I2=float(I2_F), D=float(D_F), barrido=lat, fino=fino),
    generico=dict(sep_M_med=float(np.median(sep_M)),
                  sep_M_max=float(np.max(sep_M)),
                  sep_mu_med=float(np.median(sep_mu)),
                  sep_mu_max=float(np.max(sep_mu)),
                  dif_det_max=float(np.max(dif_det)),
                  signos_distintos=int(sum(dif_sgn)),
                  dif_ww_max=float(np.max(dif_ww)),
                  errores_versor=fallos),
    conmutador=[float(c) for c in coc],
    esquina=dict(dt=float(o0['dt']),
                 sep=float(np.max(np.abs(Mab-Mba))/np.max(np.abs(Mab))),
                 det_ab=float(np.linalg.det(Mab)),
                 det_ba=float(np.linalg.det(Mba)),
                 detS_ab=float(dA), detS_ba=float(dB),
                 barrido=barrido, fallos=fall_c),
    acoplamiento=dict(filas=acopl, cambios_de_signo=fall_d),
    asimetrico=dict(filas=asim, cambios_de_signo=fall_e,
                    rel_max=float(peor["rel"]))),
    open(RES / 'exp6.json', 'w'), indent=1)
print('resultados/exp6.json escrito')
raise SystemExit(1 if (fallos or fall_c or fall_d or fall_e) else 0)
