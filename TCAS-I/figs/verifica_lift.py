"""Verificación numérica de la Prop. 1 en su forma de lift (eq. lift):
  Q_S = 1 + u n  (lifts nulos)  cumple  Q_S Q_S~ = 1 + sigma = det S  y
  Q_S x Q_S^{-1} = S x  para det S != 0,  en las tres especies, con u y n
  GENERICOS (sin alinear con ejes) en G(3,3); y es proporcional a las formas
  exponenciales del Apéndice A con c = +-sqrt|1+sigma|."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'validacion'))
import numpy as np
from comun import algebra, versor_salt, salt_desde_sigma

rng = np.random.default_rng(7)
n = 3
G = algebra(n)
lift_u = lambda u: sum(float(u[i])*G['w'][i] for i in range(n))
lift_n = lambda v: sum(float(v[j])*G['v'][j] for j in range(n))


def accion(W):
    c = float((W*~W)(0)); Wi = ~W/c
    return np.column_stack([np.array([2*float((G['v'][i] | (W*G['w'][k]*Wi))(0))
                                      for i in range(n)]) for k in range(n)])


peor = 0.0
for sigma, nombre in [(0.0, 'parabolico (i)'), (0.7, 'paso (i)'), (-0.4, 'paso (i)'),
                      (-1.67, 'reflexion (ii)'), (-2.9, 'reflexion (ii)'), (50.0, 'paso, casi grazing')]:
    S, u, nv = salt_desde_sigma(rng, n, sigma)
    Q = G['uno'] + lift_u(u)*lift_n(nv)
    QQ = Q*~Q
    e_norma = abs(float(QQ(0)) - (1 + sigma)) + float(np.max(np.abs((QQ - QQ(0)).value)))
    e_accion = float(np.max(np.abs(accion(Q) - S)))
    Wp, esp = versor_salt(u, nv, G)                 # forma exponencial (Apéndice A)
    r = Q*~Wp; c = float(r(0))
    e_prop = float(np.max(np.abs((r - c).value))) + abs(abs(c) - np.sqrt(abs(1 + sigma)))
    peor = max(peor, e_norma, e_accion, e_prop)
    print(f"{nombre:20s} sigma={sigma:+8.4f}  |QQ~-(1+s)|={e_norma:.1e}  |accion(Q)-S|={e_accion:.1e}"
          f"  Q = c W_exp con c={c:+.6f} (sqrt|1+s|={np.sqrt(abs(1+sigma)):.6f}), resto {e_prop:.1e}")
# especie singular: idempotente
S, u, nv = salt_desde_sigma(rng, n, -1.0)
Q = G['uno'] + lift_u(u)*lift_n(nv)
e_idem = float(np.max(np.abs((Q*Q - Q).value)))
print(f"singular (iii)       sigma=-1: |Q^2-Q|={e_idem:.1e}  QQ~={float((Q*~Q)(0)):+.1e}  rango S={np.linalg.matrix_rank(S)} de {n}")
assert peor < 1e-12 and e_idem < 1e-12, (peor, e_idem)
print("LIFT 1 + u n VERIFICADO")
