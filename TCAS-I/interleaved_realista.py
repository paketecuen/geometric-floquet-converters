import numpy as np
from scipy.linalg import expm
from scipy.optimize import fsolve
from clifford import Cl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

# ================= G(n,n) =================
def alg(n):
    layout, blades = Cl(n, n)
    E = [blades['e%d' % k] for k in range(1, 2*n+1)]
    w = [(E[i] + E[i+n])/2 for i in range(n)]
    v = [(E[i] - E[i+n])/2 for i in range(n)]
    return dict(n=n, E=E, w=w, v=v)

def biv(a, G):
    B = 0*G['E'][0]
    for i in range(G['n']):
        for j in range(G['n']):
            if a[i, j] != 0:
                B = B + 2*a[i, j]*(G['w'][i] ^ G['v'][j])
    return B

def mvexp(B, N=150):
    tot, term = 1 + 0*B, 1 + 0*B
    for k in range(1, N):
        term = term*B/k
        tot = tot + term
        if max(abs(term.value)) < 1e-18:
            break
    return tot

def inv(W):
    c = float((W*~W)(0)); return ~W/c

def accion(W, G):
    Wi = inv(W); n = G['n']
    return np.column_stack([np.array([2*float((G['v'][i] | (W*G['w'][k]*Wi))(0))
                                      for i in range(n)]) for k in range(n)])

def versor_salt(u, nv, eje, G):
    s = float(nv @ u)
    if 1 + s > 0:
        m = np.log(1+s)/s * np.outer(u, nv)
        return mvexp(biv(m, G)/2)
    g = np.zeros((G['n'], G['n'])); g[eje, eje] = np.log(-(1+s))
    return (G['E'][eje]*G['E'][eje+G['n']]) * mvexp(biv(g, G)/2)

# ================= convertidor realista, N fases =================
Vin, C, ESR, R = 12.0, 100e-6, 0.025, 1.5
Ron, Vd, Rd = 0.040, 0.45, 0.020
fs = 100e3; T = 1.0/fs
al = R/(R+ESR)

class Conv:
    def __init__(self, Ls, RLs, Iref):
        self.N = len(Ls); self.Ls, self.RLs, self.Iref = Ls, RLs, Iref
    def A(self, u):
        N = self.N; A = np.zeros((N+1, N+1))
        for k in range(N):
            Rk = self.RLs[k] + (Ron if u[k] else Rd)
            for j in range(N):
                A[k, j] = -al*ESR/self.Ls[k]
            A[k, k] -= Rk/self.Ls[k]
            A[k, N] = -al/self.Ls[k]
            A[N, k] = al/C
        A[N, N] = -al/(R*C)
        return A
    def b(self, u):
        bb = np.zeros(self.N+1)
        for k in range(self.N):
            bb[k] = (Vin if u[k] else -Vd)/self.Ls[k]
        return bb
    def flw(self, u, x, t):
        A, bb = self.A(u), self.b(u)
        M = np.zeros((self.N+2,)*2); M[:-1, :-1] = A; M[:-1, -1] = bb
        E = expm(M*t)
        return E[:-1, :-1]@x + E[:-1, -1]

def u_en(cv, tcs, t):
    """topologia en el instante t (fase k ON desde k*T/N hasta su cruce tcs[k])"""
    u = []
    for k in range(cv.N):
        t_on = k*T/cv.N
        u.append(1 if t_on <= t < tcs[k] else 0)
    return tuple(u)

def orbita(cv, tcs0, x0g):
    N = cv.N
    def tray(x0, tcs, tfin):
        cortes = sorted(set([k*T/N for k in range(N)] + list(tcs) + [tfin, T]))
        t, x = 0.0, np.array(x0, float)
        for tc in cortes:
            if tc <= t:  continue
            if tc > tfin: tc = tfin
            x = cv.flw(u_en(cv, tcs, (t+tc)/2), x, tc-t); t = tc
            if t >= tfin: break
        return x
    def res(z):
        x0, tcs = z[:N+1], z[N+1:]
        r = [tray(x0, tcs, tcs[k])[k] - cv.Iref for k in range(N)]
        return r + list(tray(x0, tcs, T) - x0)
    z, info, ier, msg = fsolve(res, list(x0g)+list(tcs0), full_output=True)
    assert ier == 1, msg
    return z[:cv.N+1], z[cv.N+1:]

def prediccion(cv, tcs0, x0g):
    N = cv.N
    x0, tcs = orbita(cv, tcs0, x0g)
    orden = np.argsort(tcs)
    # tramos y saltaciones en orden temporal
    eventos = sorted([k*T/N for k in range(1, N)] + list(tcs) + [T])
    fact_mat, fact_ver = [], []
    G = alg(N+1)
    t = 0.0
    x = x0.copy()
    for te in eventos:
        u = u_en(cv, tcs, (t+te)/2)
        Aseg = cv.A(u)
        fact_mat.append(expm(Aseg*(te-t)))
        fact_ver.append(mvexp(biv(Aseg*(te-t), G)/2))
        x = cv.flw(u, x, te-t); t = te
        for k in range(N):
            if abs(te - tcs[k]) < 1e-15:
                u_m = list(u); u_m[k] = 1; u_p = list(u); u_p[k] = 0
                fm = cv.A(tuple(u_m))@x + cv.b(tuple(u_m))
                fp = cv.A(tuple(u_p))@x + cv.b(tuple(u_p))
                nk = np.zeros(N+1); nk[k] = 1.0
                uk = (fp-fm)/(nk@fm)
                fact_mat.append(np.eye(N+1) + np.outer(uk, nk))
                fact_ver.append(versor_salt(uk, nk, k, G))
    M = np.eye(N+1); W = 1 + 0*G['E'][0]
    for Fm, Fv in zip(fact_mat, fact_ver):
        M = Fm @ M; W = Fv * W
    return x0, tcs, M, W, G

# ================= simulador temporal independiente =================
def simula(cv, x_ini, ciclos, pasos=2100):
    dt = T/pasos
    sched = {int(round(k*pasos/cv.N)): k for k in range(cv.N)}
    Es = {}
    def Estep(u):
        if u not in Es:
            A, bb = cv.A(u), cv.b(u)
            M = np.zeros((cv.N+2,)*2); M[:-1, :-1] = A; M[:-1, -1] = bb
            Es[u] = expm(M*dt)
        return Es[u]
    x = np.append(x_ini, 1.0)
    on = [False]*cv.N
    out = []
    for c in range(ciclos):
        for p in range(pasos):
            if p in sched:
                on[sched[p]] = True
            u = tuple(1 if o else 0 for o in on)
            xn = Estep(u) @ x
            for k in range(cv.N):
                if on[k] and xn[k] >= cv.Iref:
                    hp, hn = x[k]-cv.Iref, xn[k]-cv.Iref
                    fr = hp/(hp-hn)
                    xc = x + fr*(xn-x)
                    on[k] = False
                    u2 = tuple(1 if o else 0 for o in on)
                    A, bb = cv.A(u2), cv.b(u2)
                    M = np.zeros((cv.N+2,)*2); M[:-1, :-1] = A; M[:-1, -1] = bb
                    xn = expm(M*(1-fr)*dt) @ xc
            x = xn
        out.append(x[:-1].copy())
    return np.array(out)

def monodromia_empirica(cv, x0g, tcs0):
    n = cv.N + 1
    conv = simula(cv, x0g, 260)
    xstar = conv[-1]
    esc = np.array([0.01]*cv.N + [0.05])
    X, Y = [], []
    for d in range(n):
        for sg in (+1, -1):
            dx = np.zeros(n); dx[d] = sg*esc[d]
            tr = simula(cv, xstar+dx, 13) - xstar
            for k in range(len(tr)-1):
                X.append(tr[k]); Y.append(tr[k+1])
    X, Y = np.array(X).T, np.array(Y).T
    return Y @ np.linalg.pinv(X), xstar

# ================= los dos convertidores =================
cfg = {
 2: dict(cv=Conv([100e-6, 106e-6], [0.060, 0.052], 1.60),
         tcs0=[4.0e-6, 9.0e-6], x0g=[1.30, 1.45, 4.2]),
 3: dict(cv=Conv([100e-6, 104e-6, 97e-6], [0.060, 0.055, 0.065], 0.86),
         tcs0=[3.0e-6, 6.4e-6, 9.7e-6], x0g=[0.60, 0.70, 0.83, 3.2]),
}
resul = {}
for N, c in cfg.items():
    x0, tcs, M, W, G = prediccion(c['cv'], c['tcs0'], c['x0g'])
    Memp, xstar = monodromia_empirica(c['cv'], c['x0g'], c['tcs0'])
    mu_p, mu_e = np.linalg.eigvals(M), np.linalg.eigvals(Memp)
    resul[N] = (mu_p, mu_e, M, Memp, W)
    print("="*74)
    print(f"{N} fases (parasitos + tolerancias de L y RL):  D = {tcs[0]/T:.4f}")
    print(f"  |accion(W)-M| = {np.max(np.abs(accion(W, G)-M)):.2e}"
          f"    W~W = {float((W*~W)(0)):+.1f}")
    print(f"  mu predichos (versores) = {np.sort_complex(mu_p)}")
    print(f"  mu ajustados (simulac.) = {np.sort_complex(mu_e)}")
    print(f"  det M: predicho {np.linalg.det(M):+.4f}   empirico {np.linalg.det(Memp):+.4f}")

# ================= figura =================
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), dpi=150)
th = np.linspace(0, 2*np.pi, 400)
for ax, N, tit in zip(axes, (2, 3), ('2 fases (paridad PAR)', '3 fases (paridad IMPAR)')):
    mu_p, mu_e = resul[N][0], resul[N][1]
    ax.plot(np.cos(th), np.sin(th), '-', color='0.6', lw=0.9)
    ax.axhline(0, color='0.85', lw=0.7); ax.axvline(0, color='0.85', lw=0.7)
    ax.plot(mu_p.real, mu_p.imag, 'x', ms=11, mew=2.4, color='#c0392b',
            label='predicho (versores)')
    ax.plot(mu_e.real, mu_e.imag, 'o', ms=7, mfc='none', mew=1.6, color='#1a5276',
            label='ajustado (simulacion)')
    ax.set_title(f"{tit}\ndet M = {np.linalg.det(resul[N][2]):+.3f},  "
                 f"$W\\widetilde{{W}}$ = {float((resul[N][4]*~resul[N][4])(0)):+.0f}",
                 fontsize=10)
    ax.set_xlabel('Re $\\mu$'); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal'); ax.grid(alpha=0.2)
axes[0].set_ylabel('Im $\\mu$')
axes[0].legend(loc='lower left', fontsize=8)
fig.suptitle('Multiplicadores de Floquet: interleaved realista, prediccion vs simulacion',
             fontsize=11)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/paridad_interleaved_realista.png')
print("\nFigura guardada.")
