# Campaña de validación

Separada a propósito de `figs/`: ahí viven los scripts que **producen** las
figuras y los números del manuscrito; aquí los que **intentan romper** la
teoría. No se mezclan.

| script | qué intenta romper |
|---|---|
| `exp1_falsacion_paridad.py` | la regla de paridad $\operatorname{sign}(\det M)=(-1)^{N_r}$ y la identidad versorial, sobre un ensemble aleatorio de sistemas conmutados |
| `exp2_degeneraciones.py` | el comportamiento en las fronteras donde la teoría *debería* fallar: $\sigma\to-1$, $\sigma\to0$, grazing, $S_k$ defectuosa |
| `exp3_deadbeat.py` | la predicción nueva: $\det S=0\iff m_a=m_2$, con anulación a primer orden en un ciclo de la dirección $\ker M$ (no de una perturbación pura de corriente) |
| `exp4_orden4.py` | la forma cerrada general, con una saltación oblicua de verdad (buck con filtro LC de entrada, $n=4$) |
| `exp5_retardo_digital.py` | el retardo de cálculo del control digital, declarado como limitación en la Discusión; planta de la Sec. V-A con ESR, regulada a 5 V |
| `exp6_eventos_simultaneos.py` | la hipótesis de separación: dos conmutaciones que coinciden en el tiempo, con filtro de admisibilidad del latch (guardas al inicio del ciclo y primer cruce) |
| `exp7_modo_tension.py` | (R1) un convertidor sin latch: buck en modo tensión, evento PWM con $\sigma=0$, $\det M=e^{-T/RC}$ constante; reproduce la Tabla I de Giaouris 2008 y localiza el flip con el escalar |
| `exp8_boost.py` | (R1) otra topología: boost con control por corriente de pico, salto de campo oblicuo en $n=2$; frontera de flip, forma cerrada del hueco y simulación temporal independiente |
| `exp9_sensado.py` | (R1) la dependencia de $N_r$ con el nivel de detalle del modelo: con filtro de sensado el evento pasa a ser de paso y el signo lo lleva un multiplicador rápido |
| `exp10_coste_representaciones.py` | (R1) el coste $4^n$: multivector genérico, representación espinorial, bloques por grado, versor factorizado y matrices, con un mismo núcleo compilado |

Resultados en `resultados/*.json`, con semilla fija para que sean
reproducibles. Cada script imprime un veredicto y sale con código distinto de
cero si encuentra un contraejemplo.

Ejecutar:

```bash
PY=~/venvs/floquet_GA/bin/python     # entorno creado fuera de la carpeta sincronizada
$PY exp1_falsacion_paridad.py
$PY exp2_degeneraciones.py
$PY exp3_deadbeat.py
$PY exp4_orden4.py
$PY exp5_retardo_digital.py
$PY exp6_eventos_simultaneos.py
$PY exp7_modo_tension.py
$PY exp8_boost.py
$PY exp9_sensado.py
$PY exp10_coste_representaciones.py
```

`nuga/` contiene los programas (C# y MATLAB) que miden el versor factorizado con la biblioteca
de A. H. Eid; esa biblioteca no forma parte de este repositorio. Sus salidas están en
`resultados/exp10_nuga_*.txt`.

## Los dos que cambiaron el manuscrito

**Exp. 5.** El retardo no saca la teoría de su marco: vuelve *parabólica* la
saltación ($\sigma=0$ exacto, porque el disparo lo fija un registro y no el
estado) y mete el muestreador como un factor **de rango uno** más,
$\Sigma=I+u_\Sigma n_\Sigma^{T}$ con $\det\Sigma=K\,e_v^{T}P_x^{-1}P_d$.
Es decir: un factor de rango uno más, con las mismas especies. La ganancia del lazo
decide la especie, y con cualquier $K>0$ el canal de flip queda forzado
(con la ESR declarada, $K^*=0.2531$ por Neimark–Sacker).
Lo que era una limitación declarada es ahora una extensión.

**Exp. 6.** Aquí sí hay un límite real. Con las funciones de conmutación
desacopladas las saltaciones **conmutan exactamente** y la esquina es
removible. Con inductores acoplados y fases desiguales, $\det M$ salta al
cruzar la coincidencia (los dos límites laterales son las dos candidatas), y
el signo puede depender del orden, pero solo en casos físicamente admisibles
muy concretos: de 3000 ensayos, 1269 esquinas resuelven periodicidad y umbral,
1029 pasan el filtro de admisibilidad (ambos comparadores por debajo de su
referencia al inicio del ciclo y ningún cruce anterior en el tramo ON–ON) y
exactamente **una** tiene $N_r$ dependiente del orden, con un factor de
saltación a $10^{-4}$ de la frontera singular. Los «2 de 177» de la primera
versión no pasaban el filtro y se retiraron. La regla acierta el signo en cada
orden; lo que deja de estar definido es $N_r$. De ahí la hipótesis de
separación de eventos que lleva la Prop. 2.

## Exp. 11: la mitad par de la regla de paridad (revisión R1, 20-11-2026)

Carpeta `exp11_par_flip/`, con sus salidas en `exp11_par_flip/resultados/`.
Los scripts se importan entre sí por nombre, así que se ejecutan desde esa
carpeta.

Con dos fases $N_r=2$ y la regla de paridad permite, sin forzarlo, un par de
multiplicadores reales negativos. El par flip es
$\mu_\pm=\bar\mu\pm\sqrt{(\Delta\mu)^2/4-\chi^2}$: complejo si y solo si
$|\Delta\mu|<2\chi$, con $\Delta\mu$ la diferencia entre los multiplicadores
flip de cada fase por separado y $\chi$ la interferencia entre las dos
reflexiones ($\gamma^2T^2/(4C\sqrt{L_1L_2})$ en la frontera subarmónica).
El signo de $\chi^2$ es el del tipo de evento: dos reflexiones separan el par
fuera del eje real; dos eventos de paso, a lo largo del eje.

| script | qué hace |
|---|---|
| `par_flip_interleaved.py` | monodromía, reducción al bloque de corrientes, umbral, efecto de la rampa |
| `par_flip_forma_cerrada.py`, `numeros_texto.py` | forma cerrada frente a monodromía (Tabla S5) y los números del texto |
| `par_flip_tipo.py` | reflexión frente a paso al cruzar la pendiente deadbeat (Tabla S6) |
| `sim_dosfases.py` | simulador temporal por eventos, sin matrices de saltación |
| `td1_validacion.py` | el jacobiano del mapa temporal coincide con la monodromía a $3\cdot10^{-8}$ |
| `td2_atractor.py` | atractor más allá de la frontera, 8 desajustes por 7 distancias (Tabla S8) |
| `td3_umbral.py` | barrido fino del desajuste: el batido se apaga entre 1,0150 y 1,0155 (Tabla S7, Fig. 5) |
| `par_flip_giaouris.py` | multiplicadores publicados del boost de dos fases de Giaouris et al. (Tabla S9) |

