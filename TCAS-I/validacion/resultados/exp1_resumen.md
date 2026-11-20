# Exp. 1 — Falsación de la paridad, régimen genérico

Semilla 20260726. `|1+sigma| >= 0.05`, n = 2..8, hasta 6 eventos por periodo.

```
BLOQUE A  (matrices, 20000 ensayos, 20 s)
  (1) sign(det M) analitico != (-1)^Nr .......... 0
  (2) producto ensamblado != det analitico ...... 0   peor error rel. 6.3e-07
  (3) paridad de multiplicadores negativos ...... 0
  (4) Nr impar sin multiplicador real negativo .. 0
  descartados por cond(M) >= 1e10 ............... 224  (1.1 %)
  casos PERMITIDOS (Nr par, >=2 mult. reales <0) . 4891  (24 %)

BLOQUE B  (algebra G(2,2)..G(4,4), 200 ensayos, 463 s)
  max |accion(W) - M| relativo .................. 5.70e-11
  max desvio de |W~W| respecto de 1 ............. 5.63e-11
  contraejemplos ................................ 0
```

## Dos correcciones al propio experimento, antes de dar los números por buenos

La primera pasada arrojó 967 «contraejemplos». **Ninguno lo era.** Dos errores
de diseño míos:

1. **Sembraba sigma a 1e-9 de la frontera de clamping.** Con seis eventos así,
   `det M ~ 1e-54`: cero numérico. Pedirle el signo a eso no falsa nada, mide
   underflow. Ese estudio es legítimo pero es otro experimento — está en
   `exp2`.
2. **Pedía el signo del determinante al producto ensamblado.** La teoría lo da
   en forma cerrada,
   `det M = exp(sum tr(A_i) tau_i) * prod (1+sigma_k)`,
   así que la verdad de terreno es esa, no el LU. Con espectros que abarcan
   decadas (autovalores de 1e5 a 1e-5) `det M` es 1e-22 veces `||M||^n` y el
   LU pierde el signo.

Además la clasificación de autovalores usaba una tolerancia escalada con
`max|mu|`, que descarta como «no negativos» los autovalores pequeños de un
espectro rígido. Ahora el criterio es relativo a cada autovalor.

## Lo que sí es un resultado

- **La regla es exacta, no aproximada.** Por la factorización, el signo del
  determinante es `prod sign(1+sigma_k) = (-1)^{N_r}` idénticamente.
- **La identidad versorial aguanta a 6e-11** con `u` y `n` genéricos, no
  alineados con ejes, en tres álgebras distintas. Esto es lo que el manuscrito
  solo había comprobado en un puñado de casos.
- **4891 casos (24 %) con paridad par y dos o más multiplicadores reales
  negativos.** Esto sustancia empíricamente el enunciado corregido de la
  Prop. 2: la paridad par *permite* multiplicadores negativos, y no de forma
  excepcional. La cautela que introdujimos tras la revisión no era retórica.
- **Argumento nuevo a favor del método**, que el manuscrito no hace: en el
  1.1 % de casos con `cond(M) >= 1e10` el signo del determinante **no es
  representable** desde la monodromía ensamblada, mientras el recuento entero
  de eventos lo da exacto y gratis. La ruta espectral no puede competir ahí.
