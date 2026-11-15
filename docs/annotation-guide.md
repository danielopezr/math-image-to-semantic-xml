# Guía de anotación (borrador)

Estado: **propuesta pendiente de acuerdo con el equipo.** Las decisiones marcadas
como *pendiente* deben cerrarse antes de usar las métricas para comparar
modelos, porque hoy el F1 de intents mide sobre todo la diferencia de
convención entre etiquetas y pipeline.

La guía aplica por igual a las etiquetas del dataset
(`image-to-mathML-dataset`), al anotador determinista, al LLM y a las métricas.

## 1. Productos y sumas (decidido en el código, configurable)

- Los operadores asociativos `times`, `plus`, `and`, `or`, `union` e
  `intersection` se escriben **planos, con todos sus argumentos**:
  `times($a,$b,$c)`, no `times($a,times($b,$c))`.
- `minus`, `divided-by` y `power` llevan **exactamente 2** argumentos: su orden
  y su agrupación cambian el significado.
- Las métricas tratan como equivalentes la forma plana y la anidada, así que
  una etiqueta anidada no se penaliza; aun así, la forma plana es la
  recomendada.
- Validador: `IMAGE_TO_MATHML_NARY_ASSOCIATIVE` (por defecto `true`). No se pudo
  confirmar en la especificación W3C de intent que un infijo admita más de dos
  argumentos (Content MathML sí lo permite); si se confirma lo contrario,
  basta con desactivarlo y el pipeline anidará por pares.

## 2. Multiplicación implícita — *pendiente*

¿Se anota `times` cuando no hay un operador visible (`2x`, `i(1+r)`)?

- Las etiquetas actuales lo hacen (132 veces en 75 ecuaciones).
- El pipeline puede hacerlo con una regla determinista
  (`IMAGE_TO_MATHML_IMPLICIT_TIMES=true`). Con ella, sobre las 75 ecuaciones de
  la evaluación nocturna, el F1 de intents de la línea base sin LLM pasa de
  0,29 a 0,51.

Qué cuenta como factor (propuesta, ya implementada en la regla):

| Expresión | Anotación |
|---|---|
| `2x` | `times(2, x)` |
| `2x^2y` | `times(2, power(x,2), y)` |
| `i(1+r)^t(1+r)` | `times(i, power((1+r),t), (1+r))` |
| `2f(x)` | `times(2, f(x))` |
| `a(b+c)` | `times(a, (b+c))` (no es la función `a`) |
| `\int 2x\,dx` | `times(2, x)`; el `dx` no es factor |
| `\sin^2 x` | sin `times`: es una función aplicada |

## 3. Elementos con lectura por defecto — *pendiente*

¿Se anotan `fraction` en cada `mfrac` e `indexed-by` en cada `msub`?

- Las etiquetas casi nunca lo hacen; el anotador del pipeline siempre.
- Opción A: anotarlos siempre (explícito, más largo).
- Opción B: solo cuando la lectura no es la por defecto (por ejemplo `a/b` en
  línea), como pide el prompt del LLM. Requiere una opción para que el anotador
  deje de marcarlos.

## 4. Notación ambigua según el área — *pendiente*

- `(x, y)` en álgebra lineal: ¿`inner-product` o `ordered-pair`? Las etiquetas
  usan `inner-product` cuando el tema es ortogonalización.
- Funciones con subíndice como `y_0(t)`: las etiquetas las anotan como
  `$name($var)`.
- Derivadas con primas (`y'`, `y''`, `y^{(4)}`): las etiquetas usan
  `derivative($y, _x, n)`, con la variable como literal `_x` cuando no aparece
  en la expresión.

## 5. Reglas generales (ya aplicadas en el dataset corregido)

- Un único `<mrow>` raíz dentro de `<math>`, sin atributo `display`.
- Cada `$ref` tiene exactamente un descendiente con `arg="ref"` en su alcance.
- Solo nombres del registro W3C; un nombre que no es concepto se escribe como
  literal con `_` (`_x`).
- Lugar de la anotación según W3C: `definite-integral` y `limit` en el
  operador con sus límites; `sum` y `product` en el grupo con límites y
  cuerpo; delimitadores (`|x|`, `‖v‖`, `⌊x⌋`, `⟨u,v⟩`) y `n!` en el grupo;
  `power` solo en `msup`.
