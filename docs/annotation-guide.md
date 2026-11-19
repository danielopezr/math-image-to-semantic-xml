# Guía de anotación

Estado: **acordada por el equipo (2026-11-15).** Aplica por igual a las
etiquetas del dataset (`math-intent-mathml-dataset`, que guarda la versión
en inglés de esta guía), al anotador determinista,
al LLM y a las métricas. Cada regla indica la opción que la controla en el
código, por si hay que revisarla.

Principio general: **se anota lo que cambia o aclara la lectura; lo que ya se
lee bien por defecto no se anota.** Un intent que repite la lectura por
defecto no ayuda al lector de pantalla y es otro sitio donde equivocarse.

## 1. Multiplicación implícita: se anota

Los factores contiguos sin operador visible se anotan con `times`, mediante
una regla determinista (`IMAGE_TO_MATHML_IMPLICIT_TIMES`, activada).

| Expresión | Anotación |
|---|---|
| `2x` | `times(2, x)` |
| `2x^2y` | `times(2, power(x,2), y)` |
| `i(1+r)^t(1+r)` | `times(i, power((1+r),t), (1+r))` |
| `2f(x)` | `times(2, f(x))` |
| `2n!` | `times(2, factorial(n))` |
| `a(b+c)`, `n(n+1)` | `times(a, (b+c))`: una expresión entre paréntesis es un factor |
| `\int 2x\,dx` | `times(2, x)`; el `dx` no es factor |

No son factores: el `d` de un diferencial, las funciones con nombre
(`sin`, `sin^2`), los nombres de operador (`dim`, `ker`), los operadores
grandes y los límites. Dos números seguidos nunca son un producto (son
dígitos o índices, como el "1 1" de `a_{11}`), y en un subíndice las letras
contiguas son una lista de índices (`a_{ij}`), no un producto; un coeficiente
numérico sí lo marca (`a_{2n}`). El `d` de `dt^n` en `d^n/dt^n` tampoco es
factor.

## 2. Productos y sumas: forma plana

- Los operadores asociativos `times`, `plus`, `and`, `or`, `union` e
  `intersection` se escriben **planos, con todos sus argumentos**:
  `times($a,$b,$c)`, no `times($a,times($b,$c))`.
- `minus`, `divided-by` y `power` llevan **exactamente 2** argumentos.
- Las métricas tratan como equivalentes la forma plana y la anidada.
- Opción: `IMAGE_TO_MATHML_NARY_ASSOCIATIVE` (activada). No se pudo confirmar
  en la especificación W3C de intent que un infijo admita más de dos
  argumentos (Content MathML sí lo permite); si se confirma lo contrario,
  basta con desactivarla y el pipeline anidará por pares.

## 3. Funciones aplicadas

- `f(x)`, `g(t)`, `f_i(x)`, `f'(x)` y nombres de operador (`\operatorname{sgn}`)
  son aplicaciones: `$name($var)`.
- Un símbolo seguido de un paréntesis con **una sola variable** también es una
  función: `T(x)`, `Q(x)`, `y_0(t)`.
- Un paréntesis con comas, `T(x, y)`, es una lista de argumentos: no es un
  producto ni se anota como par.

## 4. Lecturas por defecto: no se anotan

Una fracción en `mfrac`, un subíndice en `msub` y una raíz en `msqrt`/`mroot`
**no llevan intent**: su lectura por defecto ya es la correcta. Sí se anotan:

- la fracción en línea `a/b` → `fraction($num,$den)`;
- las potencias → `power($base,$exp)` (en `msup`; nunca sobre `sin` en
  `sin^2 x` ni sobre un `msubsup` sin anidar el índice);
- `e^x` → `exponential($exp)`;
- los usos no habituales de un `mfrac`, como el coeficiente binomial.

Opción: `IMAGE_TO_MATHML_ANNOTATE_DEFAULTS` (desactivada). El pipeline también
elimina esos intents si los añade el LLM.

## 5. Notación que depende del tema

- `⟨u, v⟩` es siempre `inner-product`.
- `(u, v)` es `inner-product` **solo si el tema** habla de producto interno,
  ortogonalidad u ortogonalización (Gram-Schmidt); en otro caso se deja sin
  anotar. Opción: `IMAGE_TO_MATHML_INNER_PRODUCT_PAIRS=auto|on|off`.
- Por eso el tema de cada imagen debe estar rellenado (regla 8).

## 6. Derivadas con primas

`y'`, `y''`, `y'''` e `y^{(n)}` se anotan como `derivative($y, _x, n)`, con la
variable como **literal** (`_x`) porque no aparece en la expresión. Se aplica
en contexto de cálculo o ecuaciones diferenciales
(`IMAGE_TO_MATHML_DERIVATIVE_PRIMES=auto|on|off`; la variable se cambia con
`IMAGE_TO_MATHML_DERIVATIVE_VARIABLE`). La prima puede escribirse junto a su
base o como superíndice: las métricas tratan ambas formas como iguales.

## 7. Otras lecturas

- `n!`, `5!` y `(n-k)!` son `factorial($var)`, en el grupo que forman el
  operando y el `!`.
- Los puntos suspensivos `…` y `⋯` son un operador (`<mo>`) con
  `intent="ellipsis"`.
- `x → a`, `x → 0`, `x → +∞` (letra minúscula, flecha y valor) son
  `tends-to($var,$value)`; una aplicación como `f: A → B` no.

## 8. Tema y complejidad de cada imagen

El tema (la sección del libro, por ejemplo "orthogonal projections") y la
complejidad de cada imagen están en `metadata/<area>.csv` del dataset, en
inglés. `tools/build_eval_csv.py` los copia al CSV de evaluación del área en
`eval/` (versionado, para que todos evalúen las mismas muestras), que solo
añade el timeout.

## 9. Reglas generales

- Un único `<mrow>` raíz dentro de `<math>`, sin atributo `display`.
- Cada `$ref` tiene exactamente un descendiente con `arg="ref"` en su alcance.
- Solo nombres del registro W3C; un nombre que no es concepto se escribe como
  literal con `_` (`_x`).
- Lugar de la anotación según W3C: `definite-integral` y `limit` en el
  operador con sus límites; `sum` y `product` en el grupo con límites y
  cuerpo; delimitadores (`|x|`, `‖v‖`, `⌊x⌋`, `⟨u,v⟩`) y `n!` en el grupo.

## Efecto medido

Sobre 75 ecuaciones del dataset (OCR + anotador determinista, sin LLM), frente
a las etiquetas corregidas:

| | Convención anterior | Esta guía |
|---|---|---|
| F1 de intents | 0,36 | 0,81 |
| BLEU de estructura | 0,74 | 0,84 |
| BLEU con intents | 0,56 | 0,78 |

## Pendiente

- Rellenar el tema de las 39 imágenes etiquetadas que no lo tienen.
- Decidir si se anotan las integrales definidas, los productos con `×` o `·`
  visible y los operadores aplicados a una expresión (`T(e_k)`, `D(1+x)`).
- Etiquetar lógica matemática y probabilidad, y completar matemáticas
  discretas.
- Verificar en la especificación W3C los argumentos múltiples de `times`.
