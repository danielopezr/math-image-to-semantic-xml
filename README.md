# math-semantic-ocr

Convierte la imagen de una expresión matemática en:
LaTeX (pix2tex) → MathML de presentación (latex2mathml) → interpretación semántica (LLM con salida estructurada, opcional) → documento XML validado (lxml + XSD) → evaluación con un catálogo de reglas.

Cada etapa se ejecuta y se prueba por separado. Si una etapa falla, las dependientes se omiten y el resultado final **no** se presenta como validado.

## Arquitectura

```
imagen ─► ocr.py ─► converter.py ─► semantic_agent.py ─► intent_rules.py ─► xml_builder.py ─► xml_validator.py
          pix2tex    latex2mathml    LLM (opcional)       catálogo JSON      lxml              XSD + coherencia
                         └───────────── pipeline.py (orquesta y guarda resultados intermedios) ─────────────┘
```

| Módulo | Responsabilidad |
|---|---|
| `app/models.py` | Contratos de datos (Pydantic). |
| `app/ocr.py` | `recognize_formula`, revisión manual y repetición del OCR. |
| `app/converter.py` | `latex_to_mathml`; distingue error de sintaxis LaTeX de error de conversión. |
| `app/semantic_agent.py` | Interfaz `SemanticProvider` (Ollama local o API remota), validación del JSON. |
| `app/intent_rules.py` + `app/rules_catalog.json` | `evaluate_intents` y catálogo versionado de reglas. |
| `app/xml_builder.py`, `app/xml_validator.py`, `app/schemas/result.xsd` | Construcción y validación del XML. |
| `app/pipeline.py` | Orquestación y resultados intermedios en `outputs/<run_id>/`. |
| `app/evaluation.py` | Resumen por etapa (sin puntuación agregada). |
| `main.py` | Interfaz de línea de comandos. |

La CLI y una futura interfaz web solo deben llamar a `app.pipeline.run_pipeline(...)`.

## Instalación en Windows (PowerShell)

Requiere Python 3.10 o 3.11 (recomendado 3.11).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # si se bloquea: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip

# 1) PyTorch para CPU (pix2tex lo necesita; comprueba el comando actual en https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Resto de dependencias
pip install -r requirements.txt

# 3) Configuración local
copy .env.example .env

# 4) Genera el esquema JSON del agente semántico
python -m app.export_schema

# 5) Fija las versiones realmente instaladas
pip freeze > requirements.lock.txt
```

La primera ejecución del OCR descarga los pesos del modelo de pix2tex (necesita conexión a Internet).

Para probar sin OCR ni PyTorch: instala solo `pydantic Pillow lxml latex2mathml pytest python-dotenv` y usa `--latex`.

## Uso

```powershell
python main.py ruta\formula.png                      # OCR + MathML + reglas, sin LLM
python main.py ruta\formula.png --review             # revisar/corregir/repetir el OCR
python main.py --latex "\frac{a}{b}"                 # sin imagen ni OCR
python main.py ruta\formula.png --export salida.xml  # copia el XML final
python main.py ruta\formula.png --expected-latex "x=0"   # mide la exactitud del OCR
python main.py ruta\formula.png --provider ollama    # con LLM local
python main.py                                       # modo interactivo
```

Código de salida: `0` si el resultado quedó validado, `1` si no, `2` si hubo un error de uso.

### Agente semántico (opcional)

- **Local (sin coste por llamada):** instala [Ollama](https://ollama.com), descarga un modelo y define `OLLAMA_MODEL` en `.env`. Requiere hardware suficiente; los modelos pequeños interpretan peor.
- **Remoto:** define `LLM_BASE_URL`, `LLM_MODEL` y `LLM_API_KEY` en `.env` (API de chat tipo OpenAI). **Puede tener coste por token**; cada fórmula envía el esquema, el LaTeX y el MathML, y un reintento duplica el consumo. Consulta las tarifas de tu proveedor. La clave nunca debe estar en el código ni en el control de versiones.

## Pruebas y evaluación por etapa

```powershell
pytest -v
pytest tests\test_ocr.py -v          # etapa 1
pytest tests\test_converter.py -v    # etapa 2
pytest tests\test_semantics.py -v    # etapa 3
pytest tests\test_xml.py -v          # etapa 4
pytest tests\test_intent_rules.py -v # etapa 5
pytest tests\test_pipeline.py -v     # integración
```

Las pruebas usan componentes falsos (OCR, conversor, LLM), por lo que no requieren pix2tex ni red. Las pruebas de `test_converter.py` que usan la librería real se omiten si no está instalada.

Se evalúa cada aspecto **por separado**; no existe puntuación única:

| Aspecto | Cómo se mide |
|---|---|
| Exactitud del OCR | `--expected-latex`: coincidencia exacta y tasa de error de caracteres (sin espacios). No reconoce equivalencias de LaTeX. |
| Validez de la conversión | Estado de `converter` y comprobación del MathML (buena formación, espacio de nombres, solo presentación). |
| Validez estructural de la respuesta del LLM | Validación contra `SemanticOutput` y coherencia interna (ids, relaciones). |
| Calidad de la interpretación semántica | **Sin métrica automática fiable.** Se informan las reglas semánticas y la marca de revisión; la valoración final es humana. |
| Cumplimiento de reglas | Resultados `PASS/FAIL/REVIEW/NOT_APPLICABLE`, agrupados por resultado y por origen. |
| Validez del XML final | Buena formación, XSD y coherencia entre campos. |

## Documento XML

- Espacio de nombres del proyecto: `urn:math-semantic-ocr:result:1` (no es un estándar del W3C). El MathML se incrusta en `http://www.w3.org/1998/Math/MathML`.
- Solo se guarda el **nombre** del archivo de imagen, no su ruta.
- `validation/@overall` es `VALIDATED` únicamente si todas las etapas están `OK` y no hay errores; en otro caso es `NOT_VALIDATED` y se listan los motivos.
- Si el XML no supera el XSD se guarda como `result.INVALID.xml`.

### Qué no verifica el XSD

El XSD (1.0, el que soporta lxml) no puede comprobar: que el MathML corresponda al LaTeX; que el MathML sea válido (se incrusta sin validar); que los valores de `kind` de los nodos sean válidos; que el estado global sea coherente con el de las etapas; ni que la interpretación sea matemáticamente correcta. `xml_validator.py` añade comprobaciones de coherencia entre campos (`OVERALL_INCONSISTENT`, `STAGE_CONTENT_MISSING`); el resto queda fuera de la validación automática.

## Estándares del W3C y sus límites (estado consultado el 2026-10-30)

| Especificación | Estado | Uso en el proyecto |
|---|---|---|
| XML 1.0, Namespaces in XML, XML Schema 1.0 | Recommendations | Estructura del documento y validación. |
| MathML 3.0 (2.ª ed., 2014) | Recommendation | Raíz y espacio de nombres (`MML-001`). |
| MathML Core | Candidate Recommendation (snapshot 2025-06-24) | Conjunto de elementos (`MML-002`). |
| MathML 4.0 | Working Draft (27/10/2026) | Atributo `intent` (`MML-003`, solo informativo). |

El W3C **no** define qué significa una fórmula concreta, si una ecuación es correcta, su dominio ni sus unidades. Las reglas `SEM-*` son heurísticas del proyecto (o conclusiones del LLM) y están etiquetadas así en el catálogo. Los estados cambian: vuelve a comprobarlos antes de dar el catálogo por definitivo.

## Catálogo de reglas

`app/rules_catalog.json` (versionado). Cada regla tiene identificador, versión, descripción, condiciones de aplicación, resultado esperado, severidad, origen (`w3c_normative`, `project_heuristic`, `llm_derived`), fuente y ejemplos de prueba. Los ejemplos se ejecutan en `tests/test_intent_rules.py`, y el cargador rechaza catálogos inconsistentes con el código.

## Seguridad

- lxml se usa con un parser que no resuelve entidades, no accede a la red y no carga DTD; además se rechaza cualquier documento con `DOCTYPE` o `ENTITY` (XXE).
- El LaTeX y el MathML se envían al LLM como datos delimitados, con la instrucción de ignorar órdenes dentro de ellos. Reduce, pero no elimina, el riesgo de inyección de instrucciones desde una imagen manipulada.
- El agente descarta la confianza que declare el modelo: una autoevaluación del LLM no es una medida fiable.

## Limitaciones conocidas

- El OCR puede devolver LaTeX bien formado pero incorrecto; por eso existe la revisión manual y toda salida lleva el aviso `UNVERIFIED`.
- latex2mathml produce MathML de **presentación**: no garantiza la semántica (p. ej. `f(x)` como función o como producto).
- La clasificación de errores de sintaxis frente a conversión es heurística (por el módulo de la excepción).
- La comprobación de símbolos del LLM contra el MathML puede dar falsos positivos con símbolos implícitos (p. ej. la `d` de `dx`).
- No hay métrica automática de calidad semántica.
- Los formatos de petición a Ollama y a la API remota están escritos según la documentación recordada y no se han probado contra servidores reales.
- La lista de elementos de MathML Core debe volver a contrastarse con el CR vigente.

## Estado de verificación (2026-10-30)

Desarrollado en un entorno sin red ni pydantic. Lo ejecutado allí: las pruebas de XML, reglas y pipeline (con lxml real y un sustituto mínimo de pydantic), y la lógica de validación del MathML. **No** se han ejecutado con las dependencias reales: pytest completo, pix2tex, latex2mathml, pydantic real ni ningún LLM. Ejecuta `pytest -v` en tu entorno y trata cualquier fallo como información, no como regresión.
