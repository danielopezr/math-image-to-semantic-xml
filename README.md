# math-semantic-ocr

Convierte la imagen de una expresión matemática en LaTeX (pix2tex), MathML de presentación (latex2mathml), una **interpretación semántica estructurada** (LLM con salida estructurada y vocabulario de intents controlado), **braille matemático y lectura en voz** (MathCAT de DAISY), un documento XML validado y una evaluación con un catálogo de reglas.

Cada etapa se ejecuta y se prueba por separado. Si una etapa falla, sus dependientes se omiten, los resultados anteriores se conservan y el resultado final **no** se declara validado.

## Arquitectura

```
imagen ─► ocr ─► converter ─┬─► semantic_agent ─► (reglas) ─► enrichment (opcional) ─┐
          pix2tex  latex2mathml│                                                       ├─► xml_builder ─► xml_validator
                               ├─► braille  (MathCAT, usa el MathML ORIGINAL)         │
                               └─► speech   (MathCAT) ─► audio (proveedor TTS, opcional)┘
```

Las dependencias reales no son una cadena lineal: el braille y la lectura dependen del MathML, no de la interpretación semántica; el audio depende de la lectura.

| Módulo | Responsabilidad |
|---|---|
| `app/models.py` | Contratos de datos (Pydantic): `SemanticNode`, `SemanticOutput`, `BrailleOutput`, `SpeechOutput`, `AudioOutput`, `MathCATConfiguration`, `StageResult`, `PipelineResult`… |
| `app/ocr.py` | `recognize_formula`, revisión manual y repetición del OCR. |
| `app/converter.py` | `latex_to_mathml`; distingue error de sintaxis LaTeX de error de conversión. |
| `app/intent_vocabulary.json`, `app/intents.py` | Vocabulario controlado y versionado de intents (fuente única). |
| `app/semantic_tree.py` | Validación del árbol semántico, orden de evaluación, métricas. |
| `app/semantic_agent.py` | Interfaz `SemanticProvider` (Ollama local / API remota) y validación de la salida del LLM. |
| `app/intent_rules.py`, `app/rules_catalog.json` | `evaluate_intents` y catálogo versionado de reglas. |
| `app/enrichment.py`, `app/intent_mathml_mapping.json` | Enriquecimiento **opcional y experimental** con atributos `intent`. |
| `app/mathcat.py` | Interfaz independiente con MathCAT. |
| `app/braille.py`, `app/audio.py` | Braille Unicode; texto de lectura, síntesis TTS y reproducción. |
| `app/xml_builder.py`, `app/xml_validator.py`, `app/schemas/result.xsd` | Construcción y validación del XML. |
| `app/pipeline.py` | `run_pipeline(...)`: único punto de entrada para la CLI y una futura web. |
| `app/evaluation.py` | Resumen por aspecto (sin puntuación global). |

## Símbolo, intent y relación

Se distinguen tres conceptos: **símbolo** (lo escrito), **intent** (la función matemática del elemento dentro de la expresión) y **relación** (conexión padre-hijo entre nodos). Para `x² + 3x = 10`:

| Nodo | Símbolo | Intent | Operandos |
|---|---|---|---|
| expresión completa | — | `equation` | statement → `=` |
| `=` | `=` | `equality` | left → `+`, right → `10` |
| `+` | `+` | `addition` | term → `x²`, term → `3x` |
| `x²` | — | `power` | base → `x`, exponent → `2` |
| `3x` | — (implícito) | `multiplication` (`implicit`) | factor → `3`, factor → `x` |
| `x`, `2`, `3`, `10` | idem | `variable`, `number` | — |

`examples/ejemplo_ecuacion_construido_a_mano.xml` contiene este árbol serializado. **Está construido a mano para ilustrar el formato; no es la salida de un LLM.**

### Vocabulario de intents

40 intents en `app/intent_vocabulary.json` (versión 1.0.0), agrupados en elementos, operaciones, relaciones, estructuras, funciones y cálculo. **Es un vocabulario interno del proyecto: sus nombres no son valores oficiales de MathML ni conceptos del atributo `intent` del W3C.**

### Árbol semántico y ambigüedad

El árbol es una lista plana de nodos con `id`, `node_type`, `intent`, `symbol`, `operands` (hijos ordenados con rol), `properties`, `implicit`, `evaluation_order`, `evidence`, `ambiguous`, `candidate_intents` y `review_required`. La forma plana permite detectar ciclos, padres múltiples y nodos huérfanos.

Un nodo **ambiguo** (`xy`, `f(x)`, `|x|`…) no tiene intent definitivo: lista 2 o más `candidate_intents`, una justificación y `review_required`, y esa marca se propaga al documento.

Validaciones (`validate_semantic_output`): ids únicos, raíz y operandos existentes, ciclos, padres múltiples, nodos huérfanos, tipo de nodo e intent permitidos y coherentes, roles y aridad, propiedades, ambigüedad bien formada, coherencia de las marcas de revisión, enlaces y **correspondencia de los símbolos con la fórmula**. El orden de evaluación lo calcula el proyecto (postorden); el que envíe el LLM se descarta. La confianza declarada por el LLM se descarta. El esquema no tiene campos para LaTeX ni MathML, así que el LLM no puede devolverlos modificados; se guardan huellas SHA-256 de lo que se le envió.

La validación comprueba forma y coherencia con el vocabulario; **no demuestra que la interpretación sea matemáticamente correcta**.

## MathML, atributos `intent` y enriquecimiento

Se distinguen: (1) MathML de presentación, lo que produce latex2mathml; (2) MathML con atributos `intent` (MathML 4, **Working Draft**, no es una Recommendation); (3) el árbol semántico interno del proyecto, que no es MathML.

El enriquecimiento (`--enrich-intents`) está **desactivado por defecto y es experimental**. Solo añade atributos `intent` y `arg`, nunca cambia la estructura (se comprueba), no sobrescribe un `intent` existente, solo usa nodos definitivos y alineados sin ambigüedad, y solo se acepta si MathCAT lo procesa en modo estricto (`IntentErrorRecovery=Error`). Los nombres de concepto de `intent_mathml_mapping.json` proceden de las listas de trabajo del Math WG, siguen en evolución y **no están verificados contra una especificación cerrada**. El braille siempre usa el MathML original.

## Instalación en Windows (PowerShell)

Requiere Python 3.10 o 3.11.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # si se bloquea: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu    # comprueba el comando en pytorch.org
pip install -r requirements.txt
copy .env.example .env
python -m app.export_schema          # genera app/schemas/semantic_output.schema.json
pip freeze > requirements.lock.txt
```

Para probar sin OCR ni PyTorch: `pip install pydantic Pillow lxml latex2mathml pytest python-dotenv` y usa `--latex`.

## MathCAT (braille y lectura matemática)

MathCAT es una librería de DAISY escrita en Rust. **No se ha verificado que exista un paquete instalable con `pip`.** Su interfaz Python (MathCATForPython) se construye con pyo3 y se distribuye, por ejemplo, dentro del complemento de NVDA. Qué necesitas:

1. **La carpeta `Rules` de MathCAT** (repositorio oficial `daisy/MathCAT`). Define su ruta en `MATHCAT_RULES_DIR`.
2. **El módulo nativo de Python.** `app/mathcat.py` intenta importar `libmathcat_py` y luego `libmathcat` (nombres observados en distintas versiones; solo se admite esa lista cerrada). Obtenerlo para tu versión de Python (compilarlo desde MathCATForPython o extraerlo de una versión compatible) **queda por verificar en tu entorno**; no se incluye ninguna instrucción no comprobada.
3. Comprueba la instalación: `python -c "import libmathcat_py"` y luego las pruebas reales (más abajo).

Llamadas usadas (documentación oficial): `SetRulesDir` (la primera), `SetPreference`/`GetPreference` (`Language`, `BrailleCode`, `SpeechStyle`, `Verbosity`, `TTS`, `IntentErrorRecovery`; distinguen mayúsculas), `SetMathML`, `GetSpokenText`, `GetBraille("")`. El nombre `GetVersion` y la falta de una función de liberación no están verificados: `close()` solo suelta la referencia. MathCAT guarda su estado a nivel de proceso, así que las llamadas se serializan con un cerrojo.

### Códigos braille

| Opción del proyecto | Valor de `BrailleCode` en MathCAT |
|---|---|
| `nemeth` | `Nemeth` |
| `ueb_technical` | `UEB` (UEB técnico) |
| `cmu` | `CMU` (Código Matemático Unificado, español) |

El valor por defecto (`cmu`, pensado para español) vive en un único sitio, `MathCATConfiguration`, y se cambia con `MATHCAT_BRAILLE_CODE`, `--braille-code` o el parámetro de `run_pipeline`. Que una versión concreta de MathCAT soporte cada código hay que comprobarlo con las pruebas reales; si el motor rechaza el valor, la etapa falla con `PREFERENCE_REJECTED`.

La salida es **texto braille Unicode (U+2800–U+28FF)**. No es un archivo BRF, ni formato de embosado, ni salida para una línea braille. `--export-braille` guarda texto UTF-8 (`.txt`) y **rechaza `.brf`**, porque ese formato no está implementado. Un XML válido **no demuestra** que la transcripción sea correcta: debe revisarla una persona con conocimiento del código (`--review-braille`).

### Lectura matemática y audio

Son tres cosas distintas, con estados independientes:

1. **Texto de lectura**: lo genera MathCAT (`--speech`; idioma por defecto `es`; estilos `ClearSpeak`/`SimpleSpeak`; verbosidad `Terse`/`Medium`/`Verbose`). La documentación de MathCAT indica soporte de español; la calidad de la lectura hay que comprobarla.
2. **Síntesis de voz**: MathCAT **no genera audio**. Lo hace un proveedor TTS (`TTSProvider`). Incluido: `Pyttsx3Provider` (`--tts pyttsx3`), local, que en Windows usa SAPI5 y necesita una **voz en español instalada**; si no la hay, falla con `NO_VOICE_FOR_LANGUAGE` en lugar de leer en otro idioma. Produce WAV. Se puede añadir otro proveedor implementando la interfaz.
3. **Reproducción**: `--play` (solo Windows, WAV, `winsound`).

Sin proveedor TTS el audio queda `SKIPPED` y **no se afirma ningún archivo**; eso no invalida un resultado cuyo braille sí se generó. Si hay proveedor y falla, el estado es `FAILED` y el resultado no queda validado. El audio solo cuenta como generado si el archivo existe y tiene cabecera WAV válida. El texto se pasa al proveedor como dato; nunca se construyen comandos externos con él.

## Uso

```powershell
python main.py ruta\formula.png                                    # OCR + MathML + reglas, sin LLM
python main.py ruta\formula.png --review                           # revisar/corregir/repetir el OCR
python main.py --latex "x^2+3x=10" --provider ollama               # sin imagen, LLM local
python main.py --latex "\frac{a}{b}" --braille --braille-code cmu --export-braille salida.txt
python main.py --latex "x=0" --speech --language es                # solo texto de lectura
python main.py --latex "x=0" --audio --tts pyttsx3 --export-audio salida.wav --play
python main.py --mathml-file f.mathml --braille --speech           # solo las etapas de MathCAT
python main.py ruta\formula.png --expected-latex "x=0" --json      # resumen por aspecto
python main.py                                                     # modo interactivo
```

Con `--mathml-file` se omiten OCR y conversión; como son etapas obligatorias, ese resultado **nunca queda validado**. Código de salida: `0` validado, `1` no validado, `2` error de uso, `3` fallo al exportar.

### Agente semántico (opcional)

- **Local (sin coste por llamada):** instala [Ollama](https://ollama.com), descarga un modelo y define `OLLAMA_MODEL`. La calidad depende del modelo y del hardware.
- **Remoto:** `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` (API de chat tipo OpenAI). **Puede tener coste por token**; cada fórmula envía el esquema, el vocabulario, el LaTeX y el MathML, y un reintento duplica el consumo. La clave nunca debe estar en el código ni en el control de versiones.

## Estados de validación

Se informan por separado; no existe una puntuación global.

| Faceta | Qué significa |
|---|---|
| XML bien formado / válido según XSD | `XmlResult.well_formed` / `schema_valid` |
| MathML estructuralmente correcto | etapa `converter` + `validate_mathml` |
| Interpretación semántica | etapa `semantic_agent`; `semanticReviewRequired` indica si necesita revisión humana |
| Braille generado | etapa `braille`; `brailleGenerated` |
| Lectura generada / audio generado | etapas `speech` y `audio`; `speechGenerated`, `audioGenerated` |
| Resultado global | `VALIDATED` solo si ninguna etapa falló y todas las **obligatorias** están `OK` |

Obligatorias: `ocr`, `converter`, `semantic_agent`, `rules`, y `braille`/`speech` si se solicitaron. Opcionales: `enrichment` y `audio` (omitirlos no impide `VALIDATED`, pero un fallo en cualquiera sí).

## Documento XML

Espacio de nombres del proyecto `urn:math-semantic-ocr:result:1` (no es un estándar del W3C); el MathML va en `http://www.w3.org/1998/Math/MathML`. Solo se guardan **nombres** de archivo (imagen, audio), nunca rutas personales.

**Qué no verifica el XSD** (1.0, lo que soporta lxml): ciclos en el árbol, que los intents pertenezcan al vocabulario, la coherencia entre estados y facetas, que el MathML sea válido (se incrusta sin validar), que el MathML corresponda al LaTeX, ni que el braille o la interpretación sean correctos. `xml_validator.py` añade comprobaciones de ciclos, vocabulario, estados y facetas (`CYCLE_IN_XML`, `UNKNOWN_INTENT_IN_XML`, `OVERALL_INCONSISTENT`, `FACET_INCONSISTENT`…); lo demás queda fuera de la validación automática. El XSD sí garantiza ids de nodo únicos y que operandos, enlaces y raíz apunten a ids existentes.

Seguridad XML: parser sin resolución de entidades, sin acceso a red y sin DTD; se rechaza cualquier documento con `DOCTYPE` o `ENTITY`.

## Pruebas

```powershell
pytest -v
pytest tests\test_semantic_tree.py tests\test_semantics.py tests\test_intent_rules.py
pytest tests\test_mathcat.py tests\test_braille.py tests\test_audio.py tests\test_enrichment.py
pytest tests\test_xml.py tests\test_pipeline.py
```

Las pruebas de MathCAT usan un **motor falso** que solo registra llamadas: no reproduce resultados reales. Las pruebas de integración reales (`test_integracion_real_*` y `test_real_*`) se omiten si falta la dependencia. Para ejecutar las de MathCAT: `$env:MATHCAT_RULES_DIR="C:\ruta\a\Rules"; pytest tests\test_mathcat.py tests\test_braille.py -v`. Esas pruebas solo comprueban que la salida no está vacía y contiene celdas braille; no comparan contra una transcripción esperada.

## Estado de verificación (2026-11-01)

Desarrollado en un entorno sin red, sin pydantic, sin pytest, sin MathCAT, sin pix2tex, sin latex2mathml y sin pyttsx3.

- **Ejecutado:** 164 pruebas pasaron (OCR 9, conversión 11, árbol semántico 23, agente 13, reglas 8, MathCAT 15, braille 13, audio 16, enriquecimiento 8, XML 19, pipeline 29), con lxml y el XSD reales y **sustitutos mínimos de pydantic y pytest** que solo existen en el entorno de desarrollo. 9 pruebas se omitieron por falta de dependencias reales.
- **Verificado en la documentación oficial de MathCAT:** orden de llamadas, nombres y valores de las preferencias, `GetBraille("")`, idioma `es` y códigos Nemeth/UEB/CMU.
- **Sin verificar:** instalación del módulo nativo de MathCAT; salida braille y de lectura reales de cualquier expresión; que CMU/UEB/Nemeth y el español funcionen en tu versión; `GetVersion`; pix2tex y latex2mathml reales; pyttsx3 con una voz real; los formatos de petición a Ollama y a la API remota; la calidad de la interpretación de un LLM real; los nombres de concepto de `intent`.
- **Pendiente de tu lado:** ejecutar `pytest -v` con las dependencias reales y tratar cualquier fallo como información.

## Limitaciones conocidas

- El OCR puede devolver LaTeX bien formado pero incorrecto; por eso toda salida lleva el aviso `UNVERIFIED`.
- latex2mathml produce MathML de **presentación**: no garantiza la semántica (p. ej. `f(x)` como función o producto).
- La clasificación de errores de sintaxis frente a conversión es heurística.
- La comprobación de símbolos contra la fórmula puede dar falsos positivos con símbolos implícitos (la `d` de `dx`).
- Las relaciones encadenadas (`a<b<c`) no tienen representación propia: deben marcarse para revisión.
- No hay métrica automática de calidad semántica ni de corrección del braille.
- La selección de voz de pyttsx3 es heurística (idioma en el nombre o id de la voz).
- No se genera BRF ni salida para línea braille.
- La lista de elementos de MathML Core del catálogo debe contrastarse con el CR vigente.
