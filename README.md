# image-to-mathml

Local equation recognition module that converts an equation image or LaTeX
expression into accessible Presentation MathML annotated with W3C `intent`
attributes.

The module is designed for a larger accessibility pipeline: a previous stage
provides the mathematics area, topic and surrounding text, and this module
returns MathML that can be spoken by a screen reader.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Optional OCR backends:

```bash
.venv/bin/python -m pip install -e '.[texify]'
.venv/bin/python -m pip install -e '.[pix2tex]'
.venv/bin/python -m pip install -e '.[rapid-latex-ocr]'
.venv/bin/python -m pip install -e '.[pix2text]'
.venv/bin/python -m pip install -e '.[paddleocr]'
```

Development tools:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

The default local LLM backend is Ollama at `http://localhost:11434` with
text model `qwen2.5:3b`. The vision flow uses `qwen3-vl:4b-instruct` by default
and can be changed with `--vision-model` or
`IMAGE_TO_MATHML_OLLAMA_VISION_MODEL`. Other OpenAI-compatible local servers
(vLLM, LM Studio, Jan, llama.cpp, LocalAI) can be selected with `--backend`.

## Pipeline flow

```text
image -> vision LLM -> annotated MathML -> normalisation -> validation -> repair
```

```text
image -> OCR -> LaTeX cleanup -> latex2mathml skeleton -> normalisation ->
clean skeleton -> temporary ids -> semantic JSON LLM -> deterministic
annotation -> validation -> repair/retry -> deterministic fallback
```

Supported flows:

- `offline`: deterministic `latex2mathml` conversion plus deterministic intent
  annotation. No LLM is used.
- `hybrid`: `latex2mathml` builds the Presentation MathML skeleton, the
  existing normaliser cleans it, and simple temporary ids are added. The LLM
  returns a JSON semantic contract (concept plus positional arguments); the
  module applies it deterministically, removes the temporary ids and validates
  the result. If validation fails, the JSON request is repeated with structured
  feedback. Invalid or missing annotations are rebuilt by the deterministic
  annotator.
- `direct`: the LLM converts LaTeX directly into annotated MathML. With
  `--direct-vision`, the Ollama vision agent reads the image directly instead
  and keeps the direct flow's deterministic fallback.
- `auto`: runs the hybrid flow first; runs the direct flow only if hybrid is
  not acceptable.
- `vision`: the Ollama vision model reads the image and produces annotated
  MathML directly, without OCR or an intermediate LaTeX string. There is no
  deterministic fallback in this flow: if validation fails after the repair
  attempts, the last MathML candidate produced by the model is returned.

The default flow is `hybrid`. The `vision` flow requires `--image` and ignores
`--ocr`.

### Concept retrieval for the hybrid flow

The semantic agent does not receive the complete core concept list. Before each
LLM call, the pipeline retrieves a small set of concept cards from the local
W3C registry. Each card carries the concept name, arity, argument roles,
aliases, LaTeX triggers, MathML tags, symbols, possible target element kinds
(``targets``, preferred first), required cues and one short example. Every core
concept has a target list; for example, an inner product may target the
operator ``mo`` in ``a⋅b`` or the enclosing ``mrow`` in ``⟨a,b⟩``. Retrieval
combines structural anchors found in the MathML tree (`mfrac`, `msup`, `msqrt`,
function names, operators), LaTeX triggers and lexical aliases, then applies the
current open-concept policy. The retrieved cards are injected into the prompt
and only those concept names are offered to the model.

A second retriever covers the W3C open concept list. Every open concept has a
card generated from the local registry, while
`data/open_concept_hints.json` maps textbook areas and topics such as
`análisis`, `probabilidad`, `álgebra lineal`, `teoría de números`,
`transformada de Fourier` or `mecánica cuántica` to candidate open concepts.
The open retriever also uses LaTeX, symbol and structural cues. Its cards are
appended to the core cards after applying the current open-concept policy, so
`--allow-open-concepts` or `--allowed-open-concepts` is still required before
an open candidate can reach the model. When the core RAG is disabled, the full
core list is sent and the open retriever is not used.

Use `--no-concept-rag` to send the full core list again, or
`--concept-rag-top-k` to change the number of core cards (default `8`). Use
`--no-open-concept-rag` to disable the open retriever and
`--open-concept-rag-top-k` to change its number of cards (default `5`). The
environment variables are `IMAGE_TO_MATHML_CONCEPT_RAG`,
`IMAGE_TO_MATHML_CONCEPT_RAG_TOP_K`, `IMAGE_TO_MATHML_OPEN_CONCEPT_RAG` and
`IMAGE_TO_MATHML_OPEN_CONCEPT_RAG_TOP_K`.

## OCR backends

The image-to-LaTeX stage is pluggable. `--ocr` (or `IMAGE_TO_MATHML_OCR`)
accepts:

| Backend | Extra | Engine | Notes |
|---|---|---|---|
| `texify` | `.[texify]` | PyTorch/Transformers | Default in `auto`. Adapter shims the config crash seen with `transformers>=4.45`. |
| `pix2text` | `.[pix2text]` | ONNX or PyTorch (MFR) | `LatexOCR` formula model, good accuracy, downloads on first use. |
| `pix2tex` | `.[pix2tex]` | PyTorch | Also known as LaTeX-OCR. |
| `paddleocr` | `.[paddleocr]` | PaddlePaddle | PP-FormulaNet via PaddleOCR 3.x; heavier install. |
| `rapid_latex_ocr` | `.[rapid-latex-ocr]` | ONNX | Lightweight, no PyTorch; its package pins `numpy<2`. |
| `ollama_vision` | - | Ollama + Qwen3-VL | Sends the image directly to Ollama `/api/chat`. Low-RAM option; needs no extra Python package. |
| `none` | - | - | OCR disabled; pass `--latex` instead. |

`auto` uses the first available backend in this order: `texify`, `pix2text`,
`pix2tex`, `paddleocr`, `rapid_latex_ocr`, `ollama_vision`. Aliases such as
`latexocr`, `rapidocr`, `p2t`, `ppformulanet` and `ollama` are accepted by
`--ocr`, the Python API and the environment variable. Because `ollama_vision` only needs the
base `requests` dependency, it is the automatic fallback when no specialized
equation OCR package is installed.

OCR-specific environment variables:

- `IMAGE_TO_MATHML_OCR_DEVICE`: `cpu` or `cuda` for texify and pix2text.
- `IMAGE_TO_MATHML_TEXIFY_MODEL`: Hugging Face checkpoint (default `vikp/texify`).
- `IMAGE_TO_MATHML_PIX2TEXT_MODEL`: MFR model name (default `mfr-1.5`).
- `IMAGE_TO_MATHML_PIX2TEXT_BACKEND`: `onnx` (default) or `pytorch`.
- `IMAGE_TO_MATHML_PADDLEOCR_MODEL`: default `PP-FormulaNet_plus-M`.
- `OLLAMA_STREAM`: enable Ollama streaming (`true`/`false`, default `false`).
- `IMAGE_TO_MATHML_OLLAMA_VISION_MODEL`: default `qwen3-vl:4b-instruct`; use
  `qwen3-vl:2b` on very low-RAM machines.
- `IMAGE_TO_MATHML_OLLAMA_VISION_BASE_URL`: default `OLLAMA_BASE_URL` or
  `http://localhost:11434`. Used by both the vision agent and the
  `ollama_vision` OCR backend.
- `IMAGE_TO_MATHML_OLLAMA_VISION_TIMEOUT`: default `180` seconds (`--timeout`
  sets it together with the text-model timeout).
- `IMAGE_TO_MATHML_OLLAMA_VISION_NUM_CTX`: context window, default `4096`.
- `IMAGE_TO_MATHML_OLLAMA_VISION_NUM_PREDICT`: maximum output tokens, default
  `1024`.
- `IMAGE_TO_MATHML_OLLAMA_VISION_MAX_SIDE`: resize larger images to this maximum
  side before sending, default `1600`; use `0` to disable.

`--vision-model`, `--base-url`, `--timeout` and `--stream` apply to the
`ollama_vision` OCR backend as well as to the vision agent. Each OCR backend is
loaded once per pipeline, and the LaTeX recognised for an image is reused while
the file does not change, so repeated runs (for example evaluation retries) do
not repeat the OCR.

LLM client environment variables:

- `OLLAMA_TEMPERATURE`: sampling temperature used by every agent (default `0`).
- `OLLAMA_THINK`: `false` disables the reasoning trace of thinking models such
  as `qwen3`; `true` enables it. Unset leaves the server default. `<think>`
  blocks are removed from answers in any case.
- `IMAGE_TO_MATHML_STRUCTURED_OUTPUT`: send the JSON Schema of the semantic
  contract to the server (`format` in Ollama, `response_format` in
  OpenAI-compatible servers), restricted to the retrieved candidate concepts.
  Default `true`. Servers that reject `response_format` are retried without it.
- `IMAGE_TO_MATHML_LLM_RETRIES`: extra attempts after a connection error or an
  HTTP 502/503/504 answer, default `1`. Read timeouts are not retried by the
  client.

## Image preprocessing

Every OCR backend can run an image-enhancement pass before recognition. The
default profile is intentionally minimal because aggressive transformations can
degrade already-clean formula images:

- `safe` (default): EXIF orientation correction and background polarity
  normalisation only.
- `standard`: adds border cropping, small-angle deskew, upscaling, autocontrast
  and unsharp-mask sharpening.
- `aggressive`: adds CLAHE and median denoising on top of `standard`.
- `off`: disables preprocessing completely.

Select a profile with `--preprocess-profile standard` or
`IMAGE_TO_MATHML_PREPROCESS_PROFILE=standard`. `--no-preprocess` (or
`IMAGE_TO_MATHML_PREPROCESS=false`) is equivalent to `off`.

Individual steps can be overridden through environment variables:

- `IMAGE_TO_MATHML_PREPROCESS_PROFILE`
- `IMAGE_TO_MATHML_PREPROCESS_POLARITY`
- `IMAGE_TO_MATHML_PREPROCESS_CROP`
- `IMAGE_TO_MATHML_PREPROCESS_PADDING`
- `IMAGE_TO_MATHML_PREPROCESS_DESKEW`
- `IMAGE_TO_MATHML_PREPROCESS_MAX_ANGLE`
- `IMAGE_TO_MATHML_PREPROCESS_CONTRAST`
- `IMAGE_TO_MATHML_PREPROCESS_CLAHE`
- `IMAGE_TO_MATHML_PREPROCESS_SHARPEN`
- `IMAGE_TO_MATHML_PREPROCESS_DENOISE`
- `IMAGE_TO_MATHML_PREPROCESS_UPSCALE`
- `IMAGE_TO_MATHML_PREPROCESS_MIN_SIDE`
- `IMAGE_TO_MATHML_PREPROCESS_BINARIZE`

Images with transparency are composited on a background that contrasts with
their ink before any other step, so black formulas exported on a transparent
background are not lost.

Set `IMAGE_TO_MATHML_PREPROCESS_DEBUG_DIR` (for example to
`results/preprocessed`) to save the exact image sent to the OCR engine as
`<image>_<hash>_preprocessed.png`; the hash of the full source path keeps
images with the same name in different folders apart. Nothing is written when
the variable is not set. Deskew needs NumPy and CLAHE needs OpenCV; when they
are unavailable the pipeline silently keeps the Pillow-only steps.

## CLI

```bash
image-to-mathml --version
image-to-mathml convert --latex '\frac{a}{b}' --flow hybrid --report
image-to-mathml compare --latex '\frac{a}{b}' --output-dir out
image-to-mathml evaluate --dataset examples/evaluation.jsonl --flow hybrid
image-to-mathml check --file equation.mathml
image-to-mathml doctor
```

`convert` prints the OCR LaTeX (when `--image` is used), the component that
generated the MathML structure (`llm` or `latex2mathml`), every output returned
by the LLM during each attempt (semantic JSON in the hybrid flow, MathML in the
direct and vision flows), and the final MathML. It also reports
per-phase timings with the tool that executed each phase, the time of every LLM
attempt, the attempts total/average, and the total pipeline time. If the direct
flow needs `latex2mathml` after the LLM attempts, that fallback is labelled in
the timing report. `--json` exposes the same information under `llm_outputs`,
`timings` and `timing_summary`.

### Required input

For `convert` and `compare`, provide one of:

- `--latex` / `-l`, including `-` for stdin.
- `--image` / `-i`.
- piped LaTeX on stdin when no `--image` is supplied.

`check` requires one of `--file`, `--mathml`, or stdin. `doctor` requires no
input.

### Common arguments

- `--latex`, `-l`: LaTeX expression.
- `--image`, `-i`: equation image path.
- `--area`: mathematics area from the book title.
- `--topic`: topic detected by the previous stage.
- `--context-text`: surrounding OCR text.
- `--lang`: reader language, default `es`.
- `--flow`: `direct`, `hybrid`, `auto`, `vision`, or `offline`; default
  `hybrid`.
- `--ocr`: `auto`, `texify`, `pix2tex`, `rapid_latex_ocr`, `pix2text`,
  `paddleocr`, `ollama_vision`, or `none`; aliases such as `ollama` and
  `qwen3-vl` are also accepted.
- `--no-preprocess`: disable contrast, deskew, crop and sharpening before OCR.
- `--preprocess-profile`: `safe` (default), `standard`, `aggressive`, or `off`.
- `--strict-core`: reject open concepts, including `gamma-function`.
- `--allow-open-concepts`: allow any documented W3C open concept.
- `--allowed-open-concepts`: comma-separated open concepts to allow, for example
  `gamma-function,airy-function`.
- `--max-repairs`: maximum repair attempts for LLM flows.
- `--no-concept-rag`: send the full core concept list instead of retrieved cards.
- `--concept-rag-top-k`: maximum number of retrieved core concept cards
  (default `8`).
- `--no-open-concept-rag`: disable area/topic retrieval for open concepts.
- `--open-concept-rag-top-k`: maximum number of retrieved open concept cards
  (default `5`).
- `--model`: text LLM model name.
- `--vision-model`: Ollama vision model for the vision agent (default
  `qwen3-vl:4b-instruct`).
- `--direct-vision`: in `--flow direct`, use the Ollama vision agent on the
  image instead of the text model on LaTeX. Requires `--image`; falls back to
  the deterministic LaTeX annotator when an OCR LaTeX representation is
  available, also when the vision server or model is not available.
- `--backend`: `ollama`, `openai`, `openai-compatible`, `vllm`, `vert`, or
  `lmstudio`.
- `--base-url`: LLM server base URL.
- `--api-key`: API key for an OpenAI-compatible server.
- `--timeout`: request timeout in seconds.
- `--stream`: use Ollama streaming responses for the text and vision clients.
  It is disabled by default; `--no-stream` forces it off when `OLLAMA_STREAM`
  is enabled in the environment.
- `--json`: print JSON instead of MathML.
- `--report`: print validation details to stderr.
- `--output`, `-o`: write the MathML result to a file.
- `--quiet`, `-q`: suppress warnings.

### Open concept policy

By default, open concepts are accepted only when they are in the documented
exception list: `gamma-function`, `beta-function` and `error-function`. Other
open concepts require `--allow-open-concepts` or an explicit
`--allowed-open-concepts` list. With `--strict-core`, even the documented
exceptions are rejected. The prompts of every agent state the same policy as
the validator, including a custom `--allowed-open-concepts` list.

### Validation contract

Generated MathML must satisfy (each point is a validation error):

- one `<math>` root in the MathML namespace;
- no `display` attribute on the root;
- exactly one top-level `<mrow>` wrapping the whole expression;
- no redundant single-child `<mrow>` elements;
- the right number of children for elements with positional children
  (`mfrac`, `msup`, `msub`, `mroot`, `munder`, `mover`: 2; `msubsup`,
  `munderover`: 3) and only `mtr`/`mtd` inside tables. An empty `<mrow/>` is
  accepted as the placeholder of an empty slot, as in `\frac{}{b}`;
- valid W3C intent grammar;
- every `$reference` must have exactly one matching `arg="reference"` in its
  scope;
- known concept names and arities;
- generic `f(x)` uses `intent="$name($var)"`.

In the hybrid flow the semantic JSON can never group the children of those
positional elements, so the language model cannot change the structure.

### Deterministic annotator

The fallback annotator only adds intents whose reading is unambiguous; when a
notation has several common readings it leaves the plain Presentation MathML,
which screen readers already read correctly:

- `f(x)`-style applications only for `f`, `g`, `h`, `F`, `G`, `H`, operator
  names (`\operatorname{sgn}`), indexed names (`f_i`) and primed names (`f'`).
  `a(b+c)` or `2x(x+1)` are products and stay unannotated.
- Superscripts that are not exponents (primes, `*`, `\dagger`, `T`, complement
  marks) are not annotated as powers; `^{\top}` becomes `transpose`.
- `\beta` is never guessed as the beta function and `\Gamma` only with
  parentheses.
- Operator names written as `<mo>` (`\det`, `\max`, `\min`) are recognised;
  `\lim_{x\to0}` is not an indexed symbol; `\log_b x` becomes
  `logarithm($arg,$base)`; `x_i^2` becomes
  `power(indexed-by($base,$sub),$exp)`.
- Sums, integrals and `|x|` are left to the reader or the LLM because their
  meaning depends on context (`|A|` can be a determinant).

## BLEU evaluation

The module can compare pipeline output with labelled MathML references. Two
scores are computed for each equation and for the whole corpus:

- `structure_bleu`: canonical MathML with `intent` and `arg` removed. It
  measures how close the generated structure and symbol sequence are.
- `intent_bleu`: canonical MathML that keeps normalised `intent` and `arg`
  attributes. It measures structure and semantic annotation together.

Both documents are normalised before scoring:

- intent arguments are renamed positionally (`$base,$exp` becomes
  `$arg1,$arg2`), so equivalent role names do not reduce the score;
- `mspace`, `mphantom`, `mglyph` and alignment helpers are removed,
  `semantics`/`annotation` wrappers are ignored, and `mstyle`/`mpadded`
  wrappers are transparent;
- presentation attributes such as `stretchy`, `mathvariant`, `mathsize`,
  `mathcolor`, `displaystyle` and `columnalign` are ignored;
- attribute-less single-child `<mrow>` wrappers are unwrapped;
- text and attribute whitespace is collapsed and attribute order is canonical.

A `reference_latex` reference is converted and normalised with the same code
as the pipeline output, so an identical offline result scores 1.0. Such a
reference has no intents, so its intent scores do not measure the semantic
annotation; the report adds a note when it is used.

Besides BLEU, every sample reports exact matches (structure only and with
intents) and an intent F1 that compares the canonical intents element by
element; the report adds the validation rate, the exact-match rates and the
mean intent F1.

The `evaluate` command reads a JSON array, JSONL or CSV dataset. Each record
needs an input (`latex` or `image`) and a reference (`reference_mathml`,
`reference`, `mathml`, or `reference_latex`). The MathML can also live in a
separate file referenced with `reference_mathml_file`, `reference_file`,
`mathml_file` or `reference_path`; relative paths are resolved from the dataset
directory. Optional fields are `id`, `area`, `topic`, `context_text`, `lang`,
`flow`, `ocr`, `allow_open_concepts`, `max_repair_attempts`, `timeout` and
`eqn_complexity`.

```json
{"id":"power","latex":"x^{2}","reference_mathml":"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><msup intent=\"power($base,$exp)\"><mi arg=\"base\">x</mi><mi arg=\"exp\">2</mi></msup></mrow></math>","area":"álgebra","topic":"potencias"}
{"id":"power-file","latex":"x^{2}","reference_mathml_file":"refs/power.mathml","area":"álgebra","topic":"potencias"}
{"id":"fraction","latex":"\\frac{a}{b}","reference_latex":"\\frac{a}{b}","flow":"offline"}
```

For image datasets, CSV is supported with the columns
`id,img_path,mathml_path,area,topic,timeout,eqn_complexity`:

```csv
id,img_path,mathml_path,area,topic,timeout,eqn_complexity
alg001,alg001.jpeg,alg001.mathml,Álgebra lineal,ortogonalización,300,muy simple
alg002,alg002.jpeg,alg002.mathml,Álgebra lineal,bases ortogonales,600,compleja
```

`img_path` and `mathml_path` are resolved relative to the CSV file. `timeout` is
the initial timeout in seconds of **each request** (OCR, LLM and vision calls)
for that sample, not a budget for the whole sample: with OCR and several repair
attempts a sample can take several times that value. If Ollama or the semantic
annotator reports a timeout, the sample is retried with a larger timeout
(`initial + --timeout-increment`) up to `--max-timeout-retries` times; the OCR
result of the image is reused between retries. In CSV mode the command defaults
to the hybrid flow, `--ocr ollama_vision`, `--model qwen3:4b-instruct` and
`--vision-model qwen3-vl:4b-instruct`; each default applies only when neither
the flag nor its environment variable (`IMAGE_TO_MATHML_OCR`, `OLLAMA_MODEL`,
`IMAGE_TO_MATHML_OLLAMA_VISION_MODEL`) is set. The report header shows the
model, vision model and OCR backend that were used.

To avoid repeating absolute paths in the CSV, use a YAML file:

```yaml
csv: dataset.csv
images_dir: ${IMAGE_TO_MATHML_DATASET_DIR:-../dataset}/images
mathml_dir: ${IMAGE_TO_MATHML_DATASET_DIR:-../dataset}/labels
complexity_weights:
  muy simple: 1.0
  simple: 1.0
  media: 1.5
  compleja: 2.0
```

`csv`, `images_dir` and `mathml_dir` can be absolute or relative to the YAML
file, and may use `$VAR`, `${VAR}` or `${VAR:-default}` so the same YAML works
on several machines. `--images-dir` and `--mathml-dir` override the YAML
folders. In this example the CSV only contains file names such as
`alg001.jpeg` and `alg001.mathml`. `eval/linear-algebra.yaml` reads the
dataset folder from `IMAGE_TO_MATHML_DATASET_DIR`.

The report separates where each output came from:

- every sample with a valid reference counts in the corpus, per-class and
  weighted scores; a sample whose run failed counts as an empty output;
- an LLM flow that ended in the deterministic fallback (LLM unavailable,
  invalid answers or exhausted timeouts) is marked `respaldo` in the table and
  counted apart; the "solo LLM" corpus scores include only outputs the LLM
  really produced;
- samples that ended in a timeout show the number of retries.

The weighted BLEU is a per-equation average: each class contributes its
corpus BLEU times its weight times its number of equations. Without
`complexity_weights` every equation weighs the same. A classification that has
no explicit weight uses `1.0` and is listed in the report notes.

```bash
image-to-mathml evaluate --dataset examples/evaluation.jsonl --flow hybrid
image-to-mathml evaluate --dataset eval/data_eval.yaml \
  --timeout-increment 60 --max-timeout-retries 3 \
  --json --output results/bleu.yaml.json
image-to-mathml evaluate --dataset eval/dataset.csv --flow hybrid \
  --ocr ollama_vision --model qwen3:4b-instruct \
  --vision-model qwen3-vl:4b-instruct \
  --timeout-increment 60 --max-timeout-retries 3 \
  --json --output results/bleu.csv.json
image-to-mathml evaluate --dataset examples/evaluation.jsonl --flow offline \
  --n-gram 4 --json --output results/bleu.json
```

The command prints the corpus scores and a per-sample table. JSON output
contains the same corpus and sample data. The Python API is available through
`image_to_mathml.metrics.mathml_bleu` / `mathml_corpus_bleu` and
`image_to_mathml.evaluation.evaluate_samples`.

## Examples

Image -> LaTeX with the Ollama vision OCR backend:

```bash
ollama pull qwen3-vl:4b-instruct
image-to-mathml convert --image equation.png --ocr ollama_vision --flow hybrid
```

Direct image -> annotated MathML with the vision flow (no intermediate LaTeX):

```bash
image-to-mathml convert --image equation.png --flow vision
```

Direct flow using the vision agent with deterministic fallback:

```bash
image-to-mathml convert --image equation.png --flow direct --direct-vision
```

For very low RAM, pull `qwen3-vl:2b` and use
`--vision-model qwen3-vl:2b` or
`IMAGE_TO_MATHML_OLLAMA_VISION_MODEL=qwen3-vl:2b`. The same vision agent can be
selected in the direct flow with `IMAGE_TO_MATHML_DIRECT_VISION=1`.

Offline deterministic conversion:

```bash
image-to-mathml convert --latex 'f(x)=\frac{1}{1+e^{-x}}' --flow offline
```

Compare all flows:

```bash
image-to-mathml compare --latex '\frac{a}{b}' --output-dir results
```

Validate a document:

```bash
image-to-mathml check --file results/hybrid.mathml --json
```

Check local dependencies:

```bash
image-to-mathml doctor
```

## Development

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Run lint and formatting checks:

```bash
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the lint,
format and test steps on Python 3.10 and 3.12. It also runs `mypy` as an
informative job that does not fail the build yet, because the existing
annotations still report errors.

`examples/img_test.json` references `images/alg002.jpeg` relative to the
`examples` folder; the image belongs to the external dataset and is not
versioned, so copy it there (or edit the path) before running that example.

The concept registries are generated by `tools/build_concept_registry.py`.
Signatures that the W3C tables do not state, such as the two-argument
`beta-function`, are added by `EXTRA_SIGNATURES` in that script so they survive
regeneration.
