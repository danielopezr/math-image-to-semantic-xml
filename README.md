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
image -> OCR -> LaTeX cleanup -> MathML generation/annotation ->
normalisation -> validation -> repair/retry -> deterministic fallback
```

Supported flows:

- `offline`: deterministic `latex2mathml` conversion plus deterministic intent
  annotation. No LLM is used.
- `hybrid`: `latex2mathml` builds the Presentation MathML skeleton, then the
  LLM adds `intent` and `arg` attributes. Invalid LLM annotations are pruned
  and rebuilt deterministically.
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
- `IMAGE_TO_MATHML_OLLAMA_VISION_MODEL`: default `qwen3-vl:4b-instruct`; use
  `qwen3-vl:2b` on very low-RAM machines.
- `IMAGE_TO_MATHML_OLLAMA_VISION_BASE_URL`: default `OLLAMA_BASE_URL` or
  `http://localhost:11434`.
- `IMAGE_TO_MATHML_OLLAMA_VISION_TIMEOUT`: default `180` seconds.
- `IMAGE_TO_MATHML_OLLAMA_VISION_NUM_CTX`: context window, default `4096`.
- `IMAGE_TO_MATHML_OLLAMA_VISION_NUM_PREDICT`: maximum output tokens, default
  `1024`.
- `IMAGE_TO_MATHML_OLLAMA_VISION_MAX_SIDE`: resize larger images to this maximum
  side before sending, default `1600`; use `0` to disable.

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

Set `IMAGE_TO_MATHML_PREPROCESS_DEBUG_DIR` to override the output directory.
By default, the exact image sent to the OCR engine is written to
`results/preprocessed/<image>_preprocessed.png`. Deskew needs NumPy and CLAHE
needs OpenCV; when they are unavailable the pipeline silently keeps the
Pillow-only steps.

## CLI

```bash
image-to-mathml --version
image-to-mathml convert --latex '\frac{a}{b}' --flow hybrid --report
image-to-mathml compare --latex '\frac{a}{b}' --output-dir out
image-to-mathml check --file equation.mathml
image-to-mathml doctor
```

`convert` prints the OCR LaTeX (when `--image` is used), the component that
generated the MathML structure (`llm` or `latex2mathml`), every MathML structure
returned by the LLM during each attempt, and the final MathML. `--json` exposes
the same structures under `llm_outputs`.

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
- `--model`: text LLM model name.
- `--vision-model`: Ollama vision model for the vision agent (default
  `qwen3-vl:4b-instruct`).
- `--direct-vision`: in `--flow direct`, use the Ollama vision agent on the
  image instead of the text model on LaTeX. Requires `--image`; falls back to
  the deterministic LaTeX annotator when an OCR LaTeX representation is
  available.
- `--backend`: `ollama`, `openai`, `openai-compatible`, `vllm`, `vert`, or
  `lmstudio`.
- `--base-url`: LLM server base URL.
- `--api-key`: API key for an OpenAI-compatible server.
- `--timeout`: request timeout in seconds.
- `--json`: print JSON instead of MathML.
- `--report`: print validation details to stderr.
- `--output`, `-o`: write the MathML result to a file.
- `--quiet`, `-q`: suppress warnings.

### Open concept policy

By default, open concepts are accepted only when they are in the documented
exception list: `gamma-function`, `beta-function` and `error-function`. Other
open concepts require `--allow-open-concepts` or an explicit
`--allowed-open-concepts` list. With `--strict-core`, even the documented
exceptions are rejected.

### Validation contract

Generated MathML must satisfy:

- one `<math>` root in the MathML namespace;
- no `display` attribute on the root;
- exactly one top-level `<mrow>` wrapping the whole expression;
- no redundant single-child `<mrow>` elements;
- valid W3C intent grammar;
- every `$reference` must have a matching `arg="reference"`;
- known concept names and arities;
- generic `f(x)` uses `intent="$name($var)"`.

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
