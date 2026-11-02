"""CLI de math-semantic-ocr. Usa exclusivamente ``app.pipeline.run_pipeline``.

Ejemplos (PowerShell):
  python main.py ruta\\formula.png                                  # OCR + MathML + reglas (sin LLM)
  python main.py ruta\\formula.png --review                         # revisar/corregir/repetir el OCR
  python main.py --latex "x^2+3x=10" --provider ollama              # sin imagen, con LLM local
  python main.py --latex "\\frac{a}{b}" --braille --braille-code CMU # braille (requiere MathCAT)
  python main.py --latex "x=0" --braille --audio --language es --tts pyttsx3 --export-audio salida.wav
  python main.py --mathml-file f.mathml --braille --speech          # solo las etapas de MathCAT sobre un MathML
  python main.py                                                    # modo interactivo
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from app.audio import Pyttsx3Provider, play_audio_file
from app.braille import export_braille_text
from app.evaluation import summarize_stages
from app.mathcat import BRAILLE_CODES
from app.models import PipelineResult, StageStatus
from app.pipeline import RETRY, run_pipeline
from app.semantic_agent import OllamaProvider, OpenAICompatibleProvider, ProviderError
from app.semantic_tree import format_tree


def _header(title: str) -> None:
    print(f"\n=== {title} ===")


def show_report(report: PipelineResult, expected_latex: Optional[str], as_json: bool) -> None:
    if as_json:
        print(json.dumps(summarize_stages(report, expected_latex), indent=2, ensure_ascii=False))
        return

    _header("LaTeX reconocido")
    if report.ocr and report.ocr.latex is not None:
        print(report.ocr.latex)
        if report.ocr.reviewed_by_human and report.ocr.latex_raw is not None:
            print(f"(original del OCR: {report.ocr.latex_raw})")
    else:
        print("(sin LaTeX)")

    _header("MathML (presentacion)")
    print(report.conversion.mathml if report.conversion and report.conversion.mathml else "(no generado)")

    _header("Interpretacion semantica")
    if report.semantic and report.semantic.output:
        out = report.semantic.output
        print(f"vocabulario {out.vocabulary_version}   revision requerida: {out.review_required}")
        print("\n".join(format_tree(out)))
        for link in out.links:
            print(f"Enlace: {link.type} {link.source} -> {link.target}")
        for text in out.assumptions:
            print(f"Supuesto: {text}")
        for text in out.ambiguities:
            print(f"Ambiguedad: {text}")
    else:
        print("(no disponible: " + report.skipped.get("semantic_agent", "la etapa fallo") + ")")

    _header("Reglas")
    if report.rules:
        for r in report.rules:
            print(f"{r.rule_id:8} {r.outcome.value:15} [{r.origin.value}] {r.message}")
    else:
        print("(sin evaluar: " + report.skipped.get("rules", "no disponible") + ")")

    if report.braille is not None:
        _header("Braille matematico (texto Unicode; NO es BRF)")
        b = report.braille
        print(b.text if b.text else "(no generado)")
        print(f"codigo: {b.braille_code}   estado: {b.status.value}   revision requerida: {b.review_required}")
    if report.speech is not None:
        _header("Lectura matematica (texto)")
        print(report.speech.text if report.speech.text else "(no generada)")
        print(f"idioma: {report.speech.language}   estado: {report.speech.status.value}")
    if report.audio is not None:
        _header("Audio")
        a = report.audio
        if a.status is StageStatus.OK:
            print(f"Archivo generado: {a.file_name} ({a.size_bytes} bytes, {a.audio_format}, proveedor {a.provider})")
        else:
            print(f"Sin archivo de audio (estado {a.status.value}).")

    _header("Errores y advertencias")
    issues = []
    for result in (report.ocr, report.conversion, report.semantic, report.enrichment, report.braille,
                   report.speech, report.audio, report.xml):
        if result is not None:
            issues += [("ERROR", i) for i in result.errors] + [("AVISO", i) for i in result.warnings]
    for level, i in issues:
        print(f"{level:6} {i.stage}/{i.code}: {i.message}")
    if not issues:
        print("(ninguno)")

    _header("Estado por etapa")
    for s in report.stages:
        flags = ("obligatoria" if s.required else "opcional") + (", REVISION REQUERIDA" if s.review_required else "")
        reason = f"  [{s.skipped_reason}]" if s.skipped_reason else ""
        print(f"{s.stage:15} {s.status.value:8} {flags}{reason}")
    print(f"\nResultado final validado: {'SI' if report.validated else 'NO'}")
    print("(Un XML valido no demuestra que el braille ni la interpretacion sean correctos.)")
    if report.output_dir:
        print(f"Resultados intermedios en: {report.output_dir}")


def interactive_reviewer(ocr):
    print("\n--- Revision del LaTeX ---")
    for e in ocr.errors:
        print(f"ERROR {e.code}: {e.message}")
    print(f"LaTeX actual (intento {ocr.attempt}): {ocr.latex if ocr.latex is not None else '(ninguno)'}")
    answer = input("Enter = aceptar | escribe el LaTeX corregido | 'r' = repetir OCR: ").strip()
    if answer == "":
        return None
    return RETRY if answer.lower() == "r" else answer


def braille_reviewer(out):
    print("\n--- Revision del braille ---")
    print(f"Braille generado ({out.braille_code}): {out.text}")
    answer = input("Enter = aceptar | escribe el braille corregido: ").strip()
    return answer or None


def build_provider(name: str):
    if name == "none":
        return None
    if name == "ollama":
        return OllamaProvider.from_env()
    if name == "remote":
        return OpenAICompatibleProvider.from_env()
    raise ProviderError(f"Proveedor desconocido: {name}")


def main(argv: Optional[list[str]] = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Imagen de formula -> LaTeX -> MathML -> interpretacion -> braille/audio -> XML.")
    ap.add_argument("image", nargs="?", help="Ruta de la imagen con la formula.")
    ap.add_argument("--latex", help="LaTeX manual (omite el OCR si no se da imagen).")
    ap.add_argument("--mathml-file", help="Archivo con MathML: omite OCR y conversion (el resultado nunca quedara validado).")
    ap.add_argument("--review", action="store_true", help="Revision interactiva del LaTeX.")
    ap.add_argument("--provider", choices=["none", "ollama", "remote"], default="none", help="Proveedor LLM (por defecto: sin LLM).")
    ap.add_argument("--enrich-intents", action="store_true", help="Anade atributos intent al MathML usado en la lectura (experimental).")
    # accesibilidad
    ap.add_argument("--braille", action="store_true", help="Genera braille matematico con MathCAT.")
    ap.add_argument("--braille-code", help=f"Codigo braille: {', '.join(BRAILLE_CODES)} (por defecto el configurado).")
    ap.add_argument("--review-braille", action="store_true", help="Revision interactiva del braille.")
    ap.add_argument("--speech", action="store_true", help="Genera el texto de lectura matematica con MathCAT.")
    ap.add_argument("--language", help="Idioma de la lectura (por defecto el configurado, es).")
    ap.add_argument("--speech-style", choices=["ClearSpeak", "SimpleSpeak"])
    ap.add_argument("--verbosity", choices=["Terse", "Medium", "Verbose"])
    ap.add_argument("--audio", action="store_true", help="Sintetiza audio (requiere --tts).")
    ap.add_argument("--tts", choices=["none", "pyttsx3"], default="none", help="Proveedor TTS (por defecto: ninguno).")
    ap.add_argument("--voice-id", help="Identificador de voz del proveedor TTS.")
    ap.add_argument("--rate", type=int, help="Velocidad de la voz (proveedor TTS).")
    ap.add_argument("--play", action="store_true", help="Reproduce el audio generado (solo Windows, WAV).")
    ap.add_argument("--export-braille", metavar="ARCHIVO.txt", help="Guarda el braille como TEXTO Unicode UTF-8 (no BRF).")
    ap.add_argument("--export-audio", metavar="ARCHIVO.wav", help="Guarda el audio generado en esta ruta.")
    # salida
    ap.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "outputs"))
    ap.add_argument("--export", metavar="ARCHIVO.xml", help="Copia el XML final a esta ruta.")
    ap.add_argument("--expected-latex", help="LaTeX esperado, para medir la exactitud del OCR.")
    ap.add_argument("--json", action="store_true", help="Muestra solo el resumen por aspecto en JSON.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))

    image, latex, mathml = args.image, args.latex, None
    if args.mathml_file:
        try:
            mathml = Path(args.mathml_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"No se pudo leer {args.mathml_file}: {exc}", file=sys.stderr)
            return 2
    if image is None and latex is None and mathml is None:
        image = input("Ruta de la imagen (o Enter para escribir LaTeX): ").strip().strip('"') or None
        if image is None:
            latex = input("LaTeX: ").strip() or None
    if image is None and latex is None and mathml is None:
        print("Hace falta una imagen, un LaTeX o un MathML.", file=sys.stderr)
        return 2

    tts = Pyttsx3Provider() if args.tts == "pyttsx3" else None
    if args.export_audio:
        if tts is None:
            print("--export-audio necesita un proveedor TTS (--tts pyttsx3): sin el solo hay texto de lectura, no audio.", file=sys.stderr)
            return 2
        if Path(args.export_audio).suffix.lower() != f".{tts.output_format}":
            print(f"{tts.name} produce .{tts.output_format}; usa esa extension en --export-audio.", file=sys.stderr)
            return 2
    if args.export_braille and Path(args.export_braille).suffix.lower() in {".brf", ".brl"}:
        print("No se genera BRF: la salida es texto braille Unicode. Usa --export-braille salida.txt.", file=sys.stderr)
        return 2
    try:
        provider = build_provider(args.provider)
    except ProviderError as exc:
        print(f"Proveedor LLM no disponible: {exc}", file=sys.stderr)
        return 2

    want_audio = args.audio or bool(args.export_audio) or args.play
    report = run_pipeline(
        image_path=image, latex=latex, mathml=mathml, provider=provider,
        reviewer=interactive_reviewer if args.review else None,
        output_dir=args.output_dir, enrich_intents=args.enrich_intents,
        braille=args.braille or bool(args.export_braille), braille_code=args.braille_code,
        braille_reviewer=braille_reviewer if args.review_braille else None,
        speech=args.speech or want_audio, speech_language=args.language,
        speech_style=args.speech_style, speech_verbosity=args.verbosity,
        audio=want_audio, tts_provider=tts, audio_path=args.export_audio,
        voice_id=args.voice_id, voice_rate=args.rate,
    )
    show_report(report, args.expected_latex, args.json)

    code = 0 if report.validated else 1
    if args.export:
        if report.xml and report.xml.xml and report.xml.status is StageStatus.OK:
            Path(args.export).write_bytes(report.xml.xml)
            print(f"XML exportado a {args.export}")
        else:
            print("No se exporta el XML: no supero la validacion estructural.", file=sys.stderr)
            code = 3
    if args.export_braille:
        try:
            if report.braille is None or report.braille.status is not StageStatus.OK:
                raise ValueError("No hay braille generado que exportar.")
            export_braille_text(report.braille, args.export_braille)
            print(f"Braille (texto Unicode UTF-8) guardado en {args.export_braille}")
        except (ValueError, OSError) as exc:
            print(f"No se exporta el braille: {exc}", file=sys.stderr)
            code = 3
    if args.play:
        if report.audio is not None and report.audio.status is StageStatus.OK:
            ok, err = play_audio_file(report.audio.file_path)
            if not ok:
                print(f"No se pudo reproducir el audio: {err}", file=sys.stderr)
                code = code or 3
        else:
            print("No hay audio que reproducir.", file=sys.stderr)
            code = code or 3
    return code


if __name__ == "__main__":
    sys.exit(main())
