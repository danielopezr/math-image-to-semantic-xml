"""CLI de math-semantic-ocr.

Ejemplos:
  python main.py ruta\\formula.png                       # OCR + MathML + reglas (sin LLM)
  python main.py ruta\\formula.png --review              # revision manual del LaTeX
  python main.py --latex "x^2 + 1 = 0"                   # sin imagen ni OCR
  python main.py ruta\\formula.png --provider ollama     # con LLM local
  python main.py                                         # modo interactivo
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from app.evaluation import summarize_stages
from app.models import PipelineReport, SemanticNode, StageStatus
from app.pipeline import RETRY, run_pipeline
from app.semantic_agent import OllamaProvider, OpenAICompatibleProvider, ProviderError


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_node(node: SemanticNode, depth: int = 0) -> None:
    label = f"{node.kind}" + (f" '{node.symbol}'" if node.symbol else "")
    print("  " * depth + f"- [{node.id}] {label}  ({node.evidence})")
    for child in node.children:
        _print_node(child, depth + 1)


def show_report(report: PipelineReport, expected_latex: Optional[str], as_json: bool) -> None:
    if as_json:
        print(json.dumps(summarize_stages(report, expected_latex), indent=2, ensure_ascii=False))
        return

    _print_header("LaTeX reconocido")
    if report.ocr and report.ocr.latex is not None:
        print(report.ocr.latex)
        if report.ocr.reviewed_by_human and report.ocr.latex_raw is not None:
            print(f"(original del OCR: {report.ocr.latex_raw})")
    else:
        print("(sin LaTeX)")

    _print_header("MathML (presentacion)")
    print(report.conversion.mathml if report.conversion and report.conversion.mathml else "(no generado)")

    _print_header("Interpretacion semantica")
    if report.semantic and report.semantic.output:
        out = report.semantic.output
        print(f"Tipo: {out.expression_type}   requiere revision: {out.requires_review}")
        _print_node(out.root)
        for rel in out.relations:
            print(f"Relacion: {rel.type}  {rel.left} -> {rel.right}  ({rel.evidence})")
        for text in out.assumptions:
            print(f"Supuesto: {text}")
        for text in out.ambiguities:
            print(f"Ambiguedad: {text}")
    else:
        print("(no disponible: " + report.skipped.get("semantic_agent", "la etapa fallo") + ")")

    _print_header("Reglas")
    if report.rules:
        for r in report.rules:
            print(f"{r.rule_id:8} {r.outcome.value:15} [{r.origin.value}] {r.message}")
    else:
        print("(sin evaluar: " + report.skipped.get("rules", "no disponible") + ")")

    _print_header("Errores y advertencias")
    issues = []
    for result in (report.ocr, report.conversion, report.semantic, report.xml):
        if result is not None:
            issues += [("ERROR", i) for i in result.errors] + [("AVISO", i) for i in result.warnings]
    for level, i in issues:
        print(f"{level:6} {i.stage}/{i.code}: {i.message}")
    if not issues:
        print("(ninguno)")

    _print_header("Estado por etapa")
    for name, data in summarize_stages(report, expected_latex).items():
        print(f"{name:20} {json.dumps(data, ensure_ascii=False)}")
    for stage, reason in report.skipped.items():
        print(f"OMITIDA {stage}: {reason}")
    print(f"\nResultado final validado: {'SI' if report.validated else 'NO'}")
    if report.output_dir:
        print(f"Resultados intermedios en: {report.output_dir}")


def interactive_reviewer(ocr):
    """Muestra el LaTeX y permite aceptar, corregir o repetir el OCR."""
    print("\n--- Revision del LaTeX ---")
    if ocr.errors:
        for e in ocr.errors:
            print(f"ERROR {e.code}: {e.message}")
    print(f"LaTeX actual (intento {ocr.attempt}): {ocr.latex if ocr.latex is not None else '(ninguno)'}")
    answer = input("Enter = aceptar | escribe el LaTeX corregido | 'r' = repetir OCR: ").strip()
    if answer == "":
        return None
    if answer.lower() == "r":
        return RETRY
    return answer


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

    parser = argparse.ArgumentParser(description="Imagen de formula -> LaTeX -> MathML -> interpretacion -> XML.")
    parser.add_argument("image", nargs="?", help="Ruta de la imagen con la formula.")
    parser.add_argument("--latex", help="LaTeX manual (omite el OCR si no se da imagen).")
    parser.add_argument("--review", action="store_true", help="Revision interactiva del LaTeX.")
    parser.add_argument("--provider", choices=["none", "ollama", "remote"], default="none",
                        help="Proveedor LLM (por defecto none: sin LLM).")
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "outputs"))
    parser.add_argument("--export", metavar="ARCHIVO.xml", help="Copia el XML final a esta ruta.")
    parser.add_argument("--expected-latex", help="LaTeX esperado, para medir la exactitud del OCR.")
    parser.add_argument("--json", action="store_true", help="Muestra solo el resumen por etapa en JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))

    image, latex = args.image, args.latex
    if image is None and latex is None:
        image = input("Ruta de la imagen (o Enter para escribir LaTeX): ").strip().strip('"') or None
        if image is None:
            latex = input("LaTeX: ").strip() or None
    if image is None and latex is None:
        print("Hace falta una imagen o un LaTeX.", file=sys.stderr)
        return 2

    try:
        provider = build_provider(args.provider)
    except ProviderError as exc:
        print(f"Proveedor LLM no disponible: {exc}", file=sys.stderr)
        return 2

    report = run_pipeline(
        image_path=image, latex=latex, provider=provider,
        reviewer=interactive_reviewer if args.review else None,
        output_dir=args.output_dir,
    )
    show_report(report, args.expected_latex, args.json)

    if args.export:
        if report.xml and report.xml.xml and report.xml.status is StageStatus.OK:
            Path(args.export).write_bytes(report.xml.xml)
            print(f"XML exportado a {args.export}")
        else:
            print("No se exporta: el XML no supero la validacion estructural.", file=sys.stderr)
    return 0 if report.validated else 1


if __name__ == "__main__":
    sys.exit(main())
