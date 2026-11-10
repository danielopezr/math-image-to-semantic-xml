"""Command line interface for the image-to-MathML module."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .cli_commands import cmd_check, cmd_compare, cmd_convert, cmd_doctor
from .models import Flow
from .ocr.factory import OCR_BACKENDS


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--latex", "-l", help="LaTeX expression, or '-' for stdin")
    parser.add_argument("--image", "-i", help="Equation image path")
    parser.add_argument(
        "--area", default="", help="Mathematics area from the book title"
    )
    parser.add_argument(
        "--topic", default="", help="Topic detected by the previous stage"
    )
    parser.add_argument("--context-text", default="", help="Surrounding OCR text")
    parser.add_argument("--lang", default="es", help="Reader language (default: es)")
    parser.add_argument(
        "--flow",
        choices=[flow.value for flow in Flow],
        default=Flow.HYBRID.value,
        help="Generation flow (default: hybrid)",
    )
    parser.add_argument(
        "--ocr",
        choices=OCR_BACKENDS,
        metavar="BACKEND",
        help="Equation OCR backend: auto, texify, pix2tex, rapid_latex_ocr, "
        "pix2text, paddleocr or none (default: auto)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable contrast, deskew, crop and sharpening before OCR",
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=["safe", "standard", "aggressive", "off"],
        default=None,
        help="Image preprocessing profile (default: safe: only polarity)",
    )
    parser.add_argument(
        "--strict-core",
        action="store_true",
        help="Reject all open concepts, including gamma-function",
    )
    parser.add_argument(
        "--allow-open-concepts",
        action="store_true",
        help="Allow any documented W3C open concept (default: only the "
        "documented gamma-function, beta-function and error-function exceptions)",
    )
    parser.add_argument(
        "--allowed-open-concepts",
        default=None,
        help="Comma-separated open concepts to allow, e.g. "
        "gamma-function,airy-function",
    )
    parser.add_argument("--max-repairs", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--backend",
        default=None,
        help="LLM backend: ollama (default) or an OpenAI-compatible server "
        "(openai, vllm, vert, lmstudio, openai-compatible)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM server base URL. For OpenAI-compatible servers include /v1.",
    )
    parser.add_argument("--api-key", default=None, help="API key for the LLM server")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--output", "-o", help="Write the MathML result to this file")
    parser.add_argument("--quiet", "-q", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-to-mathml",
        description=(
            "Local equation recognition: image/LaTeX -> accessible MathML "
            "with W3C intent attributes."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser(
        "convert", help="Recognise one equation and produce MathML"
    )
    _add_common(convert)
    convert.set_defaults(func=cmd_convert)

    compare = subparsers.add_parser(
        "compare", help="Run hybrid, direct and offline flows for the same input"
    )
    _add_common(compare)
    compare.add_argument(
        "--output-dir", help="Write <flow>.mathml, comparison.json and comparison.md"
    )
    compare.set_defaults(func=cmd_compare)

    check = subparsers.add_parser("check", help="Validate a MathML + intent document")
    check.add_argument("--file", "-f", help="MathML file to validate")
    check.add_argument("--mathml", help="MathML string to validate")
    check.add_argument("--strict-core", action="store_true")
    check.add_argument("--allow-open-concepts", action="store_true")
    check.add_argument("--allowed-open-concepts", default=None)
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_check)

    doctor = subparsers.add_parser("doctor", help="Check local dependencies and Ollama")
    doctor.add_argument("--model", default=None)
    doctor.add_argument("--backend", default=None)
    doctor.add_argument("--base-url", default=None)
    doctor.add_argument("--api-key", default=None)
    doctor.add_argument("--timeout", type=float, default=None)
    doctor.add_argument("--strict-core", action="store_true")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
