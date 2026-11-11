"""Functions for Command line interface for the image-to-MathML module."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .llm.base import LLMError
from .llm.factory import build_llm
from .mathml.validator import validate_mathml
from .models import EquationRequest, Flow, MathContext
from .pipeline import EquationPipeline, compare_flows


def _settings_from_args(args: argparse.Namespace) -> Settings:
    changes: dict[str, object] = {}
    if getattr(args, "model", None):
        changes["ollama_model"] = args.model
    if getattr(args, "vision_model", None):
        changes["ollama_vision_model"] = args.vision_model
    backend = getattr(args, "backend", None) or Settings.from_env().llm_backend
    backend = backend.strip().lower()
    if getattr(args, "backend", None):
        changes["llm_backend"] = backend
    if getattr(args, "base_url", None):
        if backend in {"openai", "openai-compatible", "vllm", "vert", "lmstudio"}:
            changes["openai_base_url"] = args.base_url
        else:
            changes["ollama_base_url"] = args.base_url
    if getattr(args, "api_key", None):
        changes["openai_api_key"] = args.api_key
    if getattr(args, "timeout", None) is not None:
        changes["ollama_timeout"] = args.timeout
    if getattr(args, "max_repairs", None) is not None:
        changes["max_repair_attempts"] = args.max_repairs
    if getattr(args, "ocr", None):
        changes["ocr_backend"] = args.ocr
    if getattr(args, "no_preprocess", False):
        changes["ocr_preprocess"] = False
    if getattr(args, "preprocess_profile", None):
        changes["ocr_preprocess_profile"] = args.preprocess_profile
        if args.preprocess_profile == "off":
            changes["ocr_preprocess"] = False
    if getattr(args, "strict_core", False):
        changes["allow_open_concepts"] = False
    if getattr(args, "allow_open_concepts", False):
        changes["allow_all_open_concepts"] = True
    if getattr(args, "allowed_open_concepts", None):
        changes["allowed_open_concepts"] = tuple(
            item.strip()
            for item in args.allowed_open_concepts.split(",")
            if item.strip()
        )
    return replace(Settings.from_env(), **changes)


def _read_latex(args: argparse.Namespace) -> str | None:
    latex = getattr(args, "latex", None)
    image = getattr(args, "image", None)
    if latex == "-":
        return sys.stdin.read()
    if latex:
        return latex
    if not image:
        if not sys.stdin.isatty():
            return sys.stdin.read()
        raise SystemExit(
            "Provide --latex, --image, or pipe a LaTeX expression on stdin."
        )
    return None


def _request_from_args(args: argparse.Namespace) -> EquationRequest:
    latex = _read_latex(args)
    return EquationRequest(
        latex=latex,
        image_path=getattr(args, "image", None),
        context=MathContext(
            area=getattr(args, "area", "") or "",
            topic=getattr(args, "topic", "") or "",
            surrounding_text=getattr(args, "context_text", "") or "",
            language=getattr(args, "lang", "es") or "es",
        ),
        flow=Flow(getattr(args, "flow", "hybrid")),
        ocr_backend=getattr(args, "ocr", None),
        allow_open_concepts=False if getattr(args, "strict_core", False) else None,
        max_repair_attempts=getattr(args, "max_repairs", None),
        direct_vision=True if getattr(args, "direct_vision", False) else None,
    )


def _print_report(result, stream=sys.stderr) -> None:
    report = result.validation
    for issue in report.issues:
        print(issue.format(), file=stream)
    print(
        f"validation: {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s), {report.intent_count} intent(s), "
        f"{report.redundant_mrows} redundant mrow(s)",
        file=stream,
    )


def _llm_output_label(output, index: int) -> str:
    flow = getattr(output, "flow", "")
    flow_name = getattr(flow, "value", flow) or "llm"
    attempt = getattr(output, "attempt", index)
    accepted = getattr(output, "accepted", False)
    status = "accepted" if accepted else "rejected"
    return f"LLM output {index} ({flow_name} attempt {attempt}, {status})"


def _format_llm_outputs(result) -> list[str]:
    outputs = getattr(result, "llm_outputs", None) or []
    if not outputs:
        return []
    lines = [f"MathML structures produced by the LLM ({len(outputs)}):"]
    for index, output in enumerate(outputs, 1):
        lines.append("")
        lines.append(f"--- {_llm_output_label(output, index)} ---")
        lines.append(getattr(output, "mathml", str(output)))
    lines.append("")
    return lines


def _timing_note(timing) -> str:
    note = getattr(timing, "note", "")
    return f" ({note})" if note else ""


def _timing_group(timing) -> str:
    group = getattr(timing, "group", "")
    return f"[{group}] " if group else ""


def _format_timings(result) -> list[str]:
    timings = getattr(result, "timings", None) or []
    if not timings:
        return []

    lines = ["Timings:"]
    for timing in timings:
        if timing.attempt is not None or timing.phase == "attempt":
            continue
        lines.append(
            f"- {_timing_group(timing)}{timing.phase} [{timing.tool}]"
            f"{_timing_note(timing)}: {timing.duration_ms:.1f} ms"
        )

    attempts = [timing for timing in timings if timing.phase == "attempt"]
    if attempts:
        lines.append("Attempts:")
        per_attempt: dict[tuple[str, int], list[object]] = {}
        for timing in timings:
            if timing.attempt is not None and timing.phase != "attempt":
                key = (getattr(timing, "group", ""), timing.attempt)
                per_attempt.setdefault(key, []).append(timing)
        for total in attempts:
            key = (getattr(total, "group", ""), total.attempt)
            lines.append(
                f"- {_timing_group(total)}attempt {total.attempt} total "
                f"[{total.tool}]: {total.duration_ms:.1f} ms"
            )
            for timing in per_attempt.get(key, []):
                lines.append(
                    f"    {timing.phase} [{timing.tool}]"
                    f"{_timing_note(timing)}: {timing.duration_ms:.1f} ms"
                )
        attempts_total = sum(timing.duration_ms for timing in attempts)
        lines.append(
            f"- attempts total: {attempts_total:.1f} ms; average: "
            f"{attempts_total / len(attempts):.1f} ms ({len(attempts)} attempts)"
        )

    summary = (
        result.timing_summary()
        if hasattr(result, "timing_summary")
        else {"total_ms": getattr(result, "elapsed_seconds", 0.0) * 1000}
    )
    lines.append(f"- total pipeline: {summary.get('total_ms', 0.0):.1f} ms")
    return lines


def _format_convert_output(result) -> str:
    lines = []
    if result.ocr_backend != "none":
        lines.append(f"LaTeX OCR ({result.ocr_backend}): {result.latex}")
    lines.append(f"MathML structure: {result.structure_source}")
    lines.extend(_format_llm_outputs(result))
    lines.extend(_format_timings(result))
    lines.append("")
    lines.append(result.mathml)
    return "\n".join(lines)


def cmd_convert(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    pipeline = EquationPipeline(settings)
    request = _request_from_args(args)
    result = pipeline.run(request)

    if getattr(args, "output", None):
        Path(args.output).write_text(result.mathml + "\n", encoding="utf-8")

    if getattr(args, "json", False):
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(_format_convert_output(result))

    if not getattr(args, "quiet", False):
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if getattr(args, "report", False):
            _print_report(result)

    return 0 if result.validation.ok else 2


def _comparison_markdown(results: dict[str, object]) -> str:
    lines = ["# Comparación de flujos", ""]
    latex_ocr = next(
        (result.latex for result in results.values() if result.ocr_backend != "none"),
        None,
    )
    if latex_ocr is not None:
        lines.append(f"LaTeX OCR: {latex_ocr}")
        lines.append("")
    lines.extend(
        [
            (
                "| flujo | estructura | XML válido | aceptable | intents | "
                "errores | warnings | intentos | segundos |"
            ),
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for name, result in results.items():
        report = result.validation
        lines.append(
            f"| {name} | {result.structure_source} | {'sí' if report.ok else 'no'} | "
            f"{'sí' if report.acceptable else 'no'} | {report.intent_count} | "
            f"{len(report.errors)} | {len(report.warnings)} | "
            f"{result.attempts} | {result.elapsed_seconds:.2f} |"
        )
    lines.append("")
    for name, result in results.items():
        if result.warnings:
            lines.append(f"## {name}: advertencias")
            lines.extend(f"- {warning}" for warning in result.warnings)
            lines.append("")
    for name, result in results.items():
        timing_lines = _format_timings(result)
        if timing_lines:
            lines.append(f"## {name}: tiempos")
            lines.extend(timing_lines)
            lines.append("")
    for name, result in results.items():
        outputs = getattr(result, "llm_outputs", None) or []
        if not outputs:
            continue
        lines.append(f"## {name}: estructuras MathML del LLM")
        lines.append("")
        for index, output in enumerate(outputs, 1):
            lines.append(f"### {_llm_output_label(output, index)}")
            lines.append("")
            lines.append("```xml")
            lines.append(getattr(output, "mathml", str(output)).strip())
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    pipeline = EquationPipeline(settings)
    request = _request_from_args(args)
    results = compare_flows(request, pipeline)

    payload = {name: result.as_dict() for name, result in results.items()}
    if getattr(args, "output_dir", None):
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, result in results.items():
            (directory / f"{name}.mathml").write_text(
                result.mathml + "\n", encoding="utf-8"
            )
        (directory / "comparison.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown = _comparison_markdown(results)
        (directory / "comparison.md").write_text(markdown + "\n", encoding="utf-8")

    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif getattr(args, "output_dir", None):
        print(_comparison_markdown(results))
        print(f"\nResultados escritos en {args.output_dir}", file=sys.stderr)
    else:
        print(_comparison_markdown(results))

    return 0 if all(result.validation.ok for result in results.values()) else 2


def cmd_check(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    if getattr(args, "file", None):
        markup = Path(args.file).read_text(encoding="utf-8")
    elif getattr(args, "mathml", None):
        markup = args.mathml
    else:
        markup = sys.stdin.read()

    allowed_open = (
        None
        if settings.allow_all_open_concepts
        else set(settings.allowed_open_concepts)
    )
    if getattr(args, "allowed_open_concepts", None):
        allowed_open = {
            item.strip()
            for item in args.allowed_open_concepts.split(",")
            if item.strip()
        }
    report = validate_mathml(
        markup,
        allow_open_concepts=not getattr(args, "strict_core", False),
        allowed_open_concepts=allowed_open,
    )
    if getattr(args, "json", False):
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        for issue in report.issues:
            print(issue.format())
        print(
            f"{'OK' if report.ok else 'FAIL'}: {len(report.errors)} error(s), "
            f"{len(report.warnings)} warning(s), {report.intent_count} intent(s), "
            f"{report.redundant_mrows} redundant mrow(s)"
        )
    return 0 if report.ok else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    modules = {
        "latex2mathml": importlib.util.find_spec("latex2mathml") is not None,
        "requests": importlib.util.find_spec("requests") is not None,
        "lxml": importlib.util.find_spec("lxml") is not None,
        "Pillow": importlib.util.find_spec("PIL") is not None,
        "pix2tex": importlib.util.find_spec("pix2tex") is not None,
        "texify": importlib.util.find_spec("texify") is not None,
        "rapid_latex_ocr": importlib.util.find_spec("rapid_latex_ocr") is not None,
        "pix2text": importlib.util.find_spec("pix2text") is not None,
        "paddleocr": importlib.util.find_spec("paddleocr") is not None,
    }
    client = build_llm(settings)
    available = client.is_available()
    models: list[str] = []
    if available:
        try:
            models = client.list_models()
        except LLMError:
            models = []
    has_model = any(
        name == settings.ollama_model
        or name.startswith(settings.ollama_model + ":")
        or settings.ollama_model in name
        for name in models
    )

    payload = {
        "python": sys.version.split()[0],
        "modules": modules,
        "ocr": {
            "backend": settings.ocr_backend,
            "preprocess": settings.ocr_preprocess,
            "preprocess_profile": settings.ocr_preprocess_profile,
        },
        "llm": {
            "backend": settings.llm_backend,
            "base_url": getattr(client, "base_url", settings.ollama_base_url),
            "available": available,
            "model": settings.ollama_model,
            "model_installed": has_model,
            "models": models,
        },
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Python: {payload['python']}")
        for name, installed in modules.items():
            print(f"{name}: {'installed' if installed else 'not installed'}")
        print(f"OCR backend: {settings.ocr_backend}")
        preprocess_state = "enabled" if settings.ocr_preprocess else "disabled"
        print(
            f"OCR preprocessing: {preprocess_state} "
            f"(profile: {settings.ocr_preprocess_profile})"
        )
        print(f"LLM backend: {settings.llm_backend}")
        print(f"LLM URL: {payload['llm']['base_url']}")
        print(f"LLM reachable: {'yes' if available else 'no'}")
        print(
            f"Model {settings.ollama_model}: {'installed' if has_model else 'missing'}"
        )
        if models:
            print("Installed models: " + ", ".join(models))

    required = ("latex2mathml", "requests", "lxml", "Pillow")
    ok = all(modules[name] for name in required) and available and has_model
    return 0 if ok else 1
