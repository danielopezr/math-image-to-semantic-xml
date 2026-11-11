from types import SimpleNamespace

from image_to_mathml.cli_commands import _format_convert_output
from image_to_mathml.timing import PhaseTiming


def test_convert_output_includes_ocr_latex_and_structure_source():
    result = SimpleNamespace(
        ocr_backend="pix2tex",
        latex="x^{2}+1",
        structure_source="latex2mathml",
        mathml="<math></math>",
    )

    output = _format_convert_output(result)

    assert "LaTeX OCR (pix2tex): x^{2}+1" in output
    assert "MathML structure: latex2mathml" in output
    assert output.rstrip().endswith("<math></math>")


def test_convert_output_omits_ocr_latex_for_direct_latex_input():
    result = SimpleNamespace(
        ocr_backend="none",
        latex="x",
        structure_source="llm",
        mathml="<math></math>",
    )

    output = _format_convert_output(result)

    assert "LaTeX OCR" not in output
    assert "MathML structure: llm" in output


def test_convert_output_includes_all_llm_structures():
    result = SimpleNamespace(
        ocr_backend="none",
        latex="x",
        structure_source="llm",
        mathml="<math>final</math>",
        llm_outputs=[
            SimpleNamespace(
                flow="direct", attempt=1, accepted=False, mathml="<math>first</math>"
            ),
            SimpleNamespace(
                flow="direct", attempt=2, accepted=True, mathml="<math>second</math>"
            ),
        ],
    )

    output = _format_convert_output(result)

    assert "MathML structures produced by the LLM (2)" in output
    assert "LLM output 1 (direct attempt 1, rejected)" in output
    assert "<math>first</math>" in output
    assert "LLM output 2 (direct attempt 2, accepted)" in output
    assert "<math>second</math>" in output
    assert output.rstrip().endswith("<math>final</math>")


def test_convert_output_includes_phase_timings_and_attempt_summary():
    result = SimpleNamespace(
        ocr_backend="none",
        latex="x",
        structure_source="llm",
        mathml="<math>final</math>",
        llm_outputs=[],
        elapsed_seconds=0.2,
        timings=[
            PhaseTiming("ocr", "fake-ocr", 25.0),
            PhaseTiming("latex2mathml", "latex2mathml", 50.0, note="fallback"),
            PhaseTiming("attempt", "direct[fake]", 100.0, attempt=1),
            PhaseTiming("llm", "direct[fake]", 90.0, attempt=1),
            PhaseTiming("validate", "validator", 5.0, attempt=1),
        ],
    )

    output = _format_convert_output(result)

    assert "Timings:" in output
    assert "ocr [fake-ocr]: 25.0 ms" in output
    assert "latex2mathml [latex2mathml] (fallback): 50.0 ms" in output
    assert "attempt 1 total [direct[fake]]: 100.0 ms" in output
    assert "attempts total: 100.0 ms; average: 100.0 ms (1 attempts)" in output
    assert "total pipeline: 200.0 ms" in output
    assert output.rstrip().endswith("<math>final</math>")
