from types import SimpleNamespace

from image_to_mathml.cli_commands import _format_convert_output


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
