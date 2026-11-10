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
