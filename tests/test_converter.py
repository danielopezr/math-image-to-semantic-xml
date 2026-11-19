from image_to_mathml.mathml.converter import LatexToMathML
from image_to_mathml.mathml.normalizer import normalize_mathml


def test_aligned_system_converts_to_one_cell_per_row():
    # The OCR read alg209 (a system of equations) with \begin{aligned}, which
    # latex2mathml turned into invalid XML (raw "&").
    latex = (
        r"\begin{aligned} 2x_1 - x_2 &= \lambda x_1 \\ "
        r"2x_1 - x_2 + x_3 &= \lambda x_2 \end{aligned}"
    )
    markup = normalize_mathml(LatexToMathML().convert(latex))
    assert markup.count("<mtr") == 2
    assert markup.count("<mtd") == 2


def test_stray_ampersands_are_escaped():
    # An environment latex2mathml does not know leaves its "&" in the markup.
    markup = LatexToMathML().convert(r"\begin{unknownenv} a & b \end{unknownenv}")
    normalize_mathml(markup)
