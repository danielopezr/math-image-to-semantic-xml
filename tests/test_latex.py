import pytest

from image_to_mathml.mathml.latex import clean_latex


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"$$x^2$$", "x^2"),
        (r"$x^2$", "x^2"),
        (r"\[ x^2 \]", "x^2"),
        (r"\( x^2 \)", "x^2"),
        (r"\begin{equation} x^2 \end{equation}", "x^2"),
        (r"\begin{equation*}a=b\end{equation*}", "a=b"),
        (r"\begin{displaymath} a \end{displaymath}", "a"),
        (r"x^2 \label{eq:1} \tag{3}", "x^2"),
    ],
)
def test_clean_latex_strips_delimiters_and_noise(raw, expected):
    assert clean_latex(raw) == expected


def test_clean_latex_requires_matching_environment_names():
    raw = r"\begin{equation} x \end{math}"
    assert clean_latex(raw) == raw


def test_clean_latex_leaves_several_inline_formulas_untouched():
    assert clean_latex(r"$a$ y $b$") == r"$a$ y $b$"
