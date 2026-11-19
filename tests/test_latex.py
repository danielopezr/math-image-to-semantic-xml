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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"\mathrm{det}\left(A\right)", r"\det \left(A\right)"),
        (r"\operatorname{sin}x", r"\sin x"),
        (r"\text{ln} x", r"\ln x"),
        (r"\mathrm{rank}(A)", r"\mathrm{rank}(A)"),
        (r"\mathrm{d}x", r"\mathrm{d}x"),
    ],
)
def test_clean_latex_writes_operator_names_as_commands(raw, expected):
    assert clean_latex(raw) == expected


def test_clean_latex_drops_small_spaces_but_not_row_breaks():
    assert clean_latex(r"\sqrt{2}\,t") == r"\sqrt{2} t"
    assert clean_latex(r"\int f\;dx\!") == r"\int f dx"
    assert clean_latex(r"a \\, b") == r"a \\, b"


def test_clean_latex_keeps_spanish_words_between_spaces_as_text():
    assert clean_latex(r"s > 0 \, y \, s \neq 0") == r"s > 0 \text{ y } s \neq 0"
    assert clean_latex(r"a \quad o \quad b") == r"a \quad\text{ o }\quad b"
    assert clean_latex(r"x\,e\,y") == r"x e y"
