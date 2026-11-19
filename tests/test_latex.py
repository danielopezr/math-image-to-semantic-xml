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


def test_clean_latex_joins_several_inline_formulas():
    # The prose between formulas becomes text instead of loose "$" tokens.
    assert clean_latex(r"$a$ y $b$") == r"a \text{ y } b"
    assert (
        clean_latex(r"d) $ a_{ij} = 0 $ si $ i > j $")
        == r"\text{ d) } a_{ij} = 0 \text{ si } i > j"
    )
    assert clean_latex(r"$\frac{4a}{3bc}$.") == r"\frac{4a}{3bc} ."
    assert clean_latex(r"\((b_1 - a_1)\).") == r"(b_1 - a_1) ."


def test_clean_latex_drops_unpaired_or_stray_dollars_but_not_escaped_ones():
    assert clean_latex(r"a = 3 $ b") == "a = 3 b"
    assert clean_latex(r"f dz$$ $$= g") == "f dz = g"
    assert clean_latex(r"\$5 + x") == r"\$5 + x"


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


@pytest.mark.parametrize("env", ["aligned", "align*", "eqnarray", "split"])
def test_clean_latex_writes_alignment_environments_as_one_column_arrays(env):
    raw = (
        r"\begin{" + env + r"} 2x_1 - x_2 &= \lambda x_1 \\ "
        r"x_3 &= 0 \end{" + env + "}"
    )
    assert clean_latex(raw) == (
        r"\begin{array}{l} 2x_1 - x_2 = \lambda x_1 \\ x_3 = 0 \end{array}"
    )


def test_clean_latex_keeps_nested_matrix_cells_and_escaped_ampersands():
    raw = r"\begin{aligned} A &= \begin{bmatrix} 1 & 2 \end{bmatrix} \\ \& &= 0 \end{aligned}"
    assert clean_latex(raw) == (
        r"\begin{array}{l} A = \begin{bmatrix} 1 & 2 \end{bmatrix} \\ \& = 0 \end{array}"
    )


def test_clean_latex_wraps_bare_aligned_rows_in_an_array():
    assert (
        clean_latex(r"a &= b \\ c &= d") == r"\begin{array}{l}a = b \\ c = d\end{array}"
    )


def test_clean_latex_leaves_matrices_and_cases_untouched():
    for raw in (
        r"\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}",
        r"\begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}",
        r"\left[ \begin{array}{cc} 1 & 2 \end{array} \right]",
    ):
        assert clean_latex(raw) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"i \sen 2\theta", r"i \sin 2\theta"),
        (r"\mathrm{sen} kx", r"\sin kx"),
        (r"\operatorname{tg}x", r"\tan x"),
        (r"\senh x", r"\sinh x"),
        (r"\sentence", r"\sentence"),
    ],
)
def test_clean_latex_writes_spanish_function_names_as_commands(raw, expected):
    assert clean_latex(raw) == expected
