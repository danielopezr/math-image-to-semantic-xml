"""Compact concept cards used to retrieve intent candidates.

The cards intentionally contain retrieval signals rather than prose
definitions: canonical name, arity, argument roles, lexical aliases, LaTeX
triggers, MathML tags, symbols and short examples.  They are built from the
same registry used by the validator, so retrieval can never introduce an
unknown concept name.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .annotator import FUNCTION_CONCEPTS
from .concepts import Concept, ConceptRegistry, get_registry
from .semantic import CONCEPT_ROLES


@dataclass(frozen=True)
class CardSignature:
    fixity: str
    min_arity: int
    max_arity: int | None


@dataclass(frozen=True)
class ConceptCard:
    name: str
    kind: str
    signatures: tuple[CardSignature, ...]
    roles: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    latex: tuple[str, ...] = ()
    mathml_tags: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    requires_all: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    priority: int = 0


def _function_cards() -> dict[str, dict[str, object]]:
    grouped: dict[str, list[str]] = {}
    for token, concept in FUNCTION_CONCEPTS.items():
        grouped.setdefault(concept, []).append(token)

    cards: dict[str, dict[str, object]] = {}
    for concept, tokens in grouped.items():
        latex = tuple(
            f"\\{token}" for token in tokens if token.isascii() and token.isalpha()
        )
        symbols = tuple(sorted(set(tokens)))
        aliases = tuple(sorted({*tokens, concept, concept.replace("-", " ")}))
        example_token = next(
            (token for token in tokens if token.isascii() and token.isalpha()),
            concept,
        )
        cards[concept] = {
            "aliases": aliases,
            "latex": latex,
            "symbols": symbols,
            "examples": (f"\\{example_token}(x)",),
            "requires": tuple(sorted({*latex, *symbols})),
            "roles": ("arg",),
        }
    return cards


_STRUCTURE_OVERRIDES: dict[str, dict[str, object]] = {
    "fraction": {
        "aliases": ("fraction", "frac", "over"),
        "latex": ("\\frac", "\\dfrac", "\\tfrac"),
        "mathml_tags": ("mfrac",),
        "symbols": ("/",),
        "examples": ("\\frac{a}{b}",),
        "requires": ("mfrac", "\\frac", "/"),
        "roles": ("num", "den"),
        "priority": 5,
    },
    "power": {
        "aliases": ("power", "exponent", "squared", "cubed", "raised to"),
        "latex": ("^",),
        "mathml_tags": ("msup",),
        "symbols": ("^",),
        "examples": ("x^{2}",),
        "requires": ("msup", "^"),
        "roles": ("base", "exp"),
        "priority": 5,
    },
    "exponential": {
        "aliases": ("exponential", "exp"),
        "latex": ("e^", "\\exp"),
        "mathml_tags": ("msup",),
        "examples": ("e^{x}", "\\exp(x)"),
        "requires": ("e^", "\\exp", "exp"),
        "roles": ("exp",),
        "priority": 3,
    },
    "root": {
        "aliases": ("root", "square root", "radical", "sqrt"),
        "latex": ("\\sqrt",),
        "mathml_tags": ("msqrt", "mroot"),
        "symbols": ("\u221a",),
        "examples": ("\\sqrt{x}", "\\sqrt[3]{x}"),
        "requires": ("msqrt", "mroot", "\\sqrt", "\u221a"),
        "roles": ("radicand", "index"),
        "priority": 5,
    },
    "indexed-by": {
        "aliases": ("indexed by", "subscript", "sub", "index"),
        "latex": ("_",),
        "mathml_tags": ("msub",),
        "symbols": ("_",),
        "examples": ("x_{i}",),
        "requires": ("msub", "_"),
        "roles": ("base", "sub"),
        "priority": 4,
    },
    "factorial": {
        "aliases": ("factorial",),
        "symbols": ("!",),
        "examples": ("n!",),
        "requires": ("!",),
        "roles": ("arg",),
        "priority": 4,
    },
    "absolute-value": {
        "aliases": ("absolute value", "abs"),
        "symbols": ("|",),
        "examples": ("|x|",),
        "requires": ("|",),
        "roles": ("arg",),
        "priority": 4,
    },
    "norm": {
        "aliases": ("norm", "magnitude"),
        "symbols": ("\u2016", "\u2225"),
        "examples": ("\\lVert x \\rVert",),
        "requires": ("\u2016", "\u2225"),
        "roles": ("arg",),
        "priority": 3,
    },
    "sum": {
        "aliases": ("sum", "summation", "sigma"),
        "latex": ("\\sum",),
        "symbols": ("\u2211",),
        "mathml_tags": ("munder", "munderover", "msubsup"),
        "examples": ("\\sum_{i=1}^{n} a_{i}",),
        "requires": ("\\sum", "\u2211"),
        "roles": ("arg1", "arg2", "arg3"),
        "priority": 4,
    },
    "product": {
        "aliases": ("product", "product over", "pi product"),
        "latex": ("\\prod",),
        "symbols": ("\u220f",),
        "mathml_tags": ("munder", "munderover", "msubsup"),
        "examples": ("\\prod_{i=1}^{n} a_{i}",),
        "requires": ("\\prod", "\u220f"),
        "roles": ("arg1", "arg2", "arg3"),
        "priority": 4,
    },
    "definite-integral": {
        "aliases": ("integral", "definite integral"),
        "latex": ("\\int",),
        "symbols": ("\u222b",),
        "examples": ("\\int_{0}^{1} f(x)\\,dx",),
        "requires": ("\\int", "\u222b"),
        "roles": ("arg1", "arg2"),
        "priority": 4,
    },
    "derivative": {
        "aliases": ("derivative", "differentiate", "d dx"),
        "latex": ("\\frac{d}", "\\mathrm{d}"),
        "examples": ("\\frac{d}{dx} f(x)",),
        "requires": (),
        "roles": ("arg1", "arg2", "arg3"),
        "priority": 2,
    },
    "partial-derivative": {
        "aliases": ("partial derivative",),
        "latex": ("\\partial",),
        "symbols": ("\u2202",),
        "examples": ("\\frac{\\partial f}{\\partial x}",),
        "requires": ("\\partial", "\u2202"),
        "roles": ("arg1", "arg2", "arg3"),
        "priority": 4,
    },
    "limit": {
        "aliases": ("limit", "lim"),
        "latex": ("\\lim",),
        "symbols": ("lim",),
        "mathml_tags": ("munder", "munderover", "msub", "msubsup"),
        "examples": ("\\lim_{x \\to 0} f(x)",),
        "requires": ("\\lim", "lim"),
        "roles": ("arg",),
        "priority": 4,
    },
    "logarithm": {
        "aliases": ("logarithm", "log", "log base"),
        "latex": ("\\log",),
        "examples": ("\\log_{2}(x)",),
        "requires": ("log", "\\log"),
        "roles": ("arg", "base"),
        "priority": 4,
    },
    "natural-logarithm": {
        "aliases": ("natural logarithm", "natural log", "ln"),
        "latex": ("\\ln",),
        "examples": ("\\ln(x)",),
        "requires": ("ln", "\\ln"),
        "roles": ("arg",),
        "priority": 4,
    },
    "determinant": {
        "aliases": ("determinant", "det"),
        "latex": ("\\det",),
        "symbols": ("det",),
        "examples": ("\\det(A)",),
        "requires": ("det", "\\det"),
        "roles": ("arg",),
        "priority": 3,
    },
    "matrix": {
        "aliases": ("matrix",),
        "mathml_tags": ("mtable",),
        "examples": ("\\begin{matrix} a & b \\\\ c & d \\end{matrix}",),
        "requires": ("mtable",),
        "roles": ("arg",),
        "priority": 2,
    },
    "inner-product": {
        "aliases": ("inner product", "dot product", "scalar product"),
        "latex": ("\\cdot", "\\langle", "\\rangle"),
        "symbols": ("\u22c5", "\u00b7", "\u27e8", "\u27e9"),
        "examples": ("a \\cdot b", "\\langle a,b \\rangle"),
        "requires": ("\u22c5", "\u00b7", "\u27e8", "\\cdot", "\\langle"),
        "roles": ("arg1", "arg2"),
        "priority": 4,
    },
    "dot-product": {
        "aliases": ("dot product", "inner product", "scalar product"),
        "latex": ("\\cdot",),
        "symbols": ("\u22c5", "\u00b7"),
        "examples": ("a \\cdot b",),
        "requires": ("\u22c5", "\u00b7", "\\cdot"),
        "roles": ("arg1", "arg2"),
        "priority": 4,
    },
    "cross-product": {
        "aliases": ("cross product", "vector product"),
        "latex": ("\\times",),
        "symbols": ("\u00d7",),
        "examples": ("a \\times b",),
        "requires": ("\u00d7", "\\times"),
        "roles": ("arg1", "arg2"),
        "priority": 4,
    },
    "outer-product": {
        "aliases": ("outer product", "tensor product"),
        "latex": ("\\otimes",),
        "symbols": ("\u2297",),
        "examples": ("a \\otimes b",),
        "requires": ("\u2297", "\\otimes"),
        "roles": ("arg1", "arg2"),
        "priority": 4,
    },
    "cartesian-product": {
        "aliases": ("cartesian product", "product set"),
        "latex": ("\\times",),
        "symbols": ("\u00d7",),
        "examples": ("A \\times B",),
        "requires": ("\u00d7", "\\times"),
        "roles": ("arg1", "arg2"),
        "priority": 3,
    },
    "ordered-pair": {
        "aliases": ("ordered pair", "pair"),
        "latex": ("\\langle", "\\rangle"),
        "symbols": ("(", ")", ",", "\u27e8", "\u27e9"),
        "examples": ("(a,b)", "\\langle a,b \\rangle"),
        "requires": ("(", "\u27e8", "\\langle"),
        "requires_all": (",",),
        "roles": ("arg1", "arg2"),
        "priority": 4,
    },
    "coordinate": {
        "aliases": ("coordinate", "coordinates"),
        "symbols": ("(", ")", ",", "\u27e8", "\u27e9"),
        "examples": ("(x,y)",),
        "requires": ("(", "\u27e8", "\\langle"),
        "requires_all": (",",),
        "roles": ("arg1", "arg2"),
        "priority": 3,
    },
    "polar-coordinate": {
        "aliases": ("polar coordinate", "polar coordinates"),
        "symbols": ("(", ")", ","),
        "examples": ("(r,\\theta)",),
        "requires": ("(", "\u27e8"),
        "requires_all": (",",),
        "roles": ("arg1", "arg2"),
        "priority": 3,
    },
    "spherical-coordinate": {
        "aliases": ("spherical coordinate", "spherical coordinates"),
        "symbols": ("(", ")", ","),
        "examples": ("(r,\\theta,\\phi)",),
        "requires": ("(", "\u27e8"),
        "requires_all": (",",),
        "roles": ("arg1", "arg2", "arg3"),
        "priority": 3,
    },
    "cartesian-coordinate": {
        "aliases": ("cartesian coordinate", "cartesian coordinates"),
        "symbols": ("(", ")", ","),
        "examples": ("(x,y)",),
        "requires": ("(", "\u27e8"),
        "requires_all": (",",),
        "roles": ("arg1", "arg2"),
        "priority": 3,
    },
    "set": {
        "aliases": ("set", "set of"),
        "symbols": ("{", "}"),
        "examples": ("\\{a,b,c\\}",),
        "requires": ("{", "}"),
        "requires_all": ("{", "}"),
        "roles": ("arg",),
        "priority": 5,
    },
}

_TARGET_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("mo", "mrow"),
        (
            "and",
            "applied-to",
            "approximately",
            "composed-with",
            "congruent",
            "constraint",
            "cross-product",
            "defined-as",
            "direct-product",
            "divided-by",
            "divides",
            "does-not-belong-to",
            "does-not-divide",
            "dot-product",
            "element-of",
            "ellipsis",
            "equals",
            "equivalent-to",
            "evaluates-to",
            "given",
            "greater-than",
            "greater-than-or-equal-to",
            "identically-equals",
            "if-and-only-if",
            "implies",
            "inner-product",
            "intersection",
            "list-separator",
            "maps-to",
            "member-of",
            "minus",
            "minus-or-plus",
            "not",
            "not-equal-to",
            "not-member-of",
            "not-parallel-to",
            "not-subset",
            "not-superset",
            "obtained-from",
            "or",
            "outer-product",
            "parallel-to",
            "percent",
            "perpendicular",
            "plus",
            "plus-or-minus",
            "precedes",
            "proportional",
            "range-separator",
            "set-difference",
            "similar",
            "subset",
            "subset-or-equal",
            "succeeds",
            "such-that",
            "superset",
            "superset-or-equal",
            "tends-to",
            "tends-to-from-above",
            "tends-to-from-below",
            "tilde",
            "time-separator",
            "times",
            "union",
            "upwards-diagonal-ellipsis",
            "downwards-diagonal-ellipsis",
            "vertical-ellipsis",
            "xor",
        ),
    ),
    (
        ("mrow", "mo"),
        (
            "cartesian-coordinate",
            "closed-interval",
            "closed-open-interval",
            "coordinate",
            "fenced-group",
            "list",
            "open-closed-interval",
            "open-interval",
            "ordered-pair",
            "polar-coordinate",
            "set",
            "spherical-coordinate",
            "tuple",
        ),
    ),
    (
        ("mrow", "mo", "mover", "munder"),
        (
            "angle",
            "angle-measure",
            "annotation",
            "arc",
            "area",
            "braced-group",
            "conditional-probability",
            "directed-line-segment",
            "distance",
            "embellished-name",
            "length",
            "line",
            "line-segment",
            "measured-angle",
            "point",
            "ray",
            "right-angle",
            "volume",
        ),
    ),
    (
        ("mi", "mo", "mn", "mrow"),
        (
            "diameter",
            "differential-d",
            "empty-set",
            "exponential-e",
            "golden-ratio",
            "imaginary-i",
            "probability",
            "radius",
            "set-of-complex-numbers",
            "set-of-integers",
            "set-of-natural-numbers",
            "set-of-primes",
            "set-of-rationals",
            "set-of-reals",
        ),
    ),
    (
        ("mrow", "mi", "mo", "mover"),
        (
            "magnitude",
            "mean",
            "median",
            "mode",
            "norm",
            "span",
            "standard-deviation",
            "unit-vector",
            "variance",
            "vector",
        ),
    ),
    (
        ("mtable", "mrow"),
        (
            "identity-matrix",
            "matrix",
        ),
    ),
    (
        ("mfrac", "mrow"),
        ("fraction",),
    ),
    (
        ("mrow", "mfrac"),
        ("mixed-fraction",),
    ),
    (
        ("mfrac", "mrow", "mo"),
        (
            "quotient",
            "rate",
            "ratio",
            "remainder",
        ),
    ),
    (
        ("msup", "msubsup", "mrow"),
        ("power",),
    ),
    (
        ("msqrt", "mroot", "mrow"),
        ("root",),
    ),
    (
        ("mo", "mrow", "msqrt"),
        ("square-root-of",),
    ),
    (
        ("msub", "msubsup", "mrow"),
        ("indexed-by",),
    ),
    (
        ("msup", "mrow"),
        ("exponential",),
    ),
    (
        ("mo", "mrow"),
        (
            "absolute-value",
            "cardinality",
            "ceiling",
            "factorial",
            "floor",
            "fractional-part",
            "round",
        ),
    ),
    (
        ("mo", "mrow", "mover"),
        ("complex-conjugate",),
    ),
    (
        ("mi", "mo", "mrow"),
        (
            "complex-arg",
            "imaginary-part",
            "real-part",
        ),
    ),
    (
        ("msup", "mover", "mo", "mrow"),
        ("transpose",),
    ),
    (
        ("msup", "mrow"),
        ("inverse",),
    ),
    (
        ("mrow", "msup", "mi"),
        ("adjugate",),
    ),
    (
        ("munder", "munderover", "msub", "msubsup", "mrow"),
        (
            "limit",
            "product",
            "sum",
        ),
    ),
    (
        ("msubsup", "munderover", "msub", "munder", "mrow", "mo"),
        ("definite-integral",),
    ),
    (
        ("mfrac", "mrow", "msup"),
        (
            "derivative",
            "partial-derivative",
        ),
    ),
    (
        ("mo", "mrow", "msub", "msubsup"),
        ("evaluated-at",),
    ),
    (
        ("mrow", "mo", "mfrac", "msubsup"),
        ("binomial-coefficient",),
    ),
    (
        ("msubsup", "mrow", "mi"),
        ("pochhammer",),
    ),
    (
        ("mrow", "mo", "mtable"),
        ("permutation-cycle",),
    ),
    (
        ("mi", "mo", "mspace", "menclose", "mrow"),
        ("blank",),
    ),
    (
        ("mrow", "menclose", "mo", "mover"),
        ("highlight",),
    ),
    (
        ("mover", "mrow", "mn"),
        ("repeating-decimal",),
    ),
    (
        ("mo", "mspace", "mrow"),
        (
            "invisible-separator",
            "invisible-times",
        ),
    ),
    (
        ("mo", "mrow", "mi"),
        (
            "change",
            "curl",
            "divergence",
            "gradient",
            "number-of",
        ),
    ),
    (
        ("mrow", "mo", "mi"),
        (
            "max",
            "min",
        ),
    ),
    (
        ("mrow", "mo", "mtable"),
        ("determinant",),
    ),
    (
        ("mrow", "msub", "msubsup"),
        (
            "logarithm",
            "natural-logarithm",
        ),
    ),
)

_TARGET_OVERRIDES: dict[str, tuple[str, ...]] = {
    name: targets for targets, names in _TARGET_GROUPS for name in names
}

_TUPLE_FIELDS = {
    "aliases",
    "latex",
    "mathml_tags",
    "symbols",
    "examples",
}
_REPLACE_FIELDS = {
    "roles",
    "requires",
    "requires_all",
    "targets",
    "priority",
    "signatures",
    "kind",
}

_FUNCTION_TARGETS = (
    "mrow",
    "mi",
    "mo",
    "msub",
    "msup",
    "msubsup",
    "mfrac",
    "mover",
)
_CONSTANT_TARGETS = ("mi", "mo", "mn", "mrow")
_INFIX_TARGETS = ("mo", "mrow")
_PREFIX_TARGETS = ("mo", "mrow", "mover", "munder", "munderover")
_POSTFIX_TARGETS = ("mo", "msup", "mover", "mrow")
_SILENT_TARGETS = ("mo", "mspace", "mrow")
_NOFIX_TARGETS = ("mi", "mo", "mrow")


def _merge(data: dict[str, object], override: dict[str, object]) -> None:
    for key, value in override.items():
        if key in _REPLACE_FIELDS:
            data[key] = value
            continue
        if key in _TUPLE_FIELDS:
            current = tuple(data.get(key, ()))  # type: ignore[arg-type]
            data[key] = tuple(dict.fromkeys((*current, *tuple(value))))  # type: ignore[arg-type]
            continue
        data[key] = value


def _auto_aliases(name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                name,
                name.replace("-", " "),
                name.replace("-", ""),
            )
        )
    )


def _roles_for(name: str, signatures: tuple[CardSignature, ...]) -> tuple[str, ...]:
    roles = CONCEPT_ROLES.get(name)
    if roles is not None:
        return tuple(roles)
    arities = [
        signature.max_arity if signature.max_arity is not None else signature.min_arity
        for signature in signatures
    ]
    arity = max(arities, default=0)
    return tuple(f"arg{index}" for index in range(1, arity + 1))


def _targets_for(signatures: tuple[CardSignature, ...]) -> tuple[str, ...]:
    targets: list[str] = []
    for signature in signatures:
        if signature.fixity == "infix":
            candidates = _INFIX_TARGETS
        elif signature.fixity == "prefix":
            candidates = _PREFIX_TARGETS
        elif signature.fixity == "postfix":
            candidates = _POSTFIX_TARGETS
        elif signature.fixity == "silent":
            candidates = _SILENT_TARGETS
        elif signature.fixity == "nofix":
            candidates = _NOFIX_TARGETS
        elif signature.max_arity == 0:
            candidates = _CONSTANT_TARGETS
        else:
            candidates = _FUNCTION_TARGETS
        for candidate in candidates:
            if candidate not in targets:
                targets.append(candidate)
    return tuple(targets)


def _card_from_concept(name: str, concept: Concept) -> ConceptCard:
    signatures = tuple(
        CardSignature(
            fixity=signature.fixity,
            min_arity=signature.min_arity,
            max_arity=signature.max_arity,
        )
        for signature in concept.signatures
    )
    data: dict[str, object] = {
        "name": name,
        "kind": concept.kind,
        "signatures": signatures,
        "roles": _roles_for(name, signatures),
        "aliases": _auto_aliases(name),
        "latex": (),
        "mathml_tags": (),
        "symbols": (),
        "examples": (),
        "requires": (),
        "requires_all": (),
        "targets": _targets_for(signatures),
        "priority": 0,
    }
    _merge(data, _FUNCTION_CARD_OVERRIDES.get(name, {}))
    _merge(data, _STRUCTURE_OVERRIDES.get(name, {}))
    targets = _TARGET_OVERRIDES.get(name)
    if targets is not None:
        data["targets"] = targets
    return ConceptCard(**data)  # type: ignore[arg-type]


_FUNCTION_CARD_OVERRIDES = _function_cards()


def _generic_application_card() -> ConceptCard:
    return ConceptCard(
        name="generic-application",
        kind="core",
        signatures=(CardSignature(fixity="function", min_arity=2, max_arity=2),),
        roles=("name", "var"),
        aliases=(
            "generic application",
            "function application",
            "applied function",
            "f of x",
        ),
        examples=("f(x)", "f_i(x)"),
        targets=("mrow",),
        priority=2,
    )


def build_concept_cards(
    registry: ConceptRegistry | None = None,
) -> tuple[ConceptCard, ...]:
    registry = registry or get_registry()
    cards = [
        _card_from_concept(name, concept) for name, concept in registry.core.items()
    ]
    cards.extend(
        _card_from_concept(name, concept)
        for name, concept in registry.open.items()
        if name not in registry.core
    )
    cards.append(_generic_application_card())
    return tuple(sorted(cards, key=lambda card: card.name))


@lru_cache(maxsize=1)
def get_concept_cards() -> tuple[ConceptCard, ...]:
    return build_concept_cards(get_registry())


def get_concept_card(
    name: str, cards: tuple[ConceptCard, ...] | None = None
) -> ConceptCard | None:
    for card in cards or get_concept_cards():
        if card.name == name:
            return card
    return None


def _arity_label(signatures: tuple[CardSignature, ...]) -> str:
    labels: list[str] = []
    for signature in signatures:
        if signature.max_arity is None:
            label = f"{signature.min_arity}+"
        elif signature.min_arity == signature.max_arity:
            label = str(signature.min_arity)
        else:
            label = f"{signature.min_arity}-{signature.max_arity}"
        if label not in labels:
            labels.append(label)
    return "/".join(labels) or "0"


def render_concept_cards(cards: tuple[ConceptCard, ...] | list[ConceptCard]) -> str:
    lines: list[str] = []
    for card in cards:
        cues = tuple(
            dict.fromkeys(
                (
                    *card.aliases,
                    *card.latex,
                    *card.mathml_tags,
                    *card.symbols,
                )
            )
        )
        parts = [card.name, f"arity {_arity_label(card.signatures)}"]
        if card.roles:
            parts.append("roles " + ", ".join(card.roles))
        if card.targets:
            parts.append("targets " + ", ".join(card.targets))
        if cues:
            parts.append("cues " + ", ".join(cues))
        if card.examples:
            parts.append("example " + card.examples[0])
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines)
