"""Validation of the final MathML + intent accessibility contract."""

from __future__ import annotations

import re

from lxml import etree

from ..models import ValidationReport
from .concepts import CORE_PROPERTIES, ConceptRegistry, get_registry
from .dom import MATHML_NS, element_children, is_element, localname, path
from .intent_parser import IntentNode, IntentSyntaxError, parse_intent

_NCNAME_RE = re.compile(r"^[A-Za-z_\u0080-\uffff][A-Za-z0-9._\-\u0080-\uffff]*$")


def _resolve_reference(owner: etree._Element, name: str) -> etree._Element | None:
    def search(parent: etree._Element) -> etree._Element | None:
        for child in parent:
            if not is_element(child):
                continue
            if child.get("arg") == name:
                return child
            if child.get("arg") is not None or child.get("intent") is not None:
                continue
            found = search(child)
            if found is not None:
                return found
        return None

    return search(owner)


class _IntentValidator:
    def __init__(
        self,
        report: ValidationReport,
        registry: ConceptRegistry,
        allow_open_concepts: bool,
        allowed_open_concepts: set[str] | None = None,
    ) -> None:
        self.report = report
        self.registry = registry
        self.allow_open = allow_open_concepts
        self.allowed_open = allowed_open_concepts
        self.referenced_args: set[str] = set()

    def validate_concept_name(self, name: str, owner: etree._Element) -> bool:
        classification = self.registry.classification(name)
        if classification == "unknown":
            self.report.error(
                "intent.unknown-concept",
                f"Unknown intent concept '{name}'",
                path(owner),
            )
            return False
        if classification != "open":
            return True
        if not self.allow_open:
            self.report.error(
                "intent.open-concept",
                f"'{name}' is not in the W3C core concept list",
                path(owner),
            )
            return False
        if self.allowed_open is not None and name not in self.allowed_open:
            allowed = ", ".join(sorted(self.allowed_open)) or "none"
            self.report.error(
                "intent.open-concept-not-allowed",
                f"'{name}' is an open concept, but the current policy only allows: "
                f"{allowed}",
                path(owner),
            )
            return False
        self.report.warning(
            "intent.open-concept",
            f"'{name}' is a recognized W3C open concept, not a core concept",
            path(owner),
        )
        return True

    def validate_properties(self, node: IntentNode, owner: etree._Element) -> None:
        for prop in node.properties:
            if prop in CORE_PROPERTIES:
                continue
            self.report.warning(
                "intent.property",
                f"Unknown intent property ':{prop}'",
                path(owner),
            )

    def validate_node(self, node: IntentNode, owner: etree._Element) -> None:
        self.validate_properties(node, owner)

        if node.kind == "self":
            return

        if node.kind == "reference":
            self.referenced_args.add(node.value)
            if _resolve_reference(owner, node.value) is None:
                self.report.error(
                    "intent.unresolved-reference",
                    f"Reference '${node.value}' has no descendant with arg='{node.value}'",
                    path(owner),
                )
            return

        if node.kind == "application":
            head, *arguments = node.children
            if head.kind == "name":
                self.validate_properties(head, owner)
            else:
                self.validate_node(head, owner)
            for argument in arguments:
                self.validate_node(argument, owner)

            if head.kind == "name":
                name = head.value
                if not self.validate_concept_name(name, owner):
                    return
                if not self.registry.accepts_arity(name, len(arguments)):
                    signatures = self.registry.get(name)
                    expected = (
                        ", ".join(
                            f"{sig.min_arity}"
                            if sig.max_arity == sig.min_arity
                            else f"{sig.min_arity}+"
                            for sig in signatures.signatures  # type: ignore[union-attr]
                        )
                        if signatures
                        else "?"
                    )
                    self.report.error(
                        "intent.arity",
                        f"Concept '{name}' does not accept {len(arguments)} argument(s); "
                        f"expected {expected}",
                        path(owner),
                    )
            return

        if node.kind == "name":
            self.validate_concept_name(node.value, owner)
            return

        # Number literals need no further validation.
        return


def validate_mathml(
    markup: str,
    *,
    allow_open_concepts: bool = True,
    allowed_open_concepts: set[str] | None = None,
    registry: ConceptRegistry | None = None,
) -> ValidationReport:
    """Validate a final MathML expression against the accessibility contract."""

    report = ValidationReport()
    registry = registry or get_registry()

    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)
    try:
        root = etree.fromstring((markup or "").encode("utf-8"), parser)
    except (etree.XMLSyntaxError, ValueError) as exc:
        report.error("structure.parse", f"Invalid XML: {exc}")
        return report

    elements = [node for node in root.iter() if is_element(node)]
    report.element_count = len(elements)

    if localname(root.tag) != "math":
        report.error(
            "structure.root",
            "The root element must be <math>",
            path(root),
        )

    if not root.tag.startswith(f"{{{MATHML_NS}}}"):
        report.warning(
            "structure.namespace",
            "The root element is not in the MathML namespace",
            path(root),
        )

    if any(localname(key) == "display" for key in root.attrib):
        report.error(
            "structure.display",
            "The <math> root must not have a display attribute",
            path(root),
        )

    top_children = element_children(root)
    if len(top_children) != 1 or localname(top_children[0].tag) != "mrow":
        report.error(
            "structure.mrow",
            "The <math> root must contain exactly one top-level <mrow>",
            path(root),
        )

    if any(localname(node.tag) == "math" and node is not root for node in elements):
        report.error(
            "structure.nested-math",
            "Nested <math> elements are not allowed",
            path(root),
        )

    intent_validator = _IntentValidator(
        report,
        registry,
        allow_open_concepts,
        allowed_open_concepts,
    )
    arg_owners: dict[str, list[etree._Element]] = {}

    for element in elements:
        arg = element.get("arg")
        if arg is not None:
            report.arg_count += 1
            arg_owners.setdefault(arg, []).append(element)
            if not _NCNAME_RE.match(arg):
                report.error(
                    "arg.invalid",
                    f"arg='{arg}' is not a valid NCName",
                    path(element),
                )

        intent = element.get("intent")
        if intent is None or not intent.strip():
            continue
        report.intent_count += 1
        try:
            ast = parse_intent(intent)
        except IntentSyntaxError as exc:
            report.error(
                "intent.syntax",
                f"Invalid intent syntax: {exc}",
                path(element),
            )
            continue
        intent_validator.validate_node(ast, element)

    for arg, owners in arg_owners.items():
        if arg not in intent_validator.referenced_args:
            for owner in owners:
                report.warning(
                    "arg.unused",
                    f"arg='{arg}' is never referenced by an intent expression",
                    path(owner),
                )

    root_mrow = top_children[0] if top_children else None
    for element in elements:
        if localname(element.tag) != "mrow":
            continue
        if element is root_mrow:
            continue
        children = element_children(element)
        if element.attrib:
            continue
        if len(children) == 1 or not children:
            report.redundant_mrows += 1
            report.warning(
                "structure.redundant-mrow",
                "Unnecessary <mrow>; unwrap it or use only when it carries "
                "intent/arg or real grouping",
                path(element),
            )

    if report.intent_count == 0:
        report.warning(
            "intent.none",
            "The expression has no intent attributes",
            path(root),
        )

    return report


def intent_errors(
    element: etree._Element,
    *,
    allow_open_concepts: bool = True,
    allowed_open_concepts: set[str] | None = None,
    registry: ConceptRegistry | None = None,
) -> list[str]:
    """Return the errors of one element's intent (used by the repair agent)."""

    intent = element.get("intent")
    if not intent or not intent.strip():
        return []
    registry = registry or get_registry()
    report = ValidationReport()
    validator = _IntentValidator(
        report,
        registry,
        allow_open_concepts,
        allowed_open_concepts,
    )
    try:
        ast = parse_intent(intent)
    except IntentSyntaxError as exc:
        return [f"Invalid intent syntax: {exc}"]
    validator.validate_node(ast, element)
    return [issue.message for issue in report.errors]


def prune_orphan_args(root: etree._Element) -> int:
    """Remove arg attributes that no intent expression references.

    The hybrid agent sometimes copies placeholder args from the prompt while
    dropping the intent that used them.  Keeping those attributes makes the
    result noisy and triggers ``arg.unused`` warnings.
    """

    tree = root.getroottree()
    used: set[str] = set()
    for owner in root.iter():
        if not is_element(owner):
            continue
        intent = owner.get("intent")
        if not intent or not intent.strip():
            continue
        try:
            ast = parse_intent(intent)
        except IntentSyntaxError:
            continue
        for node in ast.walk():
            if node.kind != "reference":
                continue
            target = _resolve_reference(owner, node.value)
            if target is not None:
                used.add(tree.getpath(target))

    removed = 0
    for element in list(root.iter()):
        if not is_element(element):
            continue
        if element.get("arg") is not None and tree.getpath(element) not in used:
            del element.attrib["arg"]
            removed += 1
    return removed


def prune_invalid_intents(
    root: etree._Element,
    *,
    allow_open_concepts: bool = True,
    allowed_open_concepts: set[str] | None = None,
    registry: ConceptRegistry | None = None,
) -> int:
    """Drop invalid intent attributes so safe ones can be rebuilt later."""

    registry = registry or get_registry()
    removed = 0
    for element in list(root.iter()):
        if not is_element(element):
            continue
        if not element.get("intent"):
            continue
        if intent_errors(
            element,
            allow_open_concepts=allow_open_concepts,
            allowed_open_concepts=allowed_open_concepts,
            registry=registry,
        ):
            del element.attrib["intent"]
            removed += 1
    return removed
