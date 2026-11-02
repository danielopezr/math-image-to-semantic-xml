"""Parser for the W3C MathML ``intent`` grammar.

Grammar (W3C MathML 4, section 5.1)::

    intent             := self-property-list | expression
    self-property-list := property+ S
    expression         := S ( term property* ) S
    term               := concept-or-literal | number | reference | application
    concept-or-literal := NCName
    number             := '-'? \\d+ ( '.' \\d+ )?
    reference          := '$' NCName
    application        := expression '(' arguments? S ')'
    arguments          := expression ( ',' expression )*
    property           := S ':' NCName
    S                  := [ \\t\\n\\r]*
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

_NCNAME_RE = re.compile(r"[A-Za-z_\u0080-\uffff][A-Za-z0-9._\-\u0080-\uffff]*")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WHITESPACE = " \t\n\r"


class IntentSyntaxError(ValueError):
    """Raised when an intent value does not match the W3C grammar."""

    def __init__(self, message: str, value: str, position: int) -> None:
        self.value = value
        self.position = position
        pointer = " " * position + "^"
        super().__init__(f"{message} at column {position + 1}\n{value}\n{pointer}")


@dataclass
class IntentNode:
    kind: str
    value: str = ""
    children: list[IntentNode] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)

    def walk(self) -> Iterator[IntentNode]:
        yield self
        for child in self.children:
            yield from child.walk()


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0
        self.length = len(text)

    def peek(self) -> str:
        if self.position >= self.length:
            return ""
        return self.text[self.position]

    def skip_ws(self) -> None:
        while self.position < self.length and self.text[self.position] in _WHITESPACE:
            self.position += 1

    def fail(self, message: str) -> IntentSyntaxError:
        return IntentSyntaxError(message, self.text, self.position)

    def parse(self) -> IntentNode:
        self.skip_ws()
        if self.peek() == ":":
            node = IntentNode(kind="self", properties=self.parse_properties())
            self.skip_ws()
            if self.position != self.length:
                raise self.fail("Unexpected text after self-property list")
            return node
        node = self.parse_expression()
        self.skip_ws()
        if self.position != self.length:
            raise self.fail("Unexpected text after intent expression")
        return node

    def parse_expression(self) -> IntentNode:
        node = self.parse_primary()
        properties = self.parse_properties()
        if properties:
            node.properties = properties

        while True:
            saved = self.position
            self.skip_ws()
            if self.peek() != "(":
                self.position = saved
                break
            self.position += 1
            arguments = self.parse_arguments()
            node = IntentNode(
                kind="application",
                children=[node, *arguments],
            )
            properties = self.parse_properties()
            if properties:
                node.properties = properties
        return node

    def parse_primary(self) -> IntentNode:
        self.skip_ws()
        character = self.peek()
        if not character:
            raise self.fail("Expected a term")

        if character == "$":
            self.position += 1
            name = self.parse_ncname("Expected a reference name after '$'")
            return IntentNode(kind="reference", value=name)

        if character.isdigit() or (character == "-" and self._next_is_digit()):
            match = _NUMBER_RE.match(self.text, self.position)
            if not match:
                raise self.fail("Invalid number")
            self.position = match.end()
            return IntentNode(kind="number", value=match.group(0))

        name = self.parse_ncname("Expected a concept, literal, number or reference")
        return IntentNode(kind="name", value=name)

    def _next_is_digit(self) -> bool:
        return (
            self.position + 1 < self.length and self.text[self.position + 1].isdigit()
        )

    def parse_ncname(self, message: str) -> str:
        match = _NCNAME_RE.match(self.text, self.position)
        if not match:
            raise self.fail(message)
        self.position = match.end()
        return match.group(0)

    def parse_properties(self) -> list[str]:
        properties: list[str] = []
        while True:
            saved = self.position
            self.skip_ws()
            if self.peek() != ":":
                self.position = saved
                break
            self.position += 1
            properties.append(self.parse_ncname("Expected a property name after ':'"))
        return properties

    def parse_arguments(self) -> list[IntentNode]:
        arguments: list[IntentNode] = []
        self.skip_ws()
        if self.peek() == ")":
            self.position += 1
            return arguments
        while True:
            arguments.append(self.parse_expression())
            self.skip_ws()
            character = self.peek()
            if character == ",":
                self.position += 1
                continue
            if character == ")":
                self.position += 1
                break
            raise self.fail("Expected ',' or ')' in argument list")
        return arguments


def parse_intent(value: str) -> IntentNode:
    if not value or not value.strip():
        raise IntentSyntaxError("Empty intent", value, 0)
    return _Parser(value).parse()


FIXITY_PROPERTIES = {
    "function",
    "prefix",
    "infix",
    "postfix",
    "silent",
    "nofix",
}
