#!/usr/bin/env python3
"""Build the local W3C intent concept registries used by the validator.

The generated JSON files are committed so the runtime never needs network
access.  Run this script only when the upstream W3C documents change:

    python tools/build_concept_registry.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from lxml import html

CORE_URL = "https://w3c.github.io/mathml-docs/intent-core-concepts/"
OPEN_URL = "https://w3c.github.io/mathml-docs/intent-open-concepts/"

FIXITY_ALIASES = {
    "": "function",
    "*": "function",
    "function": "function",
    "function*": "function",
    "operator": "function",
    "operator*": "function",
    "indexed": "function",
    "indexed function": "function",
    "indexed function*": "function",
    "???": "function",
    "???*": "function",
    "infix": "infix",
    "infix*": "infix",
    "prefix": "prefix",
    "prefix*": "prefix",
    "postfix": "postfix",
    "postfix*": "postfix",
    "silent": "silent",
    "nofix": "nofix",
}

DEFAULT_FIXITY_ARITY = {
    "function": (0, None),
    "prefix": (1, 1),
    "infix": (2, 2),
    "postfix": (1, 1),
    "silent": (0, 0),
    "nofix": (0, 0),
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "image-to-mathml/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8")


def normalize_fixity(raw: str) -> str:
    value = " ".join(raw.split()).strip().lower()
    return FIXITY_ALIASES.get(value, "function")


def parse_arity(raw: str) -> tuple[int, int | None]:
    text = " ".join(raw.split())
    if not text or "$" not in text:
        return (0, 0)
    count = text.count("$")
    if "..." in text or "…" in text:
        return (count, None)
    return (count, count)


def add_signature(
    concepts: dict, name: str, fixity: str, arity: tuple[int, int | None]
) -> None:
    if not name:
        return
    entry = concepts.setdefault(name, {"signatures": []})
    signature = {"fixity": fixity, "min_arity": arity[0], "max_arity": arity[1]}
    if signature not in entry["signatures"]:
        entry["signatures"].append(signature)


def parse_template_page(markup: str, kind: str) -> dict:
    document = html.fromstring(markup)
    concepts: dict = {}
    for row in document.xpath('//tr[contains(@class, "row")]'):
        anchor = row.xpath('.//a[contains(@class, "self")]')
        if not anchor:
            continue
        name = " ".join(anchor[0].text_content().split())
        cells = row.xpath("./td")
        if not name or len(cells) < 3:
            continue
        arguments = " ".join(cells[1].text_content().split())
        property_text = " ".join(cells[2].text_content().split())
        add_signature(
            concepts, name, normalize_fixity(property_text), parse_arity(arguments)
        )

    for name, entry in concepts.items():
        entry["kind"] = kind
    return concepts


ZERO_OR_MORE = {"list", "tuple"}


def parse_default_fixity(markup: str, concepts: dict) -> None:
    document = html.fromstring(markup)
    sections = document.xpath('//h2[contains(text(), "Core Concept Default Fixity")]')
    if not sections:
        raise SystemExit("Could not find the default fixity section")
    section = sections[0]
    definition_list = None
    for sibling in section.itersiblings():
        if sibling.tag == "h2":
            break
        if sibling.tag == "dl":
            definition_list = sibling
            break
    if definition_list is None:
        raise SystemExit("Could not find the default fixity definition list")

    current_fixity = "function"
    for child in definition_list:
        if child.tag == "dt":
            bold = child.xpath("./b")
            current_fixity = normalize_fixity(
                " ".join((bold[0] if bold else child).text_content().split())
            )
        elif child.tag == "dd":
            arity = DEFAULT_FIXITY_ARITY.get(current_fixity, (0, None))
            for anchor in child.xpath('.//a[contains(@class, "self")]'):
                name = " ".join(anchor.text_content().split())
                add_signature(concepts, name, current_fixity, arity)


def merge(core: dict, open_concepts: dict) -> dict:
    merged = {name: dict(entry) for name, entry in core.items()}
    for name, entry in open_concepts.items():
        if name in merged:
            for signature in entry["signatures"]:
                if signature not in merged[name]["signatures"]:
                    merged[name]["signatures"].append(signature)
        else:
            merged[name] = dict(entry)
    return merged


def write_registry(path: Path, concepts: dict, source: str, note: str) -> None:
    payload = {
        "schema_version": 1,
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "concepts": dict(sorted(concepts.items())),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-url", default=CORE_URL)
    parser.add_argument("--open-url", default=OPEN_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src"
        / "image_to_mathml"
        / "data",
    )
    args = parser.parse_args()

    print(f"Fetching {args.core_url} ...", file=sys.stderr)
    core_markup = fetch(args.core_url)
    core = parse_template_page(core_markup, "core")
    print(f"  {len(core)} core concepts from templates", file=sys.stderr)
    parse_default_fixity(core_markup, core)
    print(f"  {len(core)} core concepts after default fixity merge", file=sys.stderr)

    print(f"Fetching {args.open_url} ...", file=sys.stderr)
    open_concepts = parse_template_page(fetch(args.open_url), "open")
    print(f"  {len(open_concepts)} open concepts", file=sys.stderr)

    for name in ZERO_OR_MORE:
        if name in core:
            core[name]["signatures"] = [
                {"fixity": "function", "min_arity": 0, "max_arity": None}
            ]
    for entry in core.values():
        entry["kind"] = "core"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_registry(
        args.output_dir / "core_concepts.json",
        core,
        args.core_url,
        "W3C MathML intent core concepts (core list + default fixity table).",
    )
    write_registry(
        args.output_dir / "open_concepts.json",
        merge(open_concepts, core),
        args.open_url,
        "W3C MathML intent open concepts. The core concepts are included too so "
        "this file can be used as the complete recognized-name registry.",
    )
    print(f"Wrote registries under {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
