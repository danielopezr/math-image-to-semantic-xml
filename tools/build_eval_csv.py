#!/usr/bin/env python3
"""Create or update the evaluation CSV/YAML of one dataset area.

    python tools/build_eval_csv.py --dataset ../image-to-mathML-dataset/chopped \\
        --area algebra-lineal --name linear-algebra --area-label "Algebra lineal"

Every labelled image of the area gets a row.  Rows that already exist in the
CSV keep their hand-written ``topic``, ``timeout`` and ``eqn_complexity``; new
rows get an empty ``topic`` (to be filled in by a person: it is what lets the
pipeline tell an inner product from an ordered pair) and a complexity
estimated from the size of the label.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from lxml import etree

FIELDS = ["id", "img_path", "mathml_path", "area", "topic", "timeout", "eqn_complexity"]

YAML_TEMPLATE = """csv: {csv}
# IMAGE_TO_MATHML_DATASET_DIR points to the local copy of the dataset
# (the "chopped" folder).  --images-dir and --mathml-dir override these.
images_dir: ${{IMAGE_TO_MATHML_DATASET_DIR:-../../image-to-mathML-dataset/chopped}}/{area}
mathml_dir: ${{IMAGE_TO_MATHML_DATASET_DIR:-../../image-to-mathML-dataset/chopped}}/{area}/labels

complexity_weights:
  muy simple: 1.0
  simple: 1.0
  media: 1.5
  compleja: 2.0
"""


def estimate_complexity(label: Path) -> str:
    """Rough class from the number of elements in the label."""

    try:
        root = etree.fromstring(label.read_text(encoding="utf-8-sig").strip().encode())
    except (etree.XMLSyntaxError, OSError):
        return "media"
    size = sum(1 for node in root.iter() if isinstance(node.tag, str))
    if size <= 15:
        return "muy simple"
    if size <= 45:
        return "simple"
    if size <= 110:
        return "media"
    return "compleja"


def timeout_for(complexity: str) -> int:
    return 600 if complexity in {"media", "compleja"} else 300


def read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle) if row.get("id")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="the 'chopped' folder"
    )
    parser.add_argument(
        "--area", required=True, help="area folder, e.g. algebra-lineal"
    )
    parser.add_argument("--name", help="output file name without extension")
    parser.add_argument("--area-label", help="value of the 'area' column")
    parser.add_argument("--default-topic", default="", help="topic for new rows")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "eval",
    )
    args = parser.parse_args()

    area_dir = args.dataset / args.area
    labels = sorted((area_dir / "labels").glob("*.txt"))
    if not labels:
        print(f"No labels under {area_dir / 'labels'}", file=sys.stderr)
        return 1

    name = args.name or args.area
    csv_path = args.out_dir / f"{name}.csv"
    existing = read_existing(csv_path)
    area_label = args.area_label or args.area.replace("-", " ").capitalize()

    rows, added = [], 0
    for label in labels:
        image = area_dir / f"{label.stem}.jpeg"
        if not image.exists():
            continue
        previous = existing.get(label.stem)
        if previous is not None:
            row = {field: previous.get(field, "") for field in FIELDS}
        else:
            complexity = estimate_complexity(label)
            row = {
                "id": label.stem,
                "img_path": image.name,
                "mathml_path": label.name,
                "area": area_label,
                "topic": args.default_topic,
                "timeout": str(timeout_for(complexity)),
                "eqn_complexity": complexity,
            }
            added += 1
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    yaml_path = args.out_dir / f"{name}.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(
            YAML_TEMPLATE.format(csv=csv_path.name, area=args.area), encoding="utf-8"
        )

    missing_topic = sum(1 for row in rows if not row["topic"].strip())
    print(
        f"{csv_path.name}: {len(rows)} rows ({added} new, {len(rows) - added} kept); "
        f"{missing_topic} without topic"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
