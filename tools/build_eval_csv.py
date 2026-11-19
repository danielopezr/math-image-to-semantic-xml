#!/usr/bin/env python3
"""Create or update the evaluation CSV/YAML of one dataset area.

    python tools/build_eval_csv.py --dataset ../math-intent-mathml-dataset \\
        --area linear-algebra

The dataset (https://github.com/danielopezr/math-intent-mathml-dataset) keeps
one ``metadata/<area>.csv`` per area with the topic and complexity of every
image.  Every labelled image gets a row here, with the topic, complexity and
area taken from that metadata (it is the single source for them: a topic is
what lets the pipeline tell an inner product from an ordered pair).  Only the
per-request ``timeout`` belongs to the evaluation: rows that already exist keep
theirs, new rows get one from their complexity.  A labelled image without a
complexity in the metadata gets one estimated from the size of its label.
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
# (math-intent-mathml-dataset).  --images-dir and --mathml-dir override these.
images_dir: ${{IMAGE_TO_MATHML_DATASET_DIR:-../../math-intent-mathml-dataset}}/areas/{area}/images
mathml_dir: ${{IMAGE_TO_MATHML_DATASET_DIR:-../../math-intent-mathml-dataset}}/areas/{area}/labels

complexity_weights:
  very simple: 1.0
  simple: 1.0
  medium: 1.5
  complex: 2.0
"""

# Evaluation names used so far (results folders keep them) for each area.
DEFAULT_NAMES = {"calculus-differential-equations": "calculus-de"}


def estimate_complexity(label: Path) -> str:
    """Rough class from the number of elements in the label."""

    try:
        root = etree.fromstring(label.read_text(encoding="utf-8-sig").strip().encode())
    except (etree.XMLSyntaxError, OSError):
        return "medium"
    size = sum(1 for node in root.iter() if isinstance(node.tag, str))
    if size <= 15:
        return "very simple"
    if size <= 45:
        return "simple"
    if size <= 110:
        return "medium"
    return "complex"


# Initial per-request timeouts in seconds.  With the CPU-only qwen3-vl:4b
# setup every 300 s request timed out and was retried (simple samples finished
# within 400 s, medium ones needed up to 800 s); a generous ceiling costs
# nothing when the request finishes earlier.
TIMEOUTS = {"very simple": 600, "simple": 600, "medium": 900, "complex": 1200}


def timeout_for(complexity: str) -> int:
    return TIMEOUTS.get(complexity, TIMEOUTS["medium"])


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle) if row.get("id")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="root of math-intent-mathml-dataset"
    )
    parser.add_argument(
        "--area", required=True, help="area folder, e.g. linear-algebra"
    )
    parser.add_argument("--name", help="output file name without extension")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "eval",
    )
    args = parser.parse_args()

    area_dir = args.dataset / "areas" / args.area
    metadata = read_rows(args.dataset / "metadata" / f"{args.area}.csv")
    if not metadata:
        print(
            f"No metadata for {args.area} in {args.dataset / 'metadata'}",
            file=sys.stderr,
        )
        return 1

    name = args.name or DEFAULT_NAMES.get(args.area, args.area)
    csv_path = args.out_dir / f"{name}.csv"
    existing = read_rows(csv_path)

    rows, added = [], 0
    for sample_id, meta in metadata.items():
        if not meta.get("label"):
            continue
        label = area_dir / meta["label"]
        image = area_dir / meta["image"]
        if not label.exists() or not image.exists():
            print(
                f"{sample_id}: missing {label if not label.exists() else image}",
                file=sys.stderr,
            )
            continue
        complexity = meta.get("complexity") or estimate_complexity(label)
        previous = existing.get(sample_id)
        if previous is None:
            added += 1
        rows.append(
            {
                "id": sample_id,
                "img_path": image.name,
                "mathml_path": label.name,
                "area": meta.get("area", ""),
                "topic": meta.get("topic", ""),
                "timeout": (previous or {}).get("timeout")
                or str(timeout_for(complexity)),
                "eqn_complexity": complexity,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    yaml_path = args.out_dir / f"{name}.yaml"
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
