"""Genera app/schemas/semantic_output.schema.json desde el modelo Pydantic.

Uso:  python -m app.export_schema
"""
import json
from pathlib import Path

from app.models import SemanticOutput


def main() -> None:
    target = Path(__file__).parent / "schemas" / "semantic_output.schema.json"
    schema = SemanticOutput.model_json_schema()
    target.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Esquema escrito en {target}")


if __name__ == "__main__":
    main()
