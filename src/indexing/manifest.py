import json
from pathlib import Path
from typing import Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "index_manifest.json"
)


def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {
            "documents": {}
        }

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_manifest(
    manifest: Dict[str, Any],
):
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )