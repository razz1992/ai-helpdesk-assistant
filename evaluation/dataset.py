"""Load local evaluation examples; LangSmith upload is a learning exercise."""
import json
from pathlib import Path
DATASET_FILE = Path(__file__).resolve().parents[1] / "data" / "evaluation_dataset.json"

def load_examples() -> list[dict]:
    return json.loads(DATASET_FILE.read_text(encoding="utf-8"))

if __name__ == "__main__":
    examples = load_examples()
    print(f"Loaded {len(examples)} evaluation examples.")
    for item in examples:
        print(item["metadata"]["case_id"], "-", item["inputs"]["question"])
