"""Starter entry point for the LangSmith evaluation exercise."""
from evaluation.dataset import load_examples

def main() -> None:
    examples = load_examples()
    print(f"Ready to evaluate {len(examples)} examples.")
    print("Next: configure LANGSMITH_API_KEY and implement dataset/experiment calls.")

if __name__ == "__main__":
    main()
