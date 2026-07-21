from pathlib import Path

from deplab.result_repair import repair_result_file, summary_json


if __name__ == "__main__":
    result = repair_result_file(Path("outputs/expanded-development-results.jsonl"))
    print(summary_json(result))
